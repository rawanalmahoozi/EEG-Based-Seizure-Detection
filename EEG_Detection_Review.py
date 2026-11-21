import os
import io
import json
import shutil
import textwrap
import urllib.request
from datetime import datetime
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import streamlit as st
from keras.models import load_model

st.set_page_config(page_title="EEG Seizure Review Assistant", layout="wide")

config = {
    "model_version": "unknown (no final_config.json found)",
    "model_file": "model_cnn_8s_final.keras",
    "mean_file": "cnn_mean_8s.npy",
    "std_file": "cnn_std_8s.npy",
    "threshold": 0.8,
    "min_event_sec": 8,
}
if os.path.exists("final_config.json"):
    with open("final_config.json") as f:
        config.update(json.load(f))

threshold = config["threshold"]
min_sec = config["min_event_sec"]

window_sec = 8
step_sec = 4
expected_fs = 256
downsample = 2
gap = 8
merge_gap = 30
strong_sec = 12

channels = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
            'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
            'FZ-CZ', 'CZ-PZ']

summary_roots = ["demo_patient/", "EEG/chb-mit-scalp-eeg-database-1.0.0/"]

decision_options = ["Not reviewed", "Confirmed seizure", "Not a seizure (reject)", "Needs more review"]
missed_text = "Added by specialist (missed by the model)"
disclaimer = ("must not be used for diagnosis or treatment. It only highlights moments of an EEG recording "
              "that may deserve a closer look. Every decision belongs to a qualified specialist.")

if "started" not in st.session_state:
    st.session_state["started"] = True
    shutil.rmtree("temp_files", ignore_errors=True)
if "reviews" not in st.session_state:
    st.session_state["reviews"] = {}
if "missed" not in st.session_state:
    st.session_state["missed"] = []


@st.cache_resource
def load_everything(model_file, mean_file, std_file):
    model = load_model(model_file)
    mean = np.load(mean_file)
    std = np.load(std_file)
    return model, mean, std


missing = []
for file_name in [config["model_file"], config["mean_file"], config["std_file"]]:
    if not os.path.exists(file_name):
        missing.append(file_name)
if len(missing) > 0:
    st.error("These files must be in the same folder as final.py: " + ", ".join(missing)
             + ". Run the final notebook first (it saves them).")
    st.stop()

model, mean, std = load_everything(config["model_file"], config["mean_file"], config["std_file"])


def save_upload(uploaded_file):
    os.makedirs("temp_files", exist_ok=True)
    path = "temp_files/" + uploaded_file.name.replace("/", "_")
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    return path


def get_channel_index(raw):
    idx = []
    for c in channels:
        if c in raw.ch_names:
            idx.append(raw.ch_names.index(c))
        elif c + "-0" in raw.ch_names:
            idx.append(raw.ch_names.index(c + "-0"))
        else:
            return None
    return idx


def read_signal(path, start_s, end_s):
    raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
    idx = get_channel_index(raw)
    fs = int(raw.info["sfreq"])
    stop = min(int(end_s * fs), raw.n_times)
    data = raw.get_data(picks=idx, start=int(start_s * fs), stop=stop) * 1e6
    return data, fs


def predict_file(path):
    raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
    idx = get_channel_index(raw)
    fs = int(raw.info["sfreq"])
    if idx is None or fs != expected_fs:
        return None

    size = window_sec * fs
    step = step_sec * fs
    n = (raw.n_times - size) // step + 1
    if n < 1:
        return None

    proba = []

    chunk = 256
    for first in range(0, n, chunk):
        last = min(n, first + chunk)
        data = raw.get_data(picks=idx, start=first * step, stop=(last - 1) * step + size) * 1e6

        windows = []
        for i in range(last - first):
            window = data[:, i * step : i * step + size][:, ::downsample].T
            windows.append(window)

        windows = np.array(windows).astype("float32")
        windows = (windows - mean) / std
        proba.extend(model.predict(windows, batch_size=256, verbose=0)[:, 1])

    smooth = []
    for i in range(len(proba)):
        a = max(0, i - 1)
        b = min(len(proba), i + 2)
        smooth.append(np.mean(proba[a:b]))

    return np.array(smooth)


def find_events(proba):
    events = []
    ev_start = None
    ev_end = None

    for i in range(len(proba)):
        if proba[i] > threshold:
            s = i * step_sec
            e = s + window_sec
            if ev_start is None:
                ev_start = s
                ev_end = e
            elif s - ev_end <= gap:
                ev_end = e
            else:
                events.append([ev_start, ev_end])
                ev_start = s
                ev_end = e

    if ev_start is not None:
        events.append([ev_start, ev_end])

    kept = []
    for ev in events:
        if ev[1] - ev[0] >= min_sec:
            kept.append(ev)
    return kept


def find_segments(proba):
    events = find_events(proba)

    groups = []
    for s, e in events:
        if len(groups) > 0 and s - groups[-1][1] <= merge_gap:
            groups[-1][1] = e
            groups[-1][2] = groups[-1][2] + (e - s)
        else:
            groups.append([s, e, e - s])

    segments = []
    for s, e, flagged in groups:
        first_window = int(s // step_sec)
        last_window = int((e - window_sec) // step_sec)
        peak = float(max(proba[first_window : last_window + 1]))

        if flagged >= strong_sec:
            strength = "strong signal"
        else:
            strength = "weak signal"
        segments.append({"start": s, "end": e, "flagged_s": flagged, "strength": strength, "peak": peak})
    return segments


def seg_id(name, seg):
    return name + "|" + str(seg["start"]) + "|" + str(seg["end"])


def fmt(sec):
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return str(h) + ":" + str(m).zfill(2) + ":" + str(s).zfill(2)


def get_start_time(path):
    raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
    return raw.info["meas_date"]


def second_of_day(start_dt):
    return start_dt.hour * 3600 + start_dt.minute * 60 + start_dt.second


def clock_text(start_dt, sec):
    if start_dt is None:
        return ""
    t = (second_of_day(start_dt) + int(sec)) % 86400
    return str(t // 3600).zfill(2) + ":" + str((t % 3600) // 60).zfill(2) + ":" + str(t % 60).zfill(2)


def read_summary(path):
    with open(path) as f:
        text = f.read()

    result = {}
    blocks = text.split("File Name:")

    for b in blocks[1:]:
        lines = b.strip().split("\n")
        file_name = lines[0].strip()

        starts = []
        ends = []
        for line in lines:
            if "Seizure" in line and "Start Time" in line:
                starts.append(int(line.split(":")[1].split()[0]))
            if "Seizure" in line and "End Time" in line:
                ends.append(int(line.split(":")[1].split()[0]))

        result[file_name] = list(zip(starts, ends))
    return result


def find_real_seizures(name):
    file_name = os.path.basename(name)
    patient = file_name[:5]
    for r in summary_roots:
        path = r + patient + "/" + patient + "-summary.txt"
        if os.path.exists(path):
            return read_summary(path).get(file_name, [])
    return None


def get_review(sid):
    if sid in st.session_state["reviews"]:
        return st.session_state["reviews"][sid]
    return {"decision": "Not reviewed", "note": ""}


def save_review(sid):
    st.session_state["reviews"][sid] = {"decision": st.session_state["decision_" + sid],
                                        "note": st.session_state["note_" + sid]}


def final_status(decision):
    if decision == "Confirmed seizure":
        return "Confirmed by specialist"
    if decision == "Not a seizure (reject)":
        return "Rejected by specialist (false alarm)"
    if decision == "Needs more review":
        return "Needs more review"
    return "Model flag only (not reviewed)"


def agreement(decision):
    if decision == "Confirmed seizure":
        return "specialist agrees with the model"
    if decision == "Not a seizure (reject)":
        return "specialist disagrees with the model"
    return "no decision yet"


def make_rows(in_folder, folder, all_segments):
    rows = []
    for name in in_folder:
        for seg in all_segments[name]:
            review = get_review(seg_id(name, seg))
            label = ""
            if review["decision"] == "Confirmed seizure":
                label = 1
            if review["decision"] == "Not a seizure (reject)":
                label = 0
            rows.append([folder, os.path.basename(name), fmt(seg["start"]), fmt(seg["end"]),
                         seg["end"] - seg["start"], seg["strength"], round(seg["peak"], 2), threshold,
                         review["decision"], review["note"], final_status(review["decision"]),
                         agreement(review["decision"]), config["model_version"], label])

    for item in st.session_state["missed"]:
        if item["recording"] in in_folder:
            rows.append([folder, os.path.basename(item["recording"]), fmt(item["start"]), fmt(item["end"]),
                         item["end"] - item["start"], "", "", threshold,
                         "Confirmed seizure", "", missed_text, "model missed it", config["model_version"], 1])

    columns = ["patient", "recording", "start_in_recording", "end_in_recording", "duration_s",
               "model_signal", "peak_probability", "threshold", "specialist_decision", "note",
               "final_status", "model_vs_specialist", "model_version", "label"]
    return pd.DataFrame(rows, columns=columns)


def add_missed(name, start, end):
    st.session_state["missed"].append({"recording": name, "start": int(start), "end": int(end)})


def clear_everything():
    shutil.rmtree("temp_files", ignore_errors=True)
    st.session_state.clear()


def explain_event(seg, start_dt, n_in_recording):
    duration = seg["end"] - seg["start"]
    text = ("The model flagged " + fmt(seg["start"]) + " to " + fmt(seg["end"]) + " (" + str(duration) + " s)"
            + ". Its seizure probability was above the threshold of " + str(threshold) + " for "
            + str(seg["flagged_s"]) + " s and reached " + str(round(seg["peak"], 2)) + " at its highest. ")
    if seg["strength"] == "strong signal":
        text = text + "This is a strong signal (at least " + str(strong_sec) + " s above the threshold). "
    else:
        text = text + "This is a weak signal (less than " + str(strong_sec) + " s above the threshold), so it is more likely to be a false alarm. "
    if n_in_recording > 1:
        text = text + "There are " + str(n_in_recording - 1) + " other possible seizure(s) in this recording. "
    if start_dt is not None:
        text = text + "It happened around " + clock_text(start_dt, seg["start"]) + " (recording clock). "
    text = text + "This is a prompt for review, not a diagnosis: a specialist must look at the EEG."
    return text


def rewrite_with_llm(facts):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    prompt = ("Rewrite the following facts about one flagged EEG segment in 3 short, plain sentences for a "
              "specialist. Use ONLY these facts. Do not add facts, do not give a diagnosis, do not suggest "
              "treatment.\n\n" + facts)
    body = json.dumps({"model": "claude-sonnet-5-5", "max_tokens": 300,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    request = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, headers={
        "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=20) as answer:
            return json.load(answer)["content"][0]["text"]
    except Exception:
        return None


def plot_signal(ax, path, start, end, flagged, real):
    data, fs = read_signal(path, start, end)
    t = np.arange(data.shape[1]) / fs + start

    for i in range(data.shape[0]):
        ax.plot(t[::2], data[i][::2] + i * 300, linewidth=0.5, color="tab:blue")
        ax.text(t[0], i * 300, channels[i], fontsize=7)

    for fs_, fe_ in flagged:
        if fs_ < end and fe_ > start:
            ax.axvspan(max(fs_, start), min(fe_, end), color="red", alpha=0.15)
    for rs_, re_ in real:
        if rs_ < end and re_ > start:
            ax.axvspan(max(rs_, start), min(re_, end), color="green", alpha=0.15)

    if len(real) > 0:
        ax.set_title("Red = flagged by the model.   Green = real seizure (demo only).")
    ax.invert_yaxis()
    ax.set_xlabel("Time (s) from the start of the recording")
    ax.set_yticks([])


def show_signal(path, start, end, flagged, real):
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    plot_signal(ax2, path, start, end, flagged, real)
    st.pyplot(fig2)
    plt.close(fig2)


def move_viewer(name, new_start):
    st.session_state["browse_pos_" + name] = new_start


def make_pdf(folder, in_folder, all_segments, paths, hours):
    all_rows = make_rows(in_folder, folder, all_segments)
    rows = all_rows[all_rows["final_status"] != missed_text]
    n_missed = len(all_rows) - len(rows)
    buffer = io.BytesIO()

    with PdfPages(buffer) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.94, "EEG Seizure Review Report", fontsize=20, weight="bold")
        fig.text(0.08, 0.915, "Created " + datetime.now().strftime("%Y-%m-%d %H:%M"), fontsize=9)

        counts = rows["specialist_decision"].value_counts()
        lines = ["Folder / patient: " + folder,
                 "Recordings: " + str(len(in_folder)) + "     Hours of EEG: " + str(round(hours, 1)),
                 "Possible seizures flagged by the model: " + str(len(rows)),
                 "   confirmed by the specialist: " + str(int(counts.get("Confirmed seizure", 0))),
                 "   rejected (false alarms): " + str(int(counts.get("Not a seizure (reject)", 0))),
                 "   needs more review: " + str(int(counts.get("Needs more review", 0))),
                 "   not reviewed yet: " + str(int(counts.get("Not reviewed", 0))),
                 "Seizures added by the specialist (missed by the model): " + str(n_missed),
                 "",
                 "Model: " + str(config["model_version"]),
                 "Window " + str(window_sec) + " s, step " + str(step_sec) + " s, threshold " + str(threshold)
                 + ", minimum event " + str(min_sec) + " s"]
        y = 0.87
        for line in lines:
            fig.text(0.08, y, line, fontsize=10)
            y = y - 0.025

        fig.text(0.08, y - 0.02, "Important", fontsize=12, weight="bold")
        fig.text(0.08, y - 0.05, textwrap.fill(disclaimer, 95), fontsize=9, va="top")
        warning_text = "The model produces false alarms. A flagged moment is a question for the specialist, not an answer."
        fig.text(0.08, y - 0.17, textwrap.fill(warning_text, 95), fontsize=9, va="top")

        y = y - 0.25
        fig.text(0.08, y, "Recordings with possible seizures", fontsize=12, weight="bold")
        y = y - 0.028
        flagged_names = []
        for name in in_folder:
            if len(all_segments[name]) > 0:
                flagged_names.append(name)
        if len(flagged_names) == 0:
            fig.text(0.08, y, "None", fontsize=8)
        for name in flagged_names[:20]:
            n_confirmed = 0
            for seg in all_segments[name]:
                if get_review(seg_id(name, seg))["decision"] == "Confirmed seizure":
                    n_confirmed = n_confirmed + 1
            fig.text(0.08, y, os.path.basename(name) + ":  " + str(len(all_segments[name]))
                     + " possible seizure(s), " + str(n_confirmed) + " confirmed", fontsize=8)
            y = y - 0.016
        if len(flagged_names) > 20:
            fig.text(0.08, y, "... and " + str(len(flagged_names) - 20) + " more recordings (see the table on the next pages)", fontsize=8)
        pdf.savefig(fig)
        plt.close(fig)

        if len(rows) == 0:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.08, 0.9, "Nothing was flagged in this folder.", fontsize=12)
            pdf.savefig(fig)
            plt.close(fig)
        per_page = 22
        for first in range(0, len(rows), per_page):
            part = rows.iloc[first : first + per_page][["recording", "start_in_recording", "duration_s", "model_signal",
                                                         "peak_probability", "specialist_decision", "note"]].astype(str)
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.axis("off")
            shown = ax.table(cellText=part.values, colLabels=["recording", "start", "duration (s)", "signal",
                                                              "peak prob.", "specialist decision", "note"],
                             loc="upper center", cellLoc="center")
            shown.auto_set_font_size(False)
            shown.set_fontsize(7)
            shown.auto_set_column_width(list(range(7)))
            ax.set_title("Possible seizures (" + str(first + 1) + " to " + str(min(first + per_page, len(rows))) + " of "
                         + str(len(rows)) + ")   |   research prototype, not for diagnosis", fontsize=9)
            pdf.savefig(fig)
            plt.close(fig)

        pages = []
        for name in in_folder:
            for seg in all_segments[name]:
                decision = get_review(seg_id(name, seg))["decision"]
                if decision in ["Confirmed seizure", "Needs more review"]:
                    pages.append([name, seg, decision])
        for name, seg, decision in pages[:10]:
            start = max(0, seg["start"] - 10)
            end = min(seg["end"] + 10, seg["start"] + 120)
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            plot_signal(ax, paths[name], start, end, [[seg["start"], seg["end"]]], [])
            ax.set_title(os.path.basename(name) + "   " + fmt(seg["start"]) + " to " + fmt(seg["end"])
                         + "   model: " + seg["strength"] + ", peak " + str(round(seg["peak"], 2))
                         + "   specialist: " + decision, fontsize=9)
            pdf.savefig(fig)
            plt.close(fig)

    return buffer.getvalue()


st.title("EEG Seizure Review Assistant")
st.caption("Research prototype. Not a medical diagnostic tool. "
           "It only highlights moments that may need a closer look by a specialist.")

mode = st.radio("How do you want to upload?", ["A whole folder", "Individual files"], horizontal=True)

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0
key = str(st.session_state["uploader_key"])

sources = {}

if mode == "A whole folder":
    uploaded = st.file_uploader("Choose a folder (only the EDF recordings are used)",
                                accept_multiple_files="directory", key="folder_" + key)
else:
    uploaded = st.file_uploader("Choose one or more EDF files",
                                type=["edf"], accept_multiple_files=True, key="files_" + key)
if uploaded:
    for f in uploaded:
        if f.name.lower().endswith(".edf"):
            sources[f.name] = f

if len(sources) > 0:
    st.write(str(len(sources)) + " EDF recordings found")

if st.button("Analyse recordings") and len(sources) > 0:
    shutil.rmtree("temp_files", ignore_errors=True)
    st.session_state["probas"] = {}
    st.session_state["paths"] = {}
    st.session_state["starts"] = {}
    st.session_state.pop("selected", None)
    st.session_state.pop("folder", None)
    bar = st.progress(0)
    for k, name in enumerate(sources):
        path = save_upload(sources[name])
        st.session_state["paths"][name] = path
        st.session_state["probas"][name] = predict_file(path)
        st.session_state["starts"][name] = get_start_time(path)
        bar.progress((k + 1) / len(sources))

    st.session_state["uploader_key"] = st.session_state["uploader_key"] + 1
    st.rerun()

if "probas" in st.session_state:
    probas = st.session_state["probas"]
    paths = st.session_state["paths"]
    starts = st.session_state["starts"]
    st.caption("The recordings below are analysed. To analyse other recordings, choose them above.")

    names = []
    skipped = []
    for name in probas:
        if probas[name] is None:
            skipped.append(name)
        else:
            names.append(name)

    if len(skipped) > 0:
        st.warning("These files could not be analysed (missing channels, unsupported sampling rate or too short): "
                   + ", ".join(skipped))

    all_segments = {}
    folder_of = {}
    folders = []
    for name in names:
        all_segments[name] = find_segments(probas[name])
        folder = os.path.dirname(name)
        if folder == "":
            folder = "(uploaded files)"
        folder_of[name] = folder
        if folder not in folders:
            folders.append(folder)

    if len(folders) == 1:
        current_folder = folders[0]
    else:
        current_folder = st.session_state.get("folder")

    st.divider()

    if current_folder is None:
        st.subheader("Folders")
        per_row = 4
        for row_start in range(0, len(folders), per_row):
            cols = st.columns(per_row)
            for j in range(per_row):
                k = row_start + j
                if k < len(folders):
                    folder = folders[k]
                    n_rec = 0
                    n_seiz = 0
                    hours = 0
                    for name in names:
                        if folder_of[name] == folder:
                            n_rec = n_rec + 1
                            n_seiz = n_seiz + len(all_segments[name])
                            hours = hours + len(probas[name]) * step_sec / 3600
                    with cols[j]:
                        with st.container(border=True):
                            st.markdown("**" + folder + "**")
                            st.caption(str(n_rec) + " recordings, " + str(round(hours, 1)) + " hours of EEG")
                            if n_seiz > 0:
                                st.markdown(":red[● " + str(n_seiz) + " possible seizures]")
                            else:
                                st.markdown(":green[✓ Nothing flagged]")
                            if st.button("Open", key="open_" + folder):
                                st.session_state["folder"] = folder
                                st.session_state.pop("selected", None)
                                st.rerun()

    else:
        if len(folders) > 1:
            if st.button("← All folders"):
                st.session_state.pop("folder", None)
                st.session_state.pop("selected", None)
                st.rerun()

        in_folder = []
        for name in names:
            if folder_of[name] == current_folder:
                in_folder.append(name)

        order_mode = st.radio("Order of the recordings", ["Review priority", "Recording time"], horizontal=True,
                              help="Review priority: the recordings where the model was most sure come first. "
                                   "This is only an order for looking at the recordings, not a medical severity score.")

        def highest_peak(name):
            peak = 0
            for seg in all_segments[name]:
                peak = max(peak, seg["peak"])
            return peak

        def start_text(name):
            if starts[name] is None:
                return ""
            return str(starts[name])

        if order_mode == "Recording time":
            in_folder.sort(key=start_text)
        else:
            in_folder.sort(key=highest_peak, reverse=True)

        hours = 0
        with_flags = 0
        total_seiz = 0
        reviewed = 0
        for name in in_folder:
            hours = hours + len(probas[name]) * step_sec / 3600
            if len(all_segments[name]) > 0:
                with_flags = with_flags + 1
            total_seiz = total_seiz + len(all_segments[name])
            for seg in all_segments[name]:
                if get_review(seg_id(name, seg))["decision"] != "Not reviewed":
                    reviewed = reviewed + 1

        st.subheader("Folder: " + current_folder)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Recordings", len(in_folder))
        c2.metric("Hours of EEG", round(hours, 1))
        c3.metric("Recordings with possible seizures", with_flags)
        c4.metric("Possible seizures", total_seiz)
        c5.metric("Reviewed", str(reviewed) + " of " + str(total_seiz))

        period_names = ["Night (00:00-06:00)", "Morning (06:00-12:00)",
                        "Afternoon (12:00-18:00)", "Evening (18:00-24:00)"]
        period_flags = [0, 0, 0, 0]
        for name in in_folder:
            if starts[name] is None:
                continue
            first = second_of_day(starts[name])
            for seg in all_segments[name]:
                hour = ((first + int(seg["start"])) // 3600) % 24
                period_flags[hour // 6] = period_flags[hour // 6] + 1

        flagged_total = sum(period_flags)
        if flagged_total > 0:
            st.subheader("At what time of the day?")
            top = period_flags.index(max(period_flags))
            st.write("Most possible seizures are in the **" + period_names[top] + "**: "
                     + str(period_flags[top]) + " of " + str(flagged_total) + ".")
            if flagged_total < 5:
                st.caption("Only " + str(flagged_total) + " possible seizures, so this is too few to call it a pattern.")

            cols = st.columns(4)
            for k in range(4):
                cols[k].metric(period_names[k], period_flags[k])
            st.caption("These are possible seizures flagged by the model (some are false alarms).")

        st.write("Every recording is listed. Click **Review** to look at one.")
        per_row = 4
        for row_start in range(0, len(in_folder), per_row):
            cols = st.columns(per_row)
            for j in range(per_row):
                k = row_start + j
                if k < len(in_folder):
                    name = in_folder[k]
                    n_seiz = len(all_segments[name])
                    n_done = 0
                    for seg in all_segments[name]:
                        if get_review(seg_id(name, seg))["decision"] != "Not reviewed":
                            n_done = n_done + 1
                    with cols[j]:
                        with st.container(border=True):
                            st.markdown("**" + os.path.basename(name) + "**")
                            st.caption(str(round(len(probas[name]) * step_sec / 3600, 1)) + " hours of EEG")
                            if n_seiz > 0:
                                st.markdown(":red[● " + str(n_seiz) + " possible seizures]")
                                st.caption(str(n_done) + " reviewed")
                            else:
                                st.markdown(":green[✓ Nothing flagged]")
                            if st.button("Review", key="review_" + name):
                                st.session_state["selected"] = name

        selected = st.session_state.get("selected")
        if selected not in in_folder:
            selected = in_folder[0]
            for name in in_folder:
                if len(all_segments[name]) > 0:
                    selected = name
                    break

        segs = all_segments[selected]
        start_dt = starts[selected]

        st.divider()
        st.subheader(os.path.basename(selected))
        info = str(round(len(probas[selected]) * step_sec / 3600, 2)) + " hours of EEG"
        if start_dt is not None:
            info = info + ", starts at " + clock_text(start_dt, 0)
        st.caption(info + ".")

        real = find_real_seizures(selected)
        real_ranges = []
        if real is not None:
            if st.checkbox("Show the real seizures (demo only, from the summary file)"):
                real_ranges = real
                if len(real) == 0:
                    st.info("The summary file says there is no seizure in this recording.")
                for ts, te in real:
                    result = "missed"
                    for seg in segs:
                        if seg["start"] < te and seg["end"] > ts:
                            result = "found"
                    st.write("Real seizure " + fmt(ts) + " to " + fmt(te) + ": " + result + " by the model")

        if len(segs) == 0:
            st.success("Nothing was flagged in this recording.")
        else:
            st.write(str(len(segs)) + " possible seizure(s) in this recording.")

            labels = []
            for i in range(len(segs)):
                seg = segs[i]
                label = ("#" + str(i + 1) + "   " + fmt(seg["start"]) + " to " + fmt(seg["end"]) + "   ("
                         + str(seg["end"] - seg["start"]) + " s, " + seg["strength"] + ")")
                if start_dt is not None:
                    label = label + "   at " + clock_text(start_dt, seg["start"])
                labels.append(label)

            pick = st.selectbox("Possible seizure", labels, key="seg_pick_" + selected)
            seg = segs[labels.index(pick)]
            sid = seg_id(selected, seg)


            start = max(0, seg["start"] - 10)
            end = min(seg["end"] + 10, seg["start"] + 120)
            if end < seg["end"] + 10:
                st.caption("This segment is long, so only its first 2 minutes are shown.")

            flagged_ranges = []
            for other in segs:
                flagged_ranges.append([other["start"], other["end"]])
            show_signal(paths[selected], start, end, flagged_ranges, real_ranges)

            st.markdown("#### Specialist review")
            review = get_review(sid)
            decision = st.radio("Your decision", decision_options, horizontal=True,
                                index=decision_options.index(review["decision"]), key="decision_" + sid,
                                on_change=save_review, args=(sid,))
            st.text_area("Notes", value=review["note"], key="note_" + sid,
                         on_change=save_review, args=(sid,))

            st.write("**Model:** " + seg["strength"] + ", highest probability " + str(round(seg["peak"], 2))
                     + ".   **Specialist:** " + decision + ".   **Final status:** " + final_status(decision) + ".")

        st.divider()
        st.subheader("Review results and report")
        rows = make_rows(in_folder, current_folder, all_segments)
        e1, e2, e3 = st.columns(3)
        pdf_key = "pdf_" + current_folder
        if e2.button("Create PDF report"):
            st.session_state[pdf_key] = make_pdf(current_folder, in_folder, all_segments, paths, hours)
        if pdf_key in st.session_state:
            e3.download_button("Download the PDF report", st.session_state[pdf_key],
                               file_name=current_folder.replace("/", "_") + "_report.pdf", mime="application/pdf")
