import glob
import hashlib
import io
import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import re
import shutil
import sqlite3
import streamlit as st
from contextlib import closing
from datetime import datetime
from keras.models import load_model
from matplotlib.patches import Patch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.platypus import (HRFlowable, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)
from xml.sax.saxutils import escape



model_file = "model_cnn_8s_final.keras"
mean_file = "cnn_mean_8s.npy"
std_file = "cnn_std_8s.npy"
threshold = 0.8
min_sec = 8

window_sec = 8            # the model looks at 8-second windows
step_sec = 4              # a new window starts every 4 seconds
expected_fs = 256         # sampling rate of the recordings (Hz)
downsample = 2            # 256 Hz -> 128 Hz, like in training
gap = 8                   # flagged windows closer than 8 s are merged into one event
merge_gap = 30            # events closer than 30 s are shown as ONE possible seizure
strong_sec = 12           # a possible seizure with at least 12 s flagged is a "strong signal"

channels = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
            'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
            'FZ-CZ', 'CZ-PZ']


summary_roots = ["demo_patient/", "EEG/chb-mit-scalp-eeg-database-1.0.0/"]

decision_options = ["Not reviewed", "Confirmed seizure", "Not a seizure (reject)", "Needs more review"]

decision_label = {"Not reviewed": "Not yet reviewed", "Confirmed seizure": "Specialist-confirmed seizure",
                  "Not a seizure (reject)": "Marked as not a seizure", "Needs more review": "Requires further specialist review"}

review_db_file = "reviews.db"     


patient_id_patterns = [r"(chb\d+)[_-]"]



def fmt(sec):
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return str(h) + ":" + str(m).zfill(2) + ":" + str(s).zfill(2)

def second_of_day(start_dt):
    return start_dt.hour * 3600 + start_dt.minute * 60 + start_dt.second

def clock_text(start_dt, sec):
    if start_dt is None:
        return ""
    t = (second_of_day(start_dt) + int(sec)) % 86400
    hour = t // 3600
    suffix = "AM"
    if hour >= 12:
        suffix = "PM"
    hour12 = hour % 12
    if hour12 == 0:
        hour12 = 12
    return str(hour12) + ":" + str((t % 3600) // 60).zfill(2) + ":" + str(t % 60).zfill(2) + " " + suffix

def duration_text(hours):
    minutes = int(round(hours * 60))
    if minutes < 60:
        return str(minutes) + " min"
    return str(minutes // 60) + " h " + str(minutes % 60).zfill(2) + " min"

def hours_of(proba):
    return len(proba) * step_sec / 3600

def flag_category(seg):
    if seg["strength"] == "strong signal":
        return "Longer detection"
    return "Shorter detection"

def chip(text, colour):
    return '<span class="status-chip status-' + colour + '">' + text + '</span>'


@st.cache_resource
def load_everything(model_file, mean_file, std_file):
    model = load_model(model_file)
    mean = np.load(mean_file)
    std = np.load(std_file)
    return model, mean, std

missing = []
for file_name in [model_file, mean_file, std_file]:
    if not os.path.exists(file_name):
        missing.append(file_name)
if len(missing) > 0:
    st.error("These files must be in the same folder as app.py: " + ", ".join(missing)
             + ". Run the final notebook first (it saves them).")
    st.stop()

model, mean, std = load_everything(model_file, mean_file, std_file)


def save_upload(uploaded_file):

    path = "temp_files/" + uploaded_file.name.replace("..", "_")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    content = uploaded_file.getbuffer()
    with open(path, "wb") as f:
        f.write(content)
    return path, hashlib.sha256(content).hexdigest()

def sha256_of_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def skip_reason(path):
    try:
        raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
        idx = get_channel_index(raw)
        fs = int(raw.info["sfreq"])
        if idx is None:
            return "The required EEG channels (standard 18-channel bipolar montage) were not all found."
        if fs != expected_fs:
            return "Sampling rate is " + str(fs) + " Hz (" + str(expected_fs) + " Hz is required)."
        if raw.n_times < window_sec * fs:
            return "The recording is too short to analyse (less than " + str(window_sec) + " seconds)."
    except Exception:
        return "The file could not be opened as an EDF recording."
    return "This recording type is not supported."


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

def get_start_time(path):
    raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
    return raw.info["meas_date"]


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


def plot_signal(ax, path, start, end, flagged, real):
    data, fs = read_signal(path, start, end)
    t = np.arange(data.shape[1]) / fs + start

    for i in range(data.shape[0]):
        ax.plot(t[::2], data[i][::2] + i * 300, linewidth=0.6, color="#14213D")  

    for fs_, fe_ in flagged:
        if fs_ < end and fe_ > start:
            ax.axvspan(max(fs_, start), min(fe_, end), color="red", alpha=0.15)
    for rs_, re_ in real:
        if rs_ < end and re_ > start:
            ax.axvspan(max(rs_, start), min(re_, end), color="green", alpha=0.15)

    ax.set_yticks([i * 300 for i in range(len(channels))])       
    ax.set_yticklabels(channels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Time in recording (seconds)")
    ax.grid(axis="x", alpha=0.2)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    legend = [Patch(color="red", alpha=0.3, label="Flagged by automated analysis")]
    if len(real) > 0:
        legend.append(Patch(color="green", alpha=0.3, label="Reference seizure (demo only)"))
    ax.legend(handles=legend, loc="upper right", fontsize=8, framealpha=0.9)

def show_signal(path, start, end, flagged, real):
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    plot_signal(ax2, path, start, end, flagged, real)
    st.pyplot(fig2)
    plt.close(fig2)


def seg_id(name, seg):
    recording = st.session_state.get("hashes", {}).get(name, name)
    return recording + "|" + str(seg["start"]) + "|" + str(seg["end"])

def open_review_db():
    conn = sqlite3.connect(review_db_file)
    conn.execute("CREATE TABLE IF NOT EXISTS reviews (recording_hash TEXT, event_start INTEGER, event_end INTEGER, "
                 "file_name TEXT, decision TEXT, note TEXT, reviewed_at TEXT, "
                 "PRIMARY KEY (recording_hash, event_start, event_end))")
    return conn

def load_saved_reviews(recording_hashes):
    try:
        with closing(open_review_db()) as conn:
            rows = conn.execute("SELECT recording_hash, event_start, event_end, decision, note, reviewed_at FROM reviews").fetchall()
    except sqlite3.Error:
        st.session_state["db_error"] = True
        return
    for recording, start, end, decision, note, saved_at in rows:
        if recording in recording_hashes and decision in decision_options:
            st.session_state["reviews"][recording + "|" + str(start) + "|" + str(end)] = {
                "decision": decision, "note": note, "time": saved_at}

def get_review(sid):
    if sid in st.session_state["reviews"]:
        return st.session_state["reviews"][sid]
    return {"decision": "Not reviewed", "note": "", "time": ""}

def save_review(sid):
    decision = st.session_state["decision_" + sid]
    note = st.session_state["note_" + sid]
    saved_at = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    st.session_state["reviews"][sid] = {"decision": decision, "note": note, "time": saved_at}

    recording, start, end = sid.rsplit("|", 2)
    file_name = st.session_state.get("file_of_hash", {}).get(recording, "")
    try:
        with closing(open_review_db()) as conn:
            with conn:
                conn.execute("INSERT OR REPLACE INTO reviews VALUES (?, ?, ?, ?, ?, ?, ?)",
                             (recording, int(start), int(end), file_name, decision, note, saved_at))
    except sqlite3.Error:
        st.session_state["db_error"] = True

def final_status(decision):
    if decision == "Confirmed seizure":
        return "Specialist-confirmed seizure"
    if decision == "Not a seizure (reject)":
        return "Marked as not a seizure"
    if decision == "Needs more review":
        return "Requires further specialist review"
    return "Not yet reviewed (automated flag only)"

def count_reviewed(name, segs):
    n = 0
    for seg in segs:
        if get_review(seg_id(name, seg))["decision"] != "Not reviewed":
            n = n + 1
    return n

NAVY = colors.HexColor("#14213D")
TEAL = colors.HexColor("#0E7C86")
LIGHT = colors.HexColor("#F1F5F9")
LINE = colors.HexColor("#DCE3EB")

report_disclaimer = ("This report is generated using a research prototype designed to highlight EEG segments that may "
                     "warrant further review. Automated model predictions are not medical diagnoses and must not be used "
                     "independently for diagnosis or treatment. All flagged events require interpretation and review by a "
                     "qualified specialist. Final clinical decisions remain the responsibility of the treating specialist.")


def patient_code(name):
    file_name = os.path.basename(name).lower()
    for pattern in patient_id_patterns:
        match = re.match(pattern, file_name)
        if match:
            return match.group(1)
    return None

def make_report_groups(names, skipped, duplicates, combine):
    groups = []
    by_key = {}
    for name in sorted(names + skipped):
        code = patient_code(name)
        if combine and code is not None:
            key = "patient_" + code
            title = "Patient " + code
        else:
            key = "file_" + name
            title = os.path.basename(name)

        if key not in by_key:
            by_key[key] = {"key": key, "title": title, "names": [], "skipped": [], "duplicates": [],
                           "is_patient": key.startswith("patient_")}
            groups.append(by_key[key])
        if name in skipped:
            by_key[key]["skipped"].append(name)
        else:
            by_key[key]["names"].append(name)

    for copy, original in duplicates.items():
        for group in groups:
            if original in group["names"] or original in group["skipped"]:
                group["duplicates"].append(copy)
    return groups

def collect_events(names, all_segments, starts, shown_name):
    events = []
    for name in names:
        for seg in all_segments[name]:
            review = get_review(seg_id(name, seg))
            events.append({"name": name, "label": shown_name[name], "seg": seg, "decision": review["decision"],
                           "note": review["note"], "time": review["time"], "start_dt": starts[name]})
    for i in range(len(events)):
        events[i]["id"] = "E" + str(i + 1).zfill(2)
    return events

def recording_info(path):
    raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
    fs = int(raw.info["sfreq"])
    return raw.n_times / fs, fs

def clean(text):
    return escape(str(text)).replace("\n", "<br/>")

def report_styles():
    s = {}
    s["title"] = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=28, leading=32, textColor=NAVY)
    s["subtitle"] = ParagraphStyle("subtitle", fontName="Helvetica", fontSize=13, leading=17, textColor=TEAL)
    s["h1"] = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=NAVY, spaceBefore=14, spaceAfter=2, keepWithNext=1)
    s["h2"] = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=NAVY, spaceBefore=6, spaceAfter=4)
    s["body"] = ParagraphStyle("body", fontName="Helvetica", fontSize=9.5, leading=14, textColor=colors.HexColor("#1F2937"))
    s["small"] = ParagraphStyle("small", fontName="Helvetica", fontSize=8, leading=10.5, textColor=colors.HexColor("#1F2937"))
    s["head"] = ParagraphStyle("head", fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=colors.white)
    s["label"] = ParagraphStyle("label", fontName="Helvetica-Bold", fontSize=8.5, leading=11, textColor=NAVY)
    s["number"] = ParagraphStyle("number", fontName="Helvetica-Bold", fontSize=24, leading=28, textColor=NAVY, alignment=TA_CENTER)
    s["card"] = ParagraphStyle("card", fontName="Helvetica", fontSize=8.5, leading=11, textColor=colors.HexColor("#44546A"), alignment=TA_CENTER)
    s["notice"] = ParagraphStyle("notice", fontName="Helvetica", fontSize=9.5, leading=14, textColor=colors.HexColor("#1F2937"))
    return s

def section_title(text, s):
    line = HRFlowable(width="100%", thickness=1.2, color=TEAL, spaceAfter=8)
    line.keepWithNext = True            
    return [Paragraph(text, s["h1"]), line]

def make_table(header, rows, widths, s, extra_style=None):
    data = [[Paragraph(h, s["head"]) for h in header]] + rows
    table = Table(data, colWidths=widths, repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), NAVY),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
             ("GRID", (0, 0), (-1, -1), 0.25, LINE),
             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
             ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]
    if extra_style:
        style = style + extra_style
    table.setStyle(TableStyle(style))
    return table

def make_numbered_canvas(identifier):
    class NumberedCanvas(canvas.Canvas):
        def __init__(self, *args, **kwargs):
            canvas.Canvas.__init__(self, *args, **kwargs)
            self.saved_pages = []

        def showPage(self):
            self.saved_pages.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            total = len(self.saved_pages)
            for page in self.saved_pages:
                self.__dict__.update(page)
                self.draw_header_footer(total)
                canvas.Canvas.showPage(self)
            canvas.Canvas.save(self)

        def draw_header_footer(self, total):
            width, height = A4
            self.setStrokeColor(TEAL)
            self.setLineWidth(1)
            self.line(2 * cm, height - 1.5 * cm, width - 2 * cm, height - 1.5 * cm)
            self.setFillColor(NAVY)
            self.setFont("Helvetica-Bold", 9)
            self.drawString(2 * cm, height - 1.2 * cm, "EEG REVIEW REPORT")
            self.setFillColor(colors.HexColor("#5B6B7F"))
            self.setFont("Helvetica", 8)
            self.drawRightString(width - 2 * cm, height - 1.2 * cm, identifier[:70])

            self.setStrokeColor(LINE)
            self.line(2 * cm, 1.5 * cm, width - 2 * cm, 1.5 * cm)
            self.drawString(2 * cm, 1.0 * cm, "Research prototype. Not a medical diagnosis.")
            self.drawRightString(width - 2 * cm, 1.0 * cm, "Page " + str(self._pageNumber) + " of " + str(total))

    return NumberedCanvas

def waveform_image(event, paths):
    seg = event["seg"]
    start = max(0, seg["start"] - 10)
    end = min(seg["end"] + 10, seg["start"] + 120)
    fig, ax = plt.subplots(figsize=(11, 5.6))
    plot_signal(ax, paths[event["name"]], start, end, [[seg["start"], seg["end"]]], [])
    ax.set_title(event["id"] + "   " + event["label"] + "   " + fmt(start) + " to " + fmt(end),
                 fontsize=10, loc="left")
    picture = io.BytesIO()
    fig.savefig(picture, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    picture.seek(0)
    return Image(picture, width=17 * cm, height=17 * cm * 5.6 / 11 * 0.97), start, end

def report_clock(start_dt, sec):
    total = second_of_day(start_dt) + int(sec)
    text = clock_text(start_dt, sec)
    if total >= 86400:
        text = text + " (+" + str(total // 86400) + " d)"
    return text


def make_pdf(group, all_segments, paths, starts, reasons, duplicates):
    names = group["names"]
    skipped = group["skipped"]
    copies = group["duplicates"]

    all_names = names + skipped + copies
    count = {}
    for n in all_names:
        count[os.path.basename(n)] = count.get(os.path.basename(n), 0) + 1
    shown_name = {}
    for n in all_names:
        if count[os.path.basename(n)] == 1:
            shown_name[n] = os.path.basename(n)
        else:
            shown_name[n] = n

    events = collect_events(names, all_segments, starts, shown_name)
    s = report_styles()

    def small(text):
        return Paragraph(text, s["small"])

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    if group["is_patient"]:
        identifier = group["title"] + " (code taken from the file names, identity not verified)"
    else:
        identifier = "Recording " + group["title"]

    infos = {}
    for name in names:
        infos[name] = recording_info(paths[name])
    total_seconds = sum(infos[n][0] for n in names)
    sampling = sorted(set(str(infos[n][1]) for n in names))

    counts = {"Confirmed seizure": 0, "Not a seizure (reject)": 0, "Needs more review": 0, "Not reviewed": 0}
    for e in events:
        counts[e["decision"]] = counts[e["decision"]] + 1

    section_number = [1]

    def heading(title):
        section_number[0] = section_number[0] + 1
        return section_title(str(section_number[0]) + ". " + title, s)

    story = []

    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("EEG REVIEW REPORT", s["title"]))
    story.append(Paragraph("Electroencephalography Analysis and Specialist Review", s["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=2, color=TEAL, spaceBefore=10, spaceAfter=14))

    info_rows = [["Patient / recording identifier", identifier],
                 ["Report generated", now],
                 ["Recordings uploaded", str(len(all_names))],
                 ["Analysed", str(len(names))],
                 ["Not analysed", str(len(skipped))]]
    if len(copies) > 0:
        info_rows.append(["Duplicate uploads (identical content)", str(len(copies)) + " (counted once)"])
    if len(names) > 0:
        info_rows.append(["Total EEG duration", duration_text(total_seconds / 3600) + " (" + fmt(total_seconds) + ")"])
        info_rows.append(["Sampling rate", ", ".join(sampling) + " Hz"])
        info_rows.append(["EEG channels analysed", str(len(channels)) + " (bipolar montage)"])
    data = [[Paragraph(clean(a), s["label"]), Paragraph(clean(b), s["body"])] for a, b in info_rows]
    info_table = Table(data, colWidths=[5.5 * cm, 11.5 * cm])
    info_table.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.25, LINE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                    ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story.append(info_table)
    story.append(Spacer(1, 0.8 * cm))

    if len(names) > 0:
        # the four summary cards
        cards = [[len(events), "Flagged events", LIGHT, LINE],
                 [counts["Confirmed seizure"], "Specialist-confirmed", colors.HexColor("#DDF3E4"), colors.HexColor("#1E7B3A")],
                 [counts["Not a seizure (reject)"], "Marked as not a seizure", LIGHT, LINE],
                 [counts["Needs more review"] + counts["Not reviewed"], "Awaiting review", colors.HexColor("#FFF4D6"), colors.HexColor("#E0A100")]]
        card_cells = []
        for number, label, back, border in cards:
            card_cells.append([Paragraph(str(number), s["number"]), Paragraph(label, s["card"])])
        card_table = Table([[c[0] for c in card_cells], [c[1] for c in card_cells]], colWidths=[4.25 * cm] * 4, rowHeights=[1.5 * cm, 1.0 * cm])
        card_style = [("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
        for i in range(4):
            card_style.append(("BACKGROUND", (i, 0), (i, 1), cards[i][2]))
            card_style.append(("BOX", (i, 0), (i, 1), 1, cards[i][3]))
        card_style.append(("LINEAFTER", (0, 0), (2, 1), 6, colors.white))
        card_table.setStyle(TableStyle(card_style))
        story.append(card_table)
        story.append(Spacer(1, 0.2 * cm))
        story.append(small("Flagged events are periods of possible seizure activity highlighted by automated analysis (not a diagnosis). "
                               "Awaiting review = \"" + decision_label["Needs more review"] + "\" (" + str(counts["Needs more review"])
                               + ") plus not yet reviewed (" + str(counts["Not reviewed"]) + ")."))

    story.append(PageBreak())
    if len(names) > 0:
        story = story + heading("Recording information")
        rows = []
        for name in names:
            code = patient_code(name)
            folder = os.path.dirname(name)
            if folder == "":
                folder = "(uploaded files)"
            where = clean(folder)
            if code is not None:
                where = where + "<br/>Patient code: " + clean(code)
            if starts[name] is not None:
                started = clean(starts[name].strftime("%Y-%m-%d %I:%M:%S %p"))
            else:
                started = "not available"
            rows.append([small(clean(shown_name[name])), small(where),
                         small(started), small(fmt(infos[name][0]))])
        story.append(make_table(["Recording", "Folder / patient", "Start time", "Duration"],
                                rows, [5.5 * cm, 4.5 * cm, 4.0 * cm, 3.0 * cm], s))

    if len(skipped) > 0:
        story = story + heading("Recordings not analysed")
        story.append(Paragraph("These recordings were not analysed. A recording that was not analysed is not seizure-free: "
                               "no result is available for it.", s["body"]))
        story.append(Spacer(1, 0.2 * cm))
        rows = []
        for name in skipped:
            folder = os.path.dirname(name)
            if folder == "":
                folder = "(uploaded files)"
            rows.append([small(clean(shown_name[name])), small(clean(folder)),
                         small(clean(reasons.get(name, "reason not available")))])
        story.append(make_table(["Recording", "Folder", "Reason"], rows, [5.5 * cm, 4.5 * cm, 7.0 * cm], s))

    if len(copies) > 0:
        story.append(Paragraph("Duplicate uploads (identical file content, counted once)", s["h2"]))
        rows = []
        for copy in copies:
            rows.append([small(clean(shown_name[copy])),
                         small("same content as " + clean(shown_name.get(duplicates[copy], os.path.basename(duplicates[copy]))))])
        story.append(make_table(["Upload", "Note"], rows, [8.5 * cm, 8.5 * cm], s))

    if len(names) == 0:
        story = story + heading("Automated analysis result")
        story.append(Paragraph("No recording in this report could be analysed, so there is no automated result.", s["body"]))
    elif len(events) == 0:
        story = story + heading("Automated analysis result")
        story.append(Paragraph("The automated analysis did not flag possible seizure activity in " + ("this recording" if len(names) == 1 else "these recordings")
                               + ". No specialist review was needed. This is an automated result and does not exclude a seizure.", s["body"]))
    else:
        story = story + heading("Summary of flagged events")
        n_longer = 0
        for e in events:
            if e["seg"]["strength"] == "strong signal":
                n_longer = n_longer + 1
        flagged_names = []
        for name in names:
            if len(all_segments[name]) > 0:
                flagged_names.append(name)

        flagged_times = [e["seg"]["flagged_s"] for e in events]
        summary = [["Flagged events", str(len(events))],
                   ["Recordings with flagged events", str(len(flagged_names)) + " of " + str(len(names))],
                   ["Longer detection (" + str(strong_sec) + " s or more) / shorter detection", str(n_longer) + " / " + str(len(events) - n_longer)],
                   ["Detected activity per event (shortest / average / longest)",
                    str(min(flagged_times)) + " s / " + str(round(sum(flagged_times) / len(flagged_times))) + " s / " + str(max(flagged_times)) + " s"],
                   ["Total detected activity", str(sum(flagged_times)) + " s"]]
        data = [[Paragraph(clean(a), s["label"]), Paragraph(clean(b), s["body"])] for a, b in summary]
        summary_table = Table(data, colWidths=[8.5 * cm, 8.5 * cm])
        summary_table.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.25, LINE), ("TOPPADDING", (0, 0), (-1, -1), 5),
                                           ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        story.append(summary_table)
        story.append(Spacer(1, 0.3 * cm))
        story.append(small("<b>Detected activity</b> is the time within an event that the automated analysis marked as possible seizure activity. "
                               "It is an automated result, not a diagnosis."))

        story.append(Paragraph("Recordings with flagged events", s["h2"]))
        rows = []
        for name in flagged_names:
            segs = all_segments[name]
            longer = len([x for x in segs if x["strength"] == "strong signal"])
            rows.append([small(clean(shown_name[name])), small(str(len(segs))),
                         small(str(longer)), small(str(len(segs) - longer))])
        story.append(make_table(["Recording", "Flagged events", "Longer detection", "Shorter detection"], rows,
                                [8 * cm, 3 * cm, 3 * cm, 3 * cm], s))

        story = story + heading("Flagged event details")
        rows = []
        for e in events:
            seg = e["seg"]
            start_text = fmt(seg["start"])
            end_text = fmt(seg["end"])
            if e["start_dt"] is not None:
                start_text = start_text + "<br/>(" + report_clock(e["start_dt"], seg["start"]) + ")"
                end_text = end_text + "<br/>(" + report_clock(e["start_dt"], seg["end"]) + ")"
            category = flag_category(seg)
            rows.append([small(e["id"]), small(clean(e["label"])),
                         small(start_text), small(end_text),
                         small(str(seg["flagged_s"]) + " s"),
                         small(category)])
        story.append(make_table(["Event", "Recording", "Start", "End", "Detected activity", "Category"], rows,
                                [1.3 * cm, 4.8 * cm, 2.8 * cm, 2.8 * cm, 2.6 * cm, 2.7 * cm], s))
        story.append(small("Times are measured from the start of each recording (hours:minutes:seconds). Times of day in brackets are shown "
                               "when the recording stores its start time (+1 d = after midnight)."))

        story = story + heading("Specialist review")
        text_colour = {"Confirmed seizure": "#1E7B3A", "Not a seizure (reject)": "#44546A",
                       "Needs more review": "#8A5A00", "Not reviewed": "#B42318"}
        rows = []
        for e in events:
            colour = text_colour[e["decision"]]
            note = clean(e["note"]) if e["note"].strip() != "" else "-"
            decision_text = '<font color="' + colour + '"><b>' + clean(decision_label[e["decision"]]) + '</b></font>'
            if e["time"] != "":
                decision_text = decision_text + "<br/>" + clean(e["time"])
            rows.append([small(e["id"]), small(clean(e["label"])),
                         small(decision_text), small(note),
                         small('<font color="' + colour + '">' + clean(final_status(e["decision"])) + '</font>')])
        story.append(make_table(["Event", "Recording", "Specialist decision", "Notes", "Review status"], rows,
                                [1.3 * cm, 3.6 * cm, 3.2 * cm, 5.3 * cm, 3.6 * cm], s))
        story.append(small("An event that has not been reviewed is only an automated flag and is never counted as a confirmed seizure. "
                               "The date and time under a decision is when the specialist saved it."))

        shown = []
        for wanted in ["Confirmed seizure", "Needs more review", "Not reviewed"]:
            for e in events:
                if e["decision"] == wanted:
                    shown.append(e)
        if len(shown) > 0:
            story.append(PageBreak())
            story = story + heading("EEG tracings")
            story.append(Paragraph("EEG tracings of events confirmed by the specialist, then events requiring further review, "
                                   "then events not yet reviewed (automated flags only). "
                                   "Red shading marks the period flagged by automated analysis.", s["body"]))
            if len(shown) > 10:
                story.append(small(str(len(shown) - 10) + " more event(s) are listed in the tables but their tracing is not shown."))
            for e in shown[:10]:
                image, start, end = waveform_image(e, paths)
                seg = e["seg"]
                block = [Paragraph("Event " + e["id"] + "  -  " + clean(e["label"]), s["h2"]),
                         small("Time in recording: " + fmt(seg["start"]) + " to " + fmt(seg["end"]) + ".   Detected activity: "
                                   + str(seg["flagged_s"]) + " s.   Status: <b>" + clean(final_status(e["decision"])) + "</b>"),
                         Spacer(1, 0.2 * cm), image]
                if e["note"].strip() != "":
                    block.append(small("<b>Specialist notes:</b> " + clean(e["note"])))
                block.append(Spacer(1, 0.5 * cm))
                story.append(KeepTogether(block))

    story.append(Spacer(1, 0.8 * cm))
    notice = [Paragraph("Important notice", s["h2"]), Paragraph(report_disclaimer, s["notice"])]
    notice_table = Table([[notice]], colWidths=[17 * cm])
    notice_table.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1, TEAL), ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                                      ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                                      ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
    story.append(KeepTogether([notice_table]))

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2.2 * cm,
                            bottomMargin=2.0 * cm, title="EEG Review Report - " + group["title"])
    doc.build(story, canvasmaker=make_numbered_canvas(group["title"]))
    return buffer.getvalue()



def analyse_uploads(uploaded):
    shutil.rmtree("temp_files", ignore_errors=True)
    hash_of_path = {}
    for f in uploaded:
        path, digest = save_upload(f)
        hash_of_path[path] = digest
    edf_paths = sorted(p.replace("\\", "/") for p in glob.glob("temp_files/**/*.[eE][dD][fF]", recursive=True))
    if len(edf_paths) == 0:
        return False

    state = st.session_state
    for key in ["probas", "paths", "starts", "hashes", "file_of_hash", "skip_reasons", "duplicates"]:
        state[key] = {}
    for key in ["selected", "folder", "db_error"]:
        state.pop(key, None)

    bar = st.progress(0, text="Preparing EEG recordings...")
    seen = {}                                     
    for k, path in enumerate(edf_paths):
        name = path[len("temp_files/"):]
        digest = hash_of_path.get(path) or sha256_of_file(path)
        if digest in seen:
            state["duplicates"][name] = seen[digest]       
        else:
            seen[digest] = name
            state["paths"][name] = path
            state["hashes"][name] = digest
            state["file_of_hash"][digest] = os.path.basename(name)
            reason = None
            try:
                proba = predict_file(path)                
                if proba is None:
                    reason = skip_reason(path)
            except Exception:
                proba = None
                reason = "The file could not be opened as an EDF recording."
            state["probas"][name] = proba
            if reason is not None:
                state["skip_reasons"][name] = reason
            try:
                state["starts"][name] = get_start_time(path)
            except Exception:
                state["starts"][name] = None
        bar.progress((k + 1) / len(edf_paths), text="Processed " + str(k + 1) + " of " + str(len(edf_paths)) + " recordings")

    load_saved_reviews(set(state["hashes"].values()))     
    return True



def show_dashboard(recordings, probas, all_segments):
    hours = 0
    with_flags = 0
    total_seiz = 0
    reviewed = 0
    for name in recordings:
        hours = hours + hours_of(probas[name])
        if len(all_segments[name]) > 0:
            with_flags = with_flags + 1
        total_seiz = total_seiz + len(all_segments[name])
        reviewed = reviewed + count_reviewed(name, all_segments[name])
    pending = total_seiz - reviewed

    row1 = st.columns(3)
    row1[0].metric("EEG recordings", len(recordings))
    row1[1].metric("Total EEG time", duration_text(hours))
    row1[2].metric("Recordings flagged", with_flags)

    row2 = st.columns(3)
    row2[0].metric("Flagged events", total_seiz)
    row2[1].metric("Reviewed", reviewed)
    with row2[2]:
        card_key = "done_card"
        if pending > 0:
            card_key = "pending_card"
        with st.container(key=card_key):
            st.metric("Awaiting review", pending)

    if total_seiz > 0:
        st.progress(reviewed / total_seiz, text="Review progress: " + str(reviewed) + " of " + str(total_seiz)
                    + " flagged events reviewed")
        if pending > 0:
            st.warning("⏳ **" + str(pending) + (" flagged event still needs" if pending == 1 else " flagged events still need") + " a specialist decision.**")
        else:
            st.success("✅ All flagged events have been reviewed.")
    return hours