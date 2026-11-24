import os
import shutil
import streamlit as st

st.set_page_config(page_title="EEG Seizure Review Assistant",
                   page_icon="brain_icon.png" if os.path.exists("brain_icon.png") else "🧠", layout="wide")

from features import (analyse_uploads, chip, clock_text, count_reviewed, decision_label, decision_options, duration_text,
                      final_status, find_real_seizures, find_segments, flag_category, fmt, get_review, hours_of,
                      make_pdf, make_report_groups, review_db_file, save_review, second_of_day, seg_id,
                      show_dashboard, show_signal)
from style import apply_style, show_banner, show_welcome_header

apply_style()

if "started" not in st.session_state:
    st.session_state["started"] = True
    shutil.rmtree("temp_files", ignore_errors=True)
st.session_state.setdefault("reviews", {})
st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("page", "upload")

show_banner()


def card_key(k, n_seiz, n_done):
    return ("pendingcard_" if n_seiz > n_done else "card_") + str(k)


def show_flags(n_seiz, n_done):
    if n_seiz == 0:
        st.markdown(chip("No events flagged", "green"), unsafe_allow_html=True)
        return
    st.markdown(chip(str(n_seiz) + (" flagged event" if n_seiz == 1 else " flagged events"), "red")
                + chip(str(n_seiz - n_done) + " awaiting review", "amber" if n_seiz > n_done else "green"),
                unsafe_allow_html=True)
    st.progress(n_done / n_seiz, text=str(n_done) + " of " + str(n_seiz) + " reviewed")


if st.session_state["page"] == "upload":
    show_welcome_header()
    key = str(st.session_state["uploader_key"])

    with st.container(border=True):
        st.markdown('<div class="upload-heading">Upload EEG recordings</div>'
                    '<div class="upload-description">Upload EDF recordings to begin. Recordings in the supported format '
                    '(256 Hz, standard 18-channel bipolar montage) are analysed and prepared for review.</div>',
                    unsafe_allow_html=True)
        mode = st.radio("Choose how to upload", ["A whole folder", "Individual files"], horizontal=True, key="upload_mode")
        if mode == "A whole folder":
            uploaded = st.file_uploader("Select an EEG folder", type=["edf"], accept_multiple_files="directory", key="folder_" + key)
        else:
            uploaded = st.file_uploader("Select EDF recordings", type=["edf"], accept_multiple_files=True, key="files_" + key)
        if uploaded:
            st.caption(str(len(uploaded)) + " files selected · " + str(round(sum(f.size for f in uploaded) / 1e6, 1)) + " MB")
            st.caption("For large datasets, consider uploading a few recordings at a time.")
        analyse = st.button("Analyse recordings", type="primary", use_container_width=True, disabled=not uploaded)

    st.markdown('<div class="workspace-note"><strong>Research prototype:</strong> Events flagged by the automated analysis '
                'are meant to support specialist review. They are not a diagnosis, and every flagged event needs specialist '
                'assessment.</div>', unsafe_allow_html=True)

    if analyse and uploaded:
        if analyse_uploads(uploaded):
            st.session_state["uploader_key"] += 1
            st.session_state["page"] = "dashboard"
            st.rerun()
        else:
            st.error("No EDF recordings were found in the uploaded content.")


if st.session_state["page"] == "dashboard" and "probas" in st.session_state:
    if st.button("← Upload new recordings"):
        st.session_state["page"] = "upload"
        st.session_state.pop("selected", None)
        st.rerun()

    probas, paths, starts = st.session_state["probas"], st.session_state["paths"], st.session_state["starts"]
    reasons, duplicates = st.session_state["skip_reasons"], st.session_state["duplicates"]
    skipped = [n for n in probas if probas[n] is None]
    names = [n for n in probas if probas[n] is not None]

    if skipped:
        lines = ["- **" + os.path.basename(n) + "**: " + reasons.get(n, "Reason not available") for n in skipped]
        st.warning("These recordings were not analysed. This does not mean they are seizure-free; no result is available "
                   "for them.\n" + "\n".join(lines))
    if duplicates:
        st.info(str(len(duplicates)) + " uploaded file(s) were identical to another recording and were counted once.")
    if st.session_state.get("db_error"):
        st.warning("Reviews could not be saved to the database file (" + review_db_file
                   + "). They will be lost when this session ends.")

    all_segments, folder_of, folders = {}, {}, []
    for name in names:
        all_segments[name] = find_segments(probas[name])
        folder_of[name] = os.path.dirname(name) or "(uploaded files)"
        if folder_of[name] not in folders:
            folders.append(folder_of[name])
    current_folder = folders[0] if len(folders) == 1 else st.session_state.get("folder")

    if current_folder is None:
        st.markdown("### Overview")
        show_dashboard(names, probas, all_segments)
        st.markdown("### Folders")
        for row_start in range(0, len(folders), 3):
            cols = st.columns(3)
            for j in range(min(3, len(folders) - row_start)):
                k = row_start + j
                folder = folders[k]
                inside = [n for n in names if folder_of[n] == folder]
                n_seiz = sum(len(all_segments[n]) for n in inside)
                n_done = sum(count_reviewed(n, all_segments[n]) for n in inside)
                hours = sum(hours_of(probas[n]) for n in inside)
                with cols[j], st.container(border=True, key=card_key(k, n_seiz, n_done)):
                    st.markdown("**" + folder + "**")
                    st.caption(str(len(inside)) + (" recording · " if len(inside) == 1 else " recordings · ")
                               + duration_text(hours) + " of EEG")
                    show_flags(n_seiz, n_done)
                    if st.button("Open folder", key="open_" + folder):
                        st.session_state["folder"] = folder
                        st.session_state.pop("selected", None)
                        st.rerun()

    else:
        if len(folders) > 1 and st.button("← All folders"):
            st.session_state.pop("folder", None)
            st.session_state.pop("selected", None)
            st.rerun()

        in_folder = [n for n in names if folder_of[n] == current_folder]
        st.markdown("### " + current_folder)
        show_dashboard(in_folder, probas, all_segments)

        period_names = ["Night (12 AM–6 AM)", "Morning (6 AM–12 PM)", "Afternoon (12 PM–6 PM)", "Evening (6 PM–12 AM)"]
        period_flags = [0, 0, 0, 0]
        for name in in_folder:
            if starts[name] is not None:
                for seg in all_segments[name]:
                    hour = ((second_of_day(starts[name]) + int(seg["start"])) // 3600) % 24
                    period_flags[hour // 6] += 1
        flagged_total = sum(period_flags)
        if flagged_total > 0:
            st.markdown("### Time of day")
            top = period_flags.index(max(period_flags))
            st.write("Most flagged events occurred in the **" + period_names[top] + "**: " + str(period_flags[top])
                     + " of " + str(flagged_total) + ".")
            if flagged_total < 5:
                st.caption("There are only " + str(flagged_total) + " flagged events. This is too few to show a pattern.")
            cols = st.columns(4)
            for k in range(4):
                cols[k].metric(period_names[k], period_flags[k])
            st.caption("These are events flagged by the automated analysis. Some may not be seizures.")

        st.markdown("### Recordings")
        st.caption("Recordings with the strongest detections are listed first. This is only a suggested order for "
                   "reviewing, not a measure of medical severity.")
        in_folder.sort(key=lambda n: max([seg["peak"] for seg in all_segments[n]], default=0), reverse=True)
        selected = st.session_state.get("selected")

        with (st.container(height=560, border=True) if len(in_folder) > 6 else st.container()):
            for row_start in range(0, len(in_folder), 3):
                cols = st.columns(3)
                for j in range(min(3, len(in_folder) - row_start)):
                    k = row_start + j
                    name = in_folder[k]
                    n_seiz = len(all_segments[name])
                    n_done = count_reviewed(name, all_segments[name])
                    with cols[j], st.container(border=True, key=card_key(k, n_seiz, n_done)):
                        st.markdown("**" + os.path.basename(name) + "**")
                        st.caption(duration_text(hours_of(probas[name])) + " of EEG")
                        show_flags(n_seiz, n_done)
                        if st.button("Review recording", key="review_" + name, use_container_width=True,
                                     type="primary" if name == selected else "secondary"):
                            st.session_state["selected"] = name
                            st.session_state["scroll_to_review"] = True
                            st.rerun()

        st.markdown("<div class='section-separator'></div><div id='eeg-review-section'></div>", unsafe_allow_html=True)
        st.markdown("### EEG review")
        if st.session_state.pop("scroll_to_review", False):
            st.components.v1.html("""<script>
                const target = window.parent.document.getElementById('eeg-review-section');
                if (target) { target.scrollIntoView({behavior: 'smooth', block: 'start'}); }
                </script>""", height=1)

        if selected not in in_folder:
            st.info("Select **Review recording** above to open a recording.")
        else:
            segs, start_dt = all_segments[selected], starts[selected]

            with st.container(border=True):
                title_col, close_col = st.columns([6, 1])
                title_col.markdown("#### " + os.path.basename(selected))
                if close_col.button("✕ Close", key="close_review", use_container_width=True):
                    st.session_state.pop("selected", None)
                    st.rerun()
                info = st.columns(4)
                for col, (label, value) in zip(info, [
                        ("Length", duration_text(hours_of(probas[selected]))),
                        ("Started at", clock_text(start_dt, 0) if start_dt is not None else "Unknown"),
                        ("Flagged events", str(len(segs))),
                        ("Review progress", str(count_reviewed(selected, segs)) + " of " + str(len(segs)))]):
                    col.caption(label)
                    col.markdown("**" + value + "**")

            real = find_real_seizures(selected)
            real_ranges = []
            if real is not None and st.checkbox("Show reference seizure annotations (demo only)"):
                real_ranges = real
                if len(real) == 0:
                    st.info("The reference summary file indicates no seizure annotations for this recording.")
                for ts, te in real:
                    detected = any(seg["start"] < te and seg["end"] > ts for seg in segs)
                    st.write("Reference seizure " + fmt(ts) + " to " + fmt(te) + ": "
                             + ("detected" if detected else "not detected") + " by the automated analysis")

            if len(segs) == 0:
                st.success("The automated analysis did not flag any events in this recording.")
            else:
                icons = {"Not reviewed": "⏳", "Confirmed seizure": "✅", "Not a seizure (reject)": "❌", "Needs more review": "❓"}
                labels = []
                for i, seg in enumerate(segs):
                    label = (icons[get_review(seg_id(selected, seg))["decision"]] + "  #" + str(i + 1) + "   " + fmt(seg["start"])
                             + " to " + fmt(seg["end"]) + "   (" + str(seg["end"] - seg["start"]) + " s detected)")
                    if start_dt is not None:
                        label += "   at " + clock_text(start_dt, seg["start"])
                    labels.append(label)

                st.caption("⏳ Awaiting review   ✅ Specialist-confirmed   ❌ Marked as not a seizure   ❓ Further review needed")
                pick = st.selectbox("Select a flagged event", range(len(segs)), format_func=lambda i: labels[i],
                                    key="seg_pick_" + selected)
                seg = segs[pick]
                sid = seg_id(selected, seg)
                start, end = max(0, seg["start"] - 10), min(seg["end"] + 10, seg["start"] + 120)
                plot_col, side_col = st.columns([3, 1.2])

                with plot_col, st.container(border=True):
                    st.markdown("**EEG · 18 channels**  " + fmt(start) + " to " + fmt(end)
                                + "  (red shading = flagged by automated analysis)")
                    if end < seg["end"] + 10:
                        st.caption("This event is long, so only its first two minutes are shown.")
                    show_signal(paths[selected], start, end, [[o["start"], o["end"]] for o in segs], real_ranges)

                with side_col:
                    with st.container(border=True):
                        st.markdown("#### Automated detection")
                        st.markdown(chip(flag_category(seg), "red" if seg["strength"] == "strong signal" else "gray"),
                                    unsafe_allow_html=True)
                        st.markdown("**Time in recording:** " + fmt(seg["start"]) + " to " + fmt(seg["end"]))
                        if start_dt is not None:
                            st.markdown("**Time of day:** " + clock_text(start_dt, seg["start"]))
                        st.markdown("**Detected activity:** " + str(seg["flagged_s"]) + " seconds")
                        st.caption("Flagged by automated analysis as possible seizure activity. This is not a diagnosis.")

                    with st.container(border=True):
                        st.markdown("#### Specialist review")
                        review = get_review(sid)
                        decision = st.radio("Your decision", decision_options, index=decision_options.index(review["decision"]),
                                            key="decision_" + sid, format_func=lambda d: decision_label[d],
                                            on_change=save_review, args=(sid,))
                        st.text_area("Notes", value=review["note"], key="note_" + sid, on_change=save_review, args=(sid,),
                                     placeholder="Add observations or review notes...")
                        colours = {"Not reviewed": "amber", "Confirmed seizure": "teal",
                                   "Not a seizure (reject)": "gray", "Needs more review": "amber"}
                        st.markdown("**Status:** " + chip(final_status(decision), colours[decision]), unsafe_allow_html=True)


    st.markdown("<div class='section-separator'></div>", unsafe_allow_html=True)
    st.markdown("### Reports")
    report_choice = st.radio(
        "Report type", ["One report per patient", "A separate report for each recording"], horizontal=True,
        help="One report per patient: recordings whose file names share the same patient code (for example chb01_03.edf and "
             "chb01_04.edf) are combined, even from different folders. The code comes from the file name only and does not "
             "verify patient identity. Recordings without a patient code get their own report. "
             "Different patients are never combined.")
    groups = make_report_groups(names, skipped, duplicates, report_choice == "One report per patient")
    st.caption("The report includes the reviews saved at the moment you create it. Create it again after changing a review.")

    events_of = [sum(len(all_segments[n]) for n in g["names"]) for g in groups]
    default = 0
    for i, g in enumerate(groups):
        if st.session_state.get("selected") in g["names"]:
            default = i
    pick_report = st.selectbox("Choose a report", range(len(groups)), index=default,
                               format_func=lambda i: groups[i]["title"] + "   (" + str(events_of[i]) + " flagged events)")
    group = groups[pick_report]

    with st.container(border=True):
        r1, r2, r3 = st.columns([4, 2, 2])
        r1.markdown("**" + group["title"] + "**")
        r1.caption(str(len(group["names"])) + " recording(s) analysed · " + str(len(group["skipped"])) + " not analysed · "
                   + str(events_of[pick_report]) + " flagged events")
        pdf_key = "pdf_" + group["key"]
        if r2.button("Create PDF report", key="make_" + group["key"], use_container_width=True):
            with st.spinner("Creating the report..."):
                st.session_state[pdf_key] = make_pdf(group, all_segments, paths, starts, reasons, duplicates)
        if pdf_key in st.session_state:
            r3.download_button("Download PDF", st.session_state[pdf_key], key="download_" + group["key"],
                               use_container_width=True, file_name=group["title"].replace(" ", "_") + "_report.pdf",
                               mime="application/pdf")

    if os.path.exists(review_db_file):
        with open(review_db_file, "rb") as f:
            st.download_button("Download saved reviews (reviews.db)", f.read(), file_name="reviews.db",
                               help="Keep a copy of the specialist reviews. They are stored on the server, which can erase them when it restarts.")
