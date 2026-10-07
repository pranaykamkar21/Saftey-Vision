from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch

from src.event_store import RETENTION_DAYS, EventStore
from src.opencv_inference import ComplianceDetector, EvidenceCapture, open_default_model, process_video


st.set_page_config(
    page_title="Saftey Vision",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --bg: #07111b; --panel: #0d1b29; --line: #1a3142; --mint: #5ee6bd; --muted: #8da2b4; }
    .stApp { background: radial-gradient(ellipse at 8% 0%, #102938 0, #07111b 42%); color: #edf5f7; }
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif !important; letter-spacing: -0.04em; }
    section[data-testid="stSidebar"] { background: #091521; border-right: 1px solid var(--line); }
    div[data-testid="stMetric"] { background: linear-gradient(145deg, #112333, #0b1722); border: 1px solid #1b3547; padding: 16px 18px; border-radius: 16px; }
    div[data-testid="stMetricLabel"] { color: #95aabc; }
    div[data-testid="stMetricValue"] { color: #eff8fa; font-family: 'Segoe UI', sans-serif; }
    .hero { padding: 25px 28px; border-radius: 22px; border: 1px solid #1d3c4e;
            background: linear-gradient(115deg, rgba(19,51,67,.95), rgba(12,28,42,.88) 62%, rgba(16,54,58,.8));
            margin: 0 0 20px 0; }
    .hero-kicker { color: #68e7c0; font-size: 12px; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
    .hero h1 { font-size: clamp(30px, 4vw, 48px); margin: 8px 0; color: #f0f8fa; }
    .hero p { color: #a9bdc8; margin: 0; font-size: 15px; max-width: 760px; }
    .section-label { color: #69e5bf; font-size: 12px; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
    .notice { background: #102332; border: 1px solid #214356; color: #b5c9d2; border-radius: 12px; padding: 12px 15px; font-size: 13px; }
    .event-card { background: #0d1b29; border: 1px solid #1a3142; padding: 14px; border-radius: 16px; margin-bottom: 12px; }
    div.stButton > button[kind="primary"] { background: #52dcb4; color: #07111b; border: 0; font-weight: 700; border-radius: 10px; }
    div.stButton > button { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

ROOT = Path(__file__).resolve().parent.parent
DETECTOR_CACHE_VERSION = "review-predictions-v1"


@st.cache_resource
def get_store() -> EventStore:
    return EventStore()


@st.cache_resource
def get_detector(selected_task: str, selected_model: str, cache_version: str) -> ComplianceDetector:
    if cache_version != DETECTOR_CACHE_VERSION:
        raise ValueError(f"Unsupported detector cache version: {cache_version}")
    return ComplianceDetector(selected_task, selected_model)


store = get_store()
store.purge_expired()

with st.sidebar:
    st.markdown("## 🛡️ SAFTEY VISION")
    st.caption("LOCAL SAFETY INTELLIGENCE")
    if st.button("↻  Refresh dashboard", use_container_width=True):
        st.rerun()
    st.divider()
    task = st.selectbox("Monitoring profile", ["mask", "helmet"], format_func=lambda value: "Face mask" if value == "mask" else "Safety helmet")
    threshold = 0.95
    st.caption("CONFIDENCE · 95% FIXED")
    save_evidence = st.toggle("Capture violation evidence", value=True)
    st.markdown(
        f'<div class="notice">Only face/head crops and event records stay on this computer. They are removed after {RETENTION_DAYS} days. No identity recognition is used.</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    has_helmet_detector = task == "helmet" and any(
        (ROOT / "models").glob("helmet_detector*/weights/best.pt")
    )
    model_label = "YOLO helmet detector" if has_helmet_detector else "MobileNetV2"
    st.caption(f"MODEL · {model_label}")
    st.caption(f"DEVICE · {'CUDA' if torch.cuda.is_available() else 'CPU'}")

try:
    model_path = open_default_model(task)
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

detector = get_detector(task, str(model_path), DETECTOR_CACHE_VERSION)
detector.confidence = threshold

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Computer vision · Local-first · Privacy aware</div>
      <h1>Saftey Vision</h1>
      <p>Clear mask and helmet checks from photos, clips, and your camera. Evidence saves only a face or head crop, never the full frame.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

all_summary = store.summary()
task_events = all_summary["by_task"].get(task, 0)
overall_events = all_summary["total"]
mask_events = all_summary["by_task"].get("mask", 0)
helmet_events = all_summary["by_task"].get("helmet", 0)

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Stored violation events", f"{overall_events:,}")
metric2.metric("Mask events", f"{mask_events:,}")
metric3.metric("Helmet events", f"{helmet_events:,}")
metric4.metric(f"Current profile · {task}", f"{task_events:,}")

analyze_tab, evidence_tab, about_tab = st.tabs(["◉  Live analysis", "▣  Evidence vault", "✳  About & model"])

with analyze_tab:
    left, right = st.columns([1.05, 1.5], gap="large")
    with left:
        st.markdown('<div class="section-label">Choose an input</div>', unsafe_allow_html=True)
        mode = st.radio("Input source", ["Image", "Video", "Webcam snapshot"], horizontal=True, label_visibility="collapsed")
        st.caption(f"Profile: **{task.title()}** · confidence threshold: **95%**")
        if not save_evidence:
            st.warning("Evidence saving is off. The analysis will not add violation photos or event records.")
    with right:
        if mode == "Image":
            uploaded = st.file_uploader("Drop a photo here", type=["jpg", "jpeg", "png"], key="image-upload")
            if uploaded and st.button("Analyze image", type="primary"):
                frame = cv2.imdecode(np.frombuffer(uploaded.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    st.error("The uploaded image could not be decoded.")
                else:
                    annotated, predictions = detector.predict_frame(frame)
                    counts = detector.count_compliance(predictions)
                    if save_evidence:
                        EvidenceCapture(detector, store).capture(
                            [*predictions, *detector.last_review_predictions], frame, "image upload"
                        )
                    st.session_state["last_annotated"] = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    st.session_state["last_counts"] = counts
                    st.session_state["last_frame_stats"] = detector.last_frame_stats.copy()

        elif mode == "Video":
            uploaded = st.file_uploader("Upload a clip", type=["mp4", "avi", "mov", "mkv"], key="video-upload")
            if uploaded and st.button("Process video", type="primary"):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    folder = Path(temporary_directory)
                    source = folder / f"input{Path(uploaded.name).suffix.lower()}"
                    destination = folder / "annotated.mp4"
                    source.write_bytes(uploaded.getvalue())
                    try:
                        counts, frame_count = process_video(
                            detector,
                            source,
                            destination,
                            evidence_store=store if save_evidence else None,
                            capture_evidence=save_evidence,
                            source_label="video upload",
                        )
                        st.session_state["last_video"] = destination.read_bytes()
                        st.session_state["last_video_name"] = f"{Path(uploaded.name).stem}_annotated.mp4"
                        st.session_state["last_counts"] = counts
                        st.session_state["last_frame_count"] = frame_count
                        st.session_state["last_frame_stats"] = {
                            "candidate_count": counts["candidates"],
                            "uncertain_count": counts["uncertain"],
                        }
                    except (OSError, RuntimeError, ValueError) as error:
                        st.error(f"Video processing failed: {error}")

        else:
            snapshot = st.camera_input("Take a webcam snapshot", key="webcam-snapshot")
            if snapshot and st.button("Analyze snapshot", type="primary"):
                frame = cv2.imdecode(np.frombuffer(snapshot.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    st.error("The webcam image could not be decoded.")
                else:
                    annotated, predictions = detector.predict_frame(frame)
                    counts = detector.count_compliance(predictions)
                    if save_evidence:
                        EvidenceCapture(detector, store).capture(
                            [*predictions, *detector.last_review_predictions],
                            frame,
                            "webcam snapshot",
                        )
                    st.session_state["last_annotated"] = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    st.session_state["last_counts"] = counts
                    st.session_state["last_frame_stats"] = detector.last_frame_stats.copy()

    with right:
        if "last_annotated" in st.session_state:
            st.image(st.session_state["last_annotated"], caption="Annotated result", width="stretch")
        if "last_video" in st.session_state and mode == "Video":
            st.video(st.session_state["last_video"])
            st.download_button(
                "Download annotated video",
                st.session_state["last_video"],
                file_name=st.session_state.get("last_video_name", "annotated.mp4"),
                mime="video/mp4",
            )
        if "last_counts" in st.session_state:
            counts = st.session_state["last_counts"]
            if mode == "Video":
                st.caption(f"Frames processed: {st.session_state.get('last_frame_count', 0)} · Counts are detection events across frames.")
            count1, count2, count3 = st.columns(3)
            count1.metric("Compliant", counts.get("compliant", 0))
            count2.metric("Violations", counts.get("non_compliant", 0))
            count3.metric("Regions detected", counts.get("total", 0))
            frame_stats = st.session_state.get("last_frame_stats", {})
            candidates = frame_stats.get("candidate_count", 0)
            uncertain = frame_stats.get("uncertain_count", 0)
            if counts.get("total", 0):
                st.success(f"Confirmed {counts['total']} {task} detection(s) at 95% confidence.")
            elif uncertain:
                st.warning(
                    f"Found {uncertain} review region(s), but none reached the fixed 95% confirmation threshold. "
                    "Red boxes are possible violations; green boxes are possible compliant detections. "
                    "Red review crops are saved to the Evidence Vault when evidence capture is on."
                )
            elif candidates == 0:
                st.info(
                    "No face/head/person region was found. This is inconclusive—not confirmation that the person is compliant. "
                    "For helmet checks, upload a clear photo with the person's head visible."
                )
            if save_evidence:
                st.success(
                    "Red violation crops are saved locally. Review candidates are labeled as review candidates in the Evidence Vault."
                )

with evidence_tab:
    st.markdown('<div class="section-label">Recent violation captures</div>', unsafe_allow_html=True)
    filter_task = st.selectbox("Filter evidence", ["All tasks", "Face mask", "Safety helmet"], key="evidence-filter")
    selected_filter = {"All tasks": None, "Face mask": "mask", "Safety helmet": "helmet"}[filter_task]
    events = store.recent_events(100, selected_filter)
    st.caption(f"Only face/head crops are shown here. Photos and records expire after {RETENTION_DAYS} days.")
    if not events:
        st.info("No evidence yet. Run an image, video, or webcam analysis with evidence capture enabled.")
    else:
        for start in range(0, len(events), 2):
            columns = st.columns(2, gap="medium")
            for column, event in zip(columns, events[start : start + 2]):
                evidence_path = store.resolve_evidence_path(event["evidence_path"])
                with column:
                    st.markdown('<div class="event-card">', unsafe_allow_html=True)
                    if evidence_path.is_file():
                        st.image(str(evidence_path), width="stretch")
                    else:
                        st.warning("Evidence image file is missing.")
                    st.markdown(f"**{event['class_name'].replace('_', ' ').title()}** · {event['confidence']:.0%}")
                    st.caption(f"{event['task'].title()} · {event['timestamp_utc']} · source: {event['source']}")
                    if st.button("Delete this event and photo", key=f"delete-{event['id']}"):
                        store.delete_event(event["id"])
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

with about_tab:
    st.markdown('<div class="section-label">Model performance · held-out test set</div>', unsafe_allow_html=True)
    if model_path.suffix.lower() == ".pt":
        st.info(
            "Helmet localization and classification use the trained YOLO object detector. "
            "Boxes are green for predicted compliant and red for predicted violations. "
            "Only detections at the fixed 95% confidence threshold are confirmed; lower-confidence boxes are marked REVIEW."
        )
    else:
        metadata_path = model_path.parent / "class_names.json"
        import json

        with open(metadata_path, encoding="utf-8") as file:
            metadata = json.load(file)
        score1, score2 = st.columns(2)
        score1.metric("Test accuracy", f"{metadata['test_accuracy']:.1%}")
        score2.metric("Input crop", f"{metadata.get('image_size', 160)} × {metadata.get('image_size', 160)} px")
        st.write("**Per-class metrics**")
        class_rows = []
        for label in metadata["class_names"]:
            class_rows.append(
                {
                    "Class": label.replace("_", " ").title(),
                    "Precision": metadata.get("test_precision_by_class", {}).get(label),
                    "Recall": metadata.get("test_recall_by_class", {}).get(label),
                    "F1": metadata.get("test_f1_by_class", {}).get(label),
                }
            )
        st.dataframe(class_rows, hide_index=True, width="stretch")
    st.caption(
        "Mask localization uses OpenCV's frontal-face cascade. The fixed 95% cutoff is conservative; "
        "manually review boxes marked REVIEW and validate on representative camera images."
    )
