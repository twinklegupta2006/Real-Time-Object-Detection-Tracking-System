"""Streamlit UI for the real-time object detection / tracking / counting
pipeline.

Thin orchestration only: every actual processing decision goes through
``src/app_core.run_pipeline``, which just wires together the existing
Config/detector/tracker/counter/pipeline classes from Phases 1-5, unchanged.
No detection, tracking, or counting logic lives in this file.

Run with:
    streamlit run streamlit_app.py
"""
import os
import tempfile

import streamlit as st

from src.app_core import COCO_CLASSES, VOC_CLASSES, AppConfigError, run_pipeline

st.set_page_config(page_title="Object Detection, Tracking & Counting", layout="wide")

st.title("Real-Time Object Detection, Tracking & Counting")
st.caption(
    "MobileNet-SSD or YOLOv8n detection, optional ByteTrack tracking, "
    "optional line-crossing counting. Upload a video and run it below."
)

with st.sidebar:
    st.header("Configuration")

    model_label = st.selectbox("Detector", ["YOLOv8n", "MobileNet-SSD"])
    model = "yolov8" if model_label == "YOLOv8n" else "mobilenet"

    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.5, 0.05)
    max_frames = st.number_input(
        "Max frames to process (0 = entire video)", min_value=0, value=300, step=50,
    )

    st.divider()
    st.subheader("Tracking")
    enable_tracking = st.checkbox("Enable ByteTrack tracking", value=False)

    st.subheader("Counting")
    enable_counting = st.checkbox(
        "Enable line-crossing counting", value=False,
        help="Automatically enables tracking too -- counting needs persistent track IDs.",
    )

    line_orientation = "horizontal"
    line_position = 0.5
    in_direction = None
    count_classes: list = []

    if enable_counting:
        if not enable_tracking:
            st.caption("Tracking auto-enabled: counting requires it.")
        line_orientation = st.radio("Line orientation", ["horizontal", "vertical"], horizontal=True)
        line_position = st.slider("Line position (fraction of frame)", 0.0, 1.0, 0.5, 0.05)

        if line_orientation == "horizontal":
            direction_options, default_index = ["down", "up"], 0
        else:
            direction_options, default_index = ["left", "right"], 1
        in_direction = st.selectbox("Direction counted as IN", direction_options, index=default_index)

        available_classes = VOC_CLASSES if model == "mobilenet" else COCO_CLASSES
        count_classes = st.multiselect(
            "Only count these classes (empty = count every class)",
            options=sorted(available_classes), default=[],
        )

st.subheader("1. Upload a video")
uploaded_file = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv"])

run_clicked = st.button("Run pipeline", type="primary", disabled=uploaded_file is None)

if run_clicked and uploaded_file is not None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = os.path.join(tmp_dir, uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        output_dir = os.path.join(tmp_dir, "output")

        with st.spinner("Running pipeline... this can take a while for long videos."):
            try:
                result = run_pipeline(
                    source_video_path=input_path,
                    model=model,
                    enable_tracking=enable_tracking,
                    enable_counting=enable_counting,
                    confidence=confidence,
                    max_frames=int(max_frames),
                    output_dir=output_dir,
                    line_orientation=line_orientation,
                    line_position=line_position,
                    in_direction=in_direction,
                    count_classes=count_classes or None,
                )
            except AppConfigError as exc:
                st.error(str(exc))
                st.stop()
            except Exception as exc:  # unexpected failure -- still a message, not a raw traceback
                st.error(f"Pipeline failed: {exc}")
                st.stop()

        st.success("Done.")

        st.subheader("2. Annotated output")
        preview_path = result.preview_video_path or result.video_path
        if preview_path and os.path.isfile(preview_path):
            with open(preview_path, "rb") as f:
                video_bytes = f.read()
            st.video(video_bytes)
            if result.preview_video_path is None:
                st.caption(
                    "Preview conversion to a browser-friendly format wasn't available, so this "
                    "is the pipeline's native output -- some browsers may not play it. The "
                    "download below works regardless."
                )
        else:
            st.info("No annotated video was produced for this run.")

        if result.video_path and os.path.isfile(result.video_path):
            with open(result.video_path, "rb") as f:
                st.download_button(
                    "Download annotated video", f, file_name=os.path.basename(result.video_path),
                )

        st.subheader("3. Summary")
        st.text(result.summary_text)

        st.subheader("4. Downloads")
        col1, col2 = st.columns(2)
        with col1:
            if result.csv_path and os.path.isfile(result.csv_path):
                with open(result.csv_path, "rb") as f:
                    st.download_button(
                        "Download detection/tracking CSV", f, file_name=os.path.basename(result.csv_path),
                    )
        with col2:
            if result.counting_csv_path and os.path.isfile(result.counting_csv_path):
                with open(result.counting_csv_path, "rb") as f:
                    st.download_button(
                        "Download counting events CSV", f,
                        file_name=os.path.basename(result.counting_csv_path),
                    )
