"""
Phase 6: The doctor-facing dashboard. Upload a scan, run the full pipeline,
see results, download a PDF report.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src" / "pipeline"))
from final_pipeline import CMBDetectionPipeline

st.set_page_config(page_title="CMB Detection Dashboard", layout="wide")
st.title("🧠 Cerebral Microbleed Detection Dashboard")
st.caption("Upload an SWI brain MRI scan (.nii.gz) to detect and localize cerebral microbleeds.")

@st.cache_resource
def load_pipeline():
    return CMBDetectionPipeline()

pipeline = load_pipeline()

uploaded_file = st.file_uploader("Upload SWI scan (.nii.gz)", type=["nii.gz", "gz"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.info("Preprocessing scan (bias correction, normalization, resampling)... this may take a moment.")
    volume = pipeline.preprocess(tmp_path)
    st.success(f"Preprocessing complete. Volume shape: {volume.shape}")

    progress_bar = st.progress(0.0, text="Running detection pipeline...")

    def update_progress(fraction):
        progress_bar.progress(fraction, text=f"Running detection pipeline... {int(fraction * 100)}%")

    detections = pipeline.detect(volume, progress_callback=update_progress)
    progress_bar.progress(1.0, text="Done")

    severity = pipeline.grade_severity(len(detections))

    col1, col2, col3 = st.columns(3)
    col1.metric("Detections found", len(detections))
    col2.metric("Severity grade", severity)
    review_count = sum(1 for d in detections if d["confidence_label"] == "REVIEW RECOMMENDED")
    col3.metric("Flagged for review", review_count)

    st.subheader("Detected Lesions")
    if detections:
        import pandas as pd
        df = pd.DataFrame([{
            "X": d["center"][0], "Y": d["center"][1], "Z (slice)": d["center"][2],
            "Confidence": d["confidence"], "Status": d["confidence_label"],
        } for d in detections])
        st.dataframe(df, use_container_width=True)

        st.subheader("Visualization")
        slice_options = sorted(set(d["center"][2] for d in detections))
        selected_slice = st.selectbox("View slice", slice_options)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(volume[:, :, selected_slice].T, cmap="gray", origin="lower")
        for d in detections:
            if d["center"][2] == selected_slice:
                x, y, _ = d["center"]
                color = "red" if d["confidence_label"] == "REVIEW RECOMMENDED" else "yellow"
                circle = plt.Circle((x, y), 4, color=color, fill=False, linewidth=2)
                ax.add_patch(circle)
        ax.set_title(f"Slice {selected_slice}")
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.info("No microbleeds detected in this scan.")

    st.session_state["last_results"] = {
        "detections": detections, "severity": severity, "filename": uploaded_file.name,
    }
else:
    st.info("👆 Upload a scan to begin.")