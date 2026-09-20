import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# PDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.units import mm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PIPELINE_PATH = PROJECT_ROOT / "src" / "pipeline"

sys.path.insert(
    0,
    str(PIPELINE_PATH)
)

from final_pipeline import CMBDetectionPipeline

st.set_page_config(
    page_title="CMB Review",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

def render_html(content):

    cleaned = " ".join(
        line.strip()
        for line in content.splitlines()
        if line.strip()
    )

    st.markdown(
        cleaned,
        unsafe_allow_html=True
    )

CSS = r"""
<style>
* {
    box-sizing: border-box;
}

html, body {
    font-family: Arial, Helvetica, sans-serif;
}

.stApp {
    background: #f7f7f4;
    color: #1d2638;
}

/* Remove default Streamlit chrome */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

section[data-testid="stSidebar"] {
    background: linear-gradient( 180deg,   #202744 0%,  #171c35 100%
    );
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] > div {
    padding: 20px 15px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 24px;
}

.brand-icon {
    width: 48px;
    height: 48px;
    border-radius: 50%;
    background: #2bc5d9;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 25px;
    box-shadow: 0 0 0 5px rgba(43,197,217,0.10);
}

.brand-title {
    color: #ffffff;
    font-size: 18px;
    font-weight: 700;
}

.brand-subtitle {
    color: #8e99b6;
    font-size: 9px;
    letter-spacing: 2px;
    margin-top: 4px;
}

.demo-card {
    background: linear-gradient(
        135deg,
        #313b60,
        #293250
    );

    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 16px;
    margin-bottom: 27px;
}

.demo-label {
    color: #b8c0d2;
    font-size: 9px;
    letter-spacing: 1.5px;
    margin-bottom: 10px;
}

.demo-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #ef7556;
    margin-right: 7px;
}

.demo-title {
    color: white;
    font-size: 15px;
    font-weight: 700;
}

.demo-sub {
    color: #9da7c0;
    font-size: 10px;
    margin-top: 5px;
}

.nav-heading {
    color: #68728e;
    font-size: 9px;
    letter-spacing: 2px;
    margin: 20px 0 9px 4px;
}

.nav-item {
    color: #b5bfd4;
    padding: 11px 12px;
    border-radius: 12px;
    margin-bottom: 4px;
    font-size: 13px;
}

.nav-item.active {
    background:linear-gradient(90deg, rgba(38,188,214,0.26), 
    rgba(38,188,214,0.07)); color: #36c9df;
} 

.safety-box {
    margin-top: 140px;
    border-top:1px solid rgba(255,255,255,0.08);
    padding-top: 18px;
}

.safety-title {
    color: #7f89a2;
    font-size: 9px;
    letter-spacing: 1.8px;
}

.safety-text {
    color: #8e98b1;
    font-size: 10px;
    line-height: 1.55;
    margin-top: 9px;
}


.user-card {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #2a3353;
    border-radius: 15px;
    padding: 10px;
    margin-top: 20px;
}

.avatar {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background: #8b4d3e;
    color: #ffd7ce;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 700;
}

.user-name {
    color: white;
    font-size: 12px;
    font-weight: 600;
}

.user-role {
    color: #919bb6;
    font-size: 9px;
    margin-top: 3px;
}

.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding:5px 0 17px 0;
    margin-bottom: 28px;
    border-bottom:1px solid #e2e3e1;
}

.breadcrumb {
    color: #707985;
    font-size: 10px;
    letter-spacing: 1.6px;
}

.local-mode {
    color: #707984;
    font-size: 11px;
}

.local-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #4da982;
    margin-right: 6px;
}

.eyebrow {
    color: #2e8998;
    font-size: 9px;
    letter-spacing: 2px;
    margin-bottom: 10px;
}

.eyebrow-dot {
    color: #ef7556;
    margin-right: 7px;
}

.hero-title {
    color: #1d273a;
    font-size: 43px;
    line-height: 1.04;
    font-weight: 700;
    letter-spacing: -1.5px;
    margin: 0;
}

.hero-title span {
    color: #218e9f;
}

.hero-subtitle {
    color: #747c86;
    font-size: 13px;
    line-height: 1.6;
    margin-top: 14px;
    max-width: 720px;
}

[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 1px dashed #b7c0c7;
    border-radius: 18px;
    padding: 12px;
}

div.stButton > button {
    border: 1px solid #d9dbdc;
    background: #ffffff;
    color: #333b49;
    border-radius: 24px;
    min-height: 42px;
    font-weight: 600;
}

div.stButton > button:hover {
    border-color: #239db0;
    color: #18879a;
}

.metric-grid {
    display: grid;
    grid-template-columns:repeat(4, minmax(0, 1fr));
    gap: 14px;
    margin-top: 22px;
    margin-bottom: 18px;
}

.metric-card {
    background: #ffffff;
    border: 1px solid #dedfdd;
    border-radius: 17px;
    padding: 19px;
    min-height: 120px;
}

.metric-label {
    color: #7e858e;
    font-size: 8px;
    letter-spacing: 1.5px;
}

.metric-value {
    color: #202a3d;
    font-size: 28px;
    font-weight: 700;
    margin-top: 13px;
}

.metric-small {
    color: #8b9199;
    font-size: 9px;
    margin-top: 5px;
}

.warning {
    background: #fff0e9;
    border: 1px solid #efd2c6;
    border-radius: 18px;
    padding: 15px 18px;
    margin: 10px 0 22px 0;
}

.warning-title {
    color: #a85d47;
    font-size: 12px;
    font-weight: 700;
}

.warning-text {
    color: #8e7065;
    font-size: 10px;
    line-height: 1.5;
    margin-top: 5px;
}

.card {
    background: #ffffff;
    border: 1px solid #dedfdd;
    border-radius: 19px;
    padding: 19px;
    margin-bottom: 17px;
}

.kicker {
    color: #318997;
    font-size: 8px;
    letter-spacing: 2px;
    margin-bottom: 7px;
}

.card-title {
    color: #202a3c;
    font-size: 20px;
    font-weight: 700;
}

.candidate-card {
    background: #ffffff;
    border: 1px solid #e0e2e3;
    border-radius: 13px;
    padding: 15px;
    margin: 7px 0;
}

.candidate-card.selected {
    background: #eff8f9;
    border-color: #a9d8df;
}

.candidate-id {
    color: #ef7556;
    font-size: 8px;
    letter-spacing: 1.3px;
}

.candidate-name {
    color: #263045;
    font-size: 14px;
    font-weight: 700;
    margin-top: 4px;
}

.candidate-type {
    color: #727a84;
    font-size: 10px;
    margin-top: 4px;
}

.candidate-confidence {
    color: #263045;
    font-size: 14px;
    font-weight: 700;
}

.progress {
    height: 5px;
    background: #e4e6e8;
    border-radius: 5px;
    overflow: hidden;
    margin-top: 9px;
}

.progress-fill {
    height: 100%;
    background: #e97051;
}

.status {
    display: inline-block;
    padding: 5px 9px;
    border-radius: 14px;
    font-size: 8px;
    letter-spacing: .4px;
}

.status-review {
    background: #fff0e8;
    color: #a45d48;
}

.status-reviewed {
    background: #e9f5ef;
    color: #4f8871;
}

.status-positive {
    background: #e8f6f7;
    color: #278596;
}

.evidence-title {
    color: #202a3c;
    font-size: 22px;
    font-weight: 700;
}

.coordinates {
    color: #7e858e;
    font-size: 9px;
    margin-top: 6px;
    letter-spacing: 1px;
}

.evidence-stat-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    margin-top: 14px;
}

.evidence-stat {
    background: #f0f2f5;
    border-radius: 15px;
    padding: 15px;
}

.stat-label {
    color: #858c95;
    font-size: 8px;
    letter-spacing: 1px;
}

.stat-value {
    color: #222b3d;
    font-size: 25px;
    font-weight: 700;
    margin-top: 6px;
}

.sub-label {
    color: #7b828a;
    font-size: 8px;
    letter-spacing: 1.5px;
    margin: 18px 0  7px 0;
}

.class-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
}

.class-option {
    border: 1px solid #d8dbdd;
    border-radius: 13px;
    padding: 12px;
    text-align: center;
    color: #606873;
    font-size: 11px;
}

.class-option.active {
    background: #e9f6f7;
    border-color: #57acbb;
    color: #278696;
    font-weight: 700;
}

.severity-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
}

.severity-option {
    border: 1px solid #d8dbdd;
    border-radius: 13px;
    padding: 11px;
    text-align: center;
    color: #606873;
    font-size: 10px;
}

.severity-option.active {
    background: #fff0e9;
    border-color: #efb9a8;
    color: #b65e47;
    font-weight: 700;
}

.explain {
    border-top: 1px solid #e1e2e2;
    margin-top: 18px;
    padding-top: 14px;
}

.explain-title {
    color: #3a414e;
    font-size: 11px;
    font-weight: 700;
}

.explain-text {
    color: #7d848d;
    font-size: 10px;
    line-height: 1.5;
    margin-top: 6px;
}

.viewer-title {
    color: #293245;
    font-size: 15px;
    font-weight: 700;
    margin-bottom: 12px;
}

.footer {
    color: #858b93;
    text-align: center;
    font-size: 9px;
    padding: 25px 0;
}

@media (max-width: 1000px) {

    .metric-grid {
        grid-template-columns:
            repeat(2, 1fr);
    }

    .hero-title {
        font-size: 35px;
    }
}
</style>
"""

st.markdown(
    "<style>" + CSS + "</style>",
    unsafe_allow_html=True
)

if "results" not in st.session_state:
    st.session_state.results = None

if "volume" not in st.session_state:
    st.session_state.volume = None

if "scan_name" not in st.session_state:
    st.session_state.scan_name = ""

if "selected_candidate" not in st.session_state:
    st.session_state.selected_candidate = 0

if "reviewed" not in st.session_state:
    st.session_state.reviewed = set()

@st.cache_resource
def load_pipeline():

    return CMBDetectionPipeline()


try:

    pipeline = load_pipeline()

except Exception as error:
    st.error("Unable to load the CMB detection pipeline.")
    st.exception(error)
    st.stop()

with st.sidebar:

    render_html(
        """
        <div class="brand">
            <div class="brand-icon">🧠</div>
            <div>
                <div class="brand-title">CMB Review</div>
                <div class="brand-subtitle">NEUROIMAGING LAB</div>
            </div>
        </div>
        """
    )


    render_html(
        """
        <div class="demo-card">
            <div class="demo-label">
                <span class="demo-dot"></span>
                DEMO ANALYSIS
            </div>
            <div class="demo-title">VALDO Review</div>
            <div class="demo-sub">
                SWI · Research inference session
            </div>
        </div>
        """
    )


    render_html(
        """
        <div class="nav-heading">WORKSPACE</div>
        <div class="nav-item active">◫ &nbsp; Review workspace</div>
        <div class="nav-item">⚗ &nbsp; Methodology</div>
        <div class="nav-item">▣ &nbsp; Literature map</div>
        """
    )


    render_html(
        """
        <div class="safety-box">
            <div class="safety-title">◇ &nbsp; SAFETY FIRST</div>
            <div class="safety-text">
                Research interface only.
                Outputs require qualified clinical review.
            </div>
        </div>

        <div class="user-card">
            <div class="avatar">KS</div>
            <div>
                <div class="user-name">Keerthana S</div>
                <div class="user-role">Capstone Team</div>
            </div>
        </div>
        """
    )

render_html(
    """
    <div class="topbar">
        <div class="breadcrumb">CMB &nbsp;/&nbsp; REVIEW</div>
        <div class="local-mode">
            <span class="local-dot"></span>
            Local demo mode
        </div>
    </div>
    """
)

render_html(
    """
    <div class="eyebrow">
        <span class="eyebrow-dot">●</span>
        REVIEW WORKSPACE / RESEARCH
    </div>

    <div class="hero-title">
        Read the evidence,<br>
        <span>not just the output.</span>
    </div>

    <div class="hero-subtitle">
        A research review surface for cerebral microbleed
        candidates, mimics, uncertainty and model attention.
    </div>
    """
)

st.markdown(
    "<div style='height:22px'></div>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload SWI brain MRI scan (.nii.gz)",
    type=["gz"],
    help="Upload a T2S/SWI NIfTI scan."
)

button1, button2, empty = st.columns(
    [1.2, 1.2, 4]
)


with button1:

    rerun = st.button(
        "↻ Re-run analysis",
        use_container_width=True
    )


with button2:

    export_report = st.button(
        "⇩ Export report",
        use_container_width=True
    )

if uploaded_file is not None:

    new_scan = (
        st.session_state.scan_name
        != uploaded_file.name
    )

    if new_scan or rerun or st.session_state.results is None:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".nii.gz"
        ) as tmp:

            tmp.write(
                uploaded_file.getvalue()
            )

            tmp_path = tmp.name


        progress = st.progress(
            0,
            text="Preparing scan..."
        )


        def update_progress(value):

            value = max(
                0.0,
                min(
                    1.0,
                    float(value)
                )
            )

            progress.progress(
                value,
                text=(
                    f"Running CMB pipeline... "
                    f"{int(value * 100)}%"
                )
            )


        try:
            volume = pipeline.preprocess(
                tmp_path
            )

            detections = pipeline.detect(
                volume,
                progress_callback=update_progress
            )

            severity = pipeline.grade_severity(
                len(detections)
            )

            st.session_state.volume = volume

            st.session_state.results = {
                "detections": detections,
                "severity": severity,
            }

            st.session_state.scan_name = (
                uploaded_file.name
            )

            st.session_state.selected_candidate = 0

            st.session_state.reviewed = set()


            progress.progress(
                1.0,
                text="Analysis complete"
            )


        except Exception as error:

            st.error(
                "The CMB inference pipeline failed."
            )

            st.exception(error)

            st.stop()

if st.session_state.results is None:

    render_html(
        """
        <div class="warning">
            <div class="warning-title">
                ⓘ &nbsp; Demo analysis — not for clinical use
            </div>
            <div class="warning-text">
                Upload an SWI scan to begin the research
                inference workflow. This prototype does not
                replace radiologist judgment.
            </div>
        </div>
        """
    )


    render_html(
        """
        <div class="footer">
            CMB Review · Research prototype · VALDO-based development
        </div>
        """
    )

    st.stop()

detections = st.session_state.results[
    "detections"
]

severity = st.session_state.results[
    "severity"
]


# ============================================================
# SUMMARY
# ============================================================

candidate_count = len(
    detections
)


review_count = sum(
    1
    for d in detections
    if d.get("confidence_label")
    == "REVIEW RECOMMENDED"
)


if detections:

    mean_confidence = float(
        np.mean(
            [
                d["confidence"]
                for d in detections
            ]
        )
    )

else:

    mean_confidence = 0.0


reviewed_count = len(
    st.session_state.reviewed
)


# ============================================================
# SUMMARY CARDS
# ============================================================

render_html(
    f"""
    <div class="metric-grid">

        <div class="metric-card">
            <div class="metric-label">CANDIDATES</div>
            <div class="metric-value">
                {candidate_count:02d}
            </div>
            <div class="metric-small">
                Stage-2 accepted candidates
            </div>
        </div>

        <div class="metric-card">
            <div class="metric-label">REVIEWED</div>
            <div class="metric-value">
                {reviewed_count:02d}
            </div>
            <div class="metric-small">
                {max(candidate_count - reviewed_count, 0)}
                still need attention
            </div>
        </div>

        <div class="metric-card">
            <div class="metric-label">MEAN CONFIDENCE</div>
            <div class="metric-value">
                {mean_confidence * 100:.1f}%
            </div>
            <div class="metric-small">
                Candidate-level estimate
            </div>
        </div>

        <div class="metric-card">
            <div class="metric-label">SCAN</div>
            <div class="metric-value"
                 style="font-size:17px;">
                {st.session_state.scan_name[:24]}
            </div>
            <div class="metric-small">
                Local inference
            </div>
        </div>

    </div>
    """
)


# ============================================================
# WARNING
# ============================================================

render_html(
    """
    <div class="warning">
        <div class="warning-title">
            ⓘ &nbsp; Demo analysis — not for clinical use
        </div>
        <div class="warning-text">
            This application demonstrates a research workflow
            using model-generated findings. It is not a medical
            diagnostic device and does not replace qualified
            radiologist judgment.
        </div>
    </div>
    """
)


# ============================================================
# MAIN COLUMNS
# ============================================================

left, right = st.columns(
    [1.0, 1.05],
    gap="large"
)


# ============================================================
# LEFT — CANDIDATE QUEUE
# ============================================================

with left:

    render_html(
        f"""
        <div class="card">
            <div class="kicker">
                ● CANDIDATE QUEUE
            </div>
            <div class="card-title">
                Findings
                <span style="
                    color:#8a9199;
                    font-size:12px;
                    font-weight:400;
                ">
                    &nbsp; {candidate_count}/{candidate_count}
                </span>
            </div>
        </div>
        """
    )


    if not detections:

        st.info(
            "No candidate detections were produced."
        )

    else:

        for index, detection in enumerate(
            detections
        ):

            x, y, z = detection[
                "center"
            ]

            confidence = float(
                detection["confidence"]
            )

            confidence_label = detection.get(
                "confidence_label",
                "High confidence"
            )


            is_selected = (
                index
                == st.session_state.selected_candidate
            )


            is_reviewed = (
                detection["id"]
                if "id" in detection
                else f"CMB-{index + 1:02d}"
            ) in st.session_state.reviewed


            candidate_id = (
                detection.get(
                    "id",
                    f"CMB-{index + 1:02d}"
                )
            )


            if is_reviewed:

                status_text = "REVIEWED"
                status_class = "status-reviewed"

            elif confidence_label == "REVIEW RECOMMENDED":

                status_text = "NEEDS REVIEW"
                status_class = "status-review"

            else:

                status_text = "NEEDS REVIEW"
                status_class = "status-review"


            card_class = (
                "candidate-card selected"
                if is_selected
                else "candidate-card"
            )


            render_html(
                f"""
                <div class="{card_class}">

                    <div style="
                        display:flex;
                        justify-content:space-between;
                        align-items:flex-start;
                    ">

                        <div>

                            <div class="candidate-id">
                                {candidate_id}
                            </div>

                            <div class="candidate-name">
                                Candidate {index + 1}
                            </div>

                            <div class="candidate-type">
                                Stage-2 accepted candidate
                            </div>

                        </div>

                        <div style="text-align:right;">

                            <span class="
                                status {status_class}
                            ">
                                {status_text}
                            </span>

                            <div class="
                                candidate-confidence
                            ">
                                {confidence * 100:.0f}% conf.
                            </div>

                        </div>

                    </div>

                    <div class="progress">

                        <div
                            class="progress-fill"
                            style="
                                width:
                                {confidence * 100:.0f}%;
                            "
                        ></div>

                    </div>

                </div>
                """
            )


            if st.button(
                f"Open {candidate_id}",
                key=f"candidate_{index}",
                use_container_width=True
            ):

                st.session_state.selected_candidate = (
                    index
                )

                st.rerun()


# ============================================================
# RIGHT — REVIEW WORKSPACE
# ============================================================

with right:

    if detections:

        selected_index = min(
            st.session_state.selected_candidate,
            len(detections) - 1
        )


        selected = detections[
            selected_index
        ]


        selected_id = selected.get(
            "id",
            f"CMB-{selected_index + 1:02d}"
        )


        x, y, z = selected[
            "center"
        ]


        confidence = float(
            selected["confidence"]
        )


        # ====================================================
        # VIEWER CONTROLS
        # ====================================================

        render_html(
            """
            <div class="card">
                <div class="kicker">
                    ● VIEWER CONTROLS
                </div>

                <div class="viewer-title">
                    Overlays
                </div>
            </div>
            """
        )


        heatmap_on = st.toggle(
            "Attention heatmap",
            value=True,
            key="heatmap"
        )


        marker_on = st.toggle(
            "Candidate markers",
            value=True,
            key="markers"
        )


        crosshair_on = st.toggle(
            "Crosshair guide",
            value=False,
            key="crosshair"
        )


        # ====================================================
        # EVIDENCE CARD
        # ====================================================

        render_html(
            f"""
            <div class="card">

                <div class="kicker">
                    ● EVIDENCE DETAIL
                </div>

                <div class="evidence-title">
                    {selected_id}
                </div>

                <div class="coordinates">
                    VOXEL &nbsp; {x} / {y} / {z}
                </div>

            """
        )


        # ====================================================
        # MRI IMAGE
        # ====================================================

        volume = st.session_state.volume


        if volume is not None:

            slice_index = int(
                np.clip(
                    z,
                    0,
                    volume.shape[2] - 1
                )
            )


            image_slice = volume[
                :,
                :,
                slice_index
            ]


            fig, ax = plt.subplots(
                figsize=(6.5, 5)
            )


            ax.imshow(
                image_slice.T,
                cmap="gray",
                origin="lower"
            )


            # ------------------------------------------------
            # MARKERS
            # ------------------------------------------------

            if marker_on:

                for candidate in detections:

                    cx, cy, cz = candidate[
                        "center"
                    ]


                    if int(cz) == slice_index:

                        radius = (
                            7
                            if candidate is selected
                            else 5
                        )


                        circle = Circle(
                            (
                                cx,
                                cy
                            ),
                            radius=radius,
                            fill=False,
                            linewidth=2,
                            edgecolor="red"
                        )


                        ax.add_patch(
                            circle
                        )


            # ------------------------------------------------
            # CROSSHAIR
            # ------------------------------------------------

            if crosshair_on:

                ax.axvline(
                    x,
                    linestyle="--",
                    linewidth=0.8
                )

                ax.axhline(
                    y,
                    linestyle="--",
                    linewidth=0.8
                )


            ax.set_title(
                f"SWI · Slice {slice_index}",
                fontsize=10
            )


            ax.axis("off")


            st.pyplot(
                fig,
                use_container_width=True
            )


            plt.close(fig)


        # ====================================================
        # HEATMAP INFORMATION
        # ====================================================

        if heatmap_on:

            render_html(
                """
                <div style="
                    background:#f1f5f7;
                    border-radius:14px;
                    padding:12px;
                    margin-top:10px;
                ">
                    <div style="
                        color:#3b4756;
                        font-size:11px;
                        font-weight:700;
                    ">
                        ✧ Explainability view
                    </div>

                    <div style="
                        color:#7c858e;
                        font-size:9px;
                        margin-top:5px;
                        line-height:1.5;
                    ">
                        The selected candidate is shown with
                        its model confidence and spatial
                        evidence. Grad-CAM visualization can
                        be connected to the existing
                        severity/Grad-CAM module separately.
                    </div>
                </div>
                """
            )


        # ====================================================
        # STATS
        # ====================================================

        render_html(
            f"""
            <div class="evidence-stat-grid">

                <div class="evidence-stat">
                    <div class="stat-label">
                        CONFIDENCE
                    </div>
                    <div class="stat-value">
                        {confidence * 100:.0f}%
                    </div>
                </div>

                <div class="evidence-stat">
                    <div class="stat-label">
                        SLICE
                    </div>
                    <div class="stat-value">
                        {slice_index}
                    </div>
                </div>

            </div>
            """
        )


        # ====================================================
        # CLASSIFICATION
        # ====================================================

        render_html(
            """
            <div class="sub-label">
                CLASSIFICATION
            </div>

            <div class="class-grid">

                <div class="class-option active">
                    Stage-2 candidate
                </div>

                <div class="class-option">
                    Mimic rejected
                </div>

            </div>
            """
        )


        # ====================================================
        # SEVERITY
        # ====================================================

        if severity == "None":

            severity_display = "Low"

        elif severity == "Mild":

            severity_display = "Low"

        elif severity == "Moderate":

            severity_display = "Moderate"

        else:

            severity_display = "High"


        render_html(
            f"""
            <div class="sub-label">
                REVIEW SEVERITY
            </div>

            <div class="severity-grid">

                <div class="
                    severity-option
                    {"active" if severity_display == "Low" else ""}
                ">
                    Low
                </div>

                <div class="
                    severity-option
                    {"active" if severity_display == "Moderate" else ""}
                ">
                    Moderate
                </div>

                <div class="
                    severity-option
                    {"active" if severity_display == "High" else ""}
                ">
                    High
                </div>

            </div>
            """
        )


        # ====================================================
        # EXPLAINABILITY NOTE
        # ====================================================

        render_html(
            """
            <div class="explain">

                <div class="explain-title">
                    ✧ Explainability note
                </div>

                <div class="explain-text">
                    Model attention is intended to indicate
                    regions associated with the prediction.
                    Attention maps do not establish causality
                    or clinical significance.
                </div>

            </div>

            </div>
            """
        )


        # ====================================================
        # REVIEW BUTTONS
        # ====================================================

        review_col1, review_col2 = st.columns(2)


        with review_col1:

            if st.button(
                "✓ Mark as reviewed",
                use_container_width=True
            ):

                st.session_state.reviewed.add(
                    selected_id
                )

                st.rerun()


        with review_col2:

            if st.button(
                "Next candidate →",
                use_container_width=True
            ):

                st.session_state.selected_candidate = (
                    (
                        selected_index + 1
                    )
                    % len(detections)
                )

                st.rerun()


# ============================================================
# PDF REPORT
# ============================================================

def create_pdf_report():

    from io import BytesIO

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )


    styles = getSampleStyleSheet()

    title_style = styles["Title"]

    heading_style = styles["Heading2"]

    normal_style = styles["BodyText"]


    story = []


    story.append(
        Paragraph(
            "CMB Review — Research Report",
            title_style
        )
    )


    story.append(
        Spacer(
            1,
            8
        )
    )


    story.append(
        Paragraph(
            f"Scan: {st.session_state.scan_name}",
            normal_style
        )
    )


    story.append(
        Paragraph(
            "Research prototype — not for clinical use.",
            normal_style
        )
    )


    story.append(
        Spacer(
            1,
            12
        )
    )


    story.append(
        Paragraph(
            "Summary",
            heading_style
        )
    )


    story.append(
        Paragraph(
            f"Candidate count: {len(detections)}",
            normal_style
        )
    )


    story.append(
        Paragraph(
            f"Mean confidence: "
            f"{mean_confidence * 100:.1f}%",
            normal_style
        )
    )


    story.append(
        Paragraph(
            f"Severity: {severity}",
            normal_style
        )
    )


    story.append(
        Spacer(
            1,
            12
        )
    )


    story.append(
        Paragraph(
            "Candidate Findings",
            heading_style
        )
    )


    table_data = [
        [
            "Candidate",
            "X",
            "Y",
            "Z",
            "Confidence",
            "Status"
        ]
    ]


    for i, detection in enumerate(
        detections
    ):

        cx, cy, cz = detection[
            "center"
        ]


        table_data.append(
            [
                detection.get(
                    "id",
                    f"CMB-{i + 1:02d}"
                ),
                str(cx),
                str(cy),
                str(cz),
                f"{detection['confidence'] * 100:.1f}%",
                detection.get(
                    "confidence_label",
                    "High confidence"
                ),
            ]
        )


    table = Table(
        table_data,
        repeatRows=1
    )


    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#25304a"
                    )
                ),

                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.HexColor(
                        "#cccccc"
                    )
                ),

                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE"
                ),
            ]
        )
    )


    story.append(
        table
    )


    story.append(
        Spacer(
            1,
            16
        )
    )


    story.append(
        Paragraph(
            "Clinical safety note",
            heading_style
        )
    )


    story.append(
        Paragraph(
            "This report is generated by a research "
            "prototype. Model predictions require "
            "qualified clinical review and should not "
            "be used as an independent diagnosis.",
            normal_style
        )
    )


    document.build(
        story
    )


    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# EXPORT REPORT
# ============================================================

if export_report:

    pdf_data = create_pdf_report()


    st.download_button(
        label="⬇ Download CMB PDF Report",
        data=pdf_data,
        file_name=(
            Path(
                st.session_state.scan_name
            ).stem
            + "_CMB_Report.pdf"
        ),
        mime="application/pdf",
    )


# ============================================================
# FOOTER
# ============================================================

render_html(
    """
    <div class="footer">
        CMB Review · Research interface only ·
        VALDO-based development
        <br>
        Model attention indicates focus and does not establish
        causality or clinical significance.
    </div>
    """
)