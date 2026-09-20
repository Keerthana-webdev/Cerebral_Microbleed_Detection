"""
============================================================
CMB REVIEW
Neuroimaging Lab
Streamlit Research Dashboard
============================================================

Visual design inspired by the supplied CMB Review reference UI.

REAL PIPELINE:
SWI
 -> preprocessing
 -> Stage 1 CNN
 -> Stage 2 CNN
 -> NMS
 -> confidence
 -> severity
 -> Grad-CAM
 -> PDF report

Research prototype only.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"

PIPELINE_DIR = (
    SRC_DIR / "pipeline"
)

GRADCAM_DIR = (
    SRC_DIR / "severity_gradcam"
)

sys.path.insert(
    0,
    str(PIPELINE_DIR)
)

sys.path.insert(
    0,
    str(GRADCAM_DIR)
)

sys.path.insert(
    0,
    str(PROJECT_ROOT / "dashboard")
)


from final_pipeline import CMBDetectionPipeline

from gradcam_utils import (
    generate_gradcam,
    create_attention_overlay
)

from report_generator import (
    generate_pdf_report
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CMB Review",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

@import url(
    'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap'
);


/* ---------------------------------------------------------
   GLOBAL
--------------------------------------------------------- */

html,
body,
[class*="css"] {

    font-family:
        'DM Sans',
        sans-serif;

}


.stApp {

    background:
        #f7f7f4;

    color:
        #172033;

}


/* ---------------------------------------------------------
   HIDE STREAMLIT DEFAULT UI
--------------------------------------------------------- */

#MainMenu {

    visibility:
        hidden;

}


footer {

    visibility:
        hidden;

}


header {

    visibility:
        hidden;

}


/* ---------------------------------------------------------
   SIDEBAR
--------------------------------------------------------- */

section[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            #202743 0%,
            #161b34 100%
        );

    border-right:
        1px solid
        rgba(
            255,
            255,
            255,
            0.08
        );

}


section[data-testid="stSidebar"]
> div {

    padding:
        18px 16px;

}


/* ---------------------------------------------------------
   BRAND
--------------------------------------------------------- */

.brand {

    display:
        flex;

    align-items:
        center;

    gap:
        12px;

    margin-bottom:
        24px;

}


.brand-icon {

    width:
        48px;

    height:
        48px;

    border-radius:
        50%;

    background:
        linear-gradient(
            135deg,
            #28c6db,
            #1e9db9
        );

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    font-size:
        25px;

    box-shadow:
        0 0 0 5px
        rgba(
            40,
            198,
            219,
            0.08
        );

}


.brand-title {

    color:
        white;

    font-size:
        18px;

    font-weight:
        700;

    line-height:
        1.1;

}


.brand-subtitle {

    color:
        #8f9bb8;

    font-family:
        'Space Mono',
        monospace;

    font-size:
        9px;

    letter-spacing:
        2px;

    margin-top:
        5px;

}


/* ---------------------------------------------------------
   DEMO CARD
--------------------------------------------------------- */

.demo-card {

    background:
        linear-gradient(
            135deg,
            #313b60,
            #293251
        );

    border:
        1px solid
        rgba(
            255,
            255,
            255,
            0.08
        );

    border-radius:
        17px;

    padding:
        16px;

    margin:
        8px 0 28px 0;

}


.demo-label {

    color:
        #b6bfd4;

    font-family:
        'Space Mono',
        monospace;

    font-size:
        9px;

    letter-spacing:
        1.5px;

    margin-bottom:
        10px;

}


.demo-dot {

    display:
        inline-block;

    width:
        7px;

    height:
        7px;

    border-radius:
        50%;

    background:
        #ef7656;

    margin-right:
        7px;

}


.demo-title {

    color:
        white;

    font-weight:
        700;

    font-size:
        15px;

}


.demo-sub {

    color:
        #9ca8c4;

    font-size:
        11px;

    margin-top:
        5px;

}


/* ---------------------------------------------------------
   SIDEBAR NAV
--------------------------------------------------------- */

.nav-label {

    color:
        #67718c;

    font-family:
        'Space Mono',
        monospace;

    font-size:
        9px;

    letter-spacing:
        2px;

    margin:
        20px 0 9px 4px;

}


.nav-item {

    color:
        #b6bfd4;

    padding:
        11px 12px;

    border-radius:
        12px;

    margin-bottom:
        4px;

    font-size:
        13px;

}


.nav-item.active {

    background:
        linear-gradient(
            90deg,
            rgba(
                31,
                190,
                216,
                0.25
            ),
            rgba(
                31,
                190,
                216,
                0.08
            )
        );

    color:
        #34c8df;

}


.nav-icon {

    margin-right:
        10px;

}


/* ---------------------------------------------------------
   SAFETY
--------------------------------------------------------- */

.safety {

    position:
        absolute;

    bottom:
        92px;

    left:
        18px;

    right:
        18px;

    border-top:
        1px solid
        rgba(
            255,
            255,
            255,
            0.09
        );

    padding-top:
        18px;

}


.safety-title {

    color:
        #77809a;

    font-family:
        'Space Mono',
        monospace;

    font-size:
        9px;

    letter-spacing:
        1.7px;

}


.safety-text {

    color:
        #8e98b2;

    font-size:
        11px;

    line-height:
        1.5;

    margin-top:
        10px;

}


/* ---------------------------------------------------------
   USER
--------------------------------------------------------- */

.user-card {

    position:
        absolute;

    bottom:
        18px;

    left:
        18px;

    right:
        18px;

    background:
        #2a3354;

    border-radius:
        16px;

    padding:
        10px;

    display:
        flex;

    align-items:
        center;

    gap:
        10px;

}


.avatar {

    width:
        38px;

    height:
        38px;

    border-radius:
        50%;

    background:
        #8e4f3e;

    color:
        #ffd4c9;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    font-weight:
        700;

    font-size:
        12px;

}


.user-name {

    color:
        white;

    font-size:
        12px;

    font-weight:
        600;

}


.user-role {

    color:
        #8d98b5;

    font-size:
        10px;

}


/* ---------------------------------------------------------
   TOP BAR
--------------------------------------------------------- */

.topbar {

    display:
        flex;

    justify-content:
        space-between;

    align-items:
        center;

    border-bottom:
        1px solid
        #e3e4e1;

    padding:
        6px 0 18px 0;

    margin-bottom:
        28px;

}


.breadcrumb {

    color:
        #77808d;

    font-family:
        'Space Mono',
        monospace;

    font-size:
        11px;

    letter-spacing:
        1.5px;

}


.local-mode {

    color:
        #707984;

    font-size:
        12px;

}


.local-dot {

    display:
        inline-block;

    width:
        7px;

    height:
        7px;

    border-radius:
        50%;

    background:
        #4ba980;

    margin-right:
        6px;

}


/* ---------------------------------------------------------
   HERO
--------------------------------------------------------- */

.hero-row {

    display:
        flex;

    justify-content:
        space-between;

    align-items:
        flex-start;

    margin-bottom:
        24px;

}


.eyebrow {

    color:
        #2a8795;

    font-family:
        'Space Mono',
        monospace;

    font-size:
        10px;

    letter-spacing:
        2px;

    margin-bottom:
        10px;

}


.eyebrow-dot {

    color:
        #ef7656;

    margin-right:
        7px;

}


.hero-title {

    font-size:
        42px;

    line-height:
        1.05;

    font-weight:
        700;

    letter-spacing:
        -1.5px;

    color:
        #1c263b;

    margin:
        0;

}


.hero-title span {

    color:
        #218fa1;

}


.hero-subtitle {

    color:
        #727a84;

    font-size:
        14px;

    max-width:
        620px;

    margin-top:
        14px;

    line-height:
        1.6;

}


/* ---------------------------------------------------------
   BUTTONS
--------------------------------------------------------- */

div.stButton > button {

    border-radius:
        24px;

    border:
        1px solid
        #d9dbd9;

    background:
        white;

    color:
        #2d3442;

    font-weight:
        600;

    padding:
        9px 18px;

}


div.stButton > button:hover {

    border-color:
        #1e9db9;

    color:
        #16849a;

}


.export-button button {

    background:
        #279cb0 !important;

    color:
        white !important;

    border:
        none !important;

}


/* ---------------------------------------------------------
   SUMMARY CARDS
--------------------------------------------------------- */

.metric-grid {

    display:
        grid;

    grid-template-columns:
        repeat(
            4,
            1fr
        );

    gap:
        14px;

    margin-bottom:
        18px;

}


.metric-card {

    background:
        #ffffff;

    border:
        1px solid
        #e0e1df;

    border-radius:
        17px;

    padding:
        20px;

    min-height:
        120px;

}


.metric-label {

    color:
        #7d838d;

    font-family:
        'Space Mono',
        monospace;

    font-size:
        9px;

    letter-spacing:
        1.4px;

}


.metric-value {

    color:
        #1c2537;

    font-size:
        28px;

    font-weight:
        700;

    margin-top:
        15px;

}


.metric-small {

    color:
        #8b9198;

    font-size:
        10px;

    margin-top:
        4px;

}


/* ---------------------------------------------------------
   SAFETY BANNER
--------------------------------------------------------- */

.warning {

    background:
        #fff0e9;

    border:
        1px solid
        #f0d4c8;

    border-radius:
        18px;

    padding:
        14px 18px;

    margin:
        12px 0 24px 0;

}


.warning-title {

    color:
        #a85b45;

    font-size:
        13px;

    font-weight:
        700;

}


.warning-text {

    color:
        #8c6d62;

    font-size:
        11px;

    margin-top:
        5px;

}


/* ---------------------------------------------------------
   SECTION
--------------------------------------------------------- */

.section-card {

    background:
        white;

    border:
        1px solid
        #dedfdd;

    border-radius:
        20px;

    padding:
        20px;

    margin-bottom:
        18px;

}


.section-kicker {

    color:
        #318998;

    font-family:
        'Space Mono',
        monospace;

    font-size:
        9px;

    letter-spacing:
        2px;

    margin-bottom:
        8px;

}


.section-title {

    color:
        #1b2437;

    font-size:
        21px;

    font-weight:
        700;

}


/* ---------------------------------------------------------
   CANDIDATE CARD
--------------------------------------------------------- */

.candidate {

    border:
        1px solid
        #e1e3e4;

    border-radius:
        12px;

    padding:
        15px;

    margin:
        8px 0;

    background:
        #ffffff;

}


.candidate.selected {

    background:
        #eef7f8;

    border-color:
        #a7d9df;

}


.candidate-header {

    display:
        flex;

    justify-content:
        space-between;

    align-items:
        center;

}


.candidate-id {

    color:
        #ef7656;

    font-family:
        'Space Mono',
        monospace;

    font-size:
        9px;

}


.candidate-name {

    color:
        #263045;

    font-size:
        15px;

    font-weight:
        700;

    margin-top:
        4px;

}


.candidate-type {

    color:
        #69717c;

    font-size:
        11px;

    margin-top:
        5px;

}


.confidence-number {

    color:
        #263045;

    font-size:
        15px;

    font-weight:
        700;

}


.confidence-bar {

    height:
        5px;

    border-radius:
        5px;

    background:
        #e4e7eb;

    margin-top:
        10px;

    overflow:
        hidden;

}


.confidence-fill {

    height:
        100%;

    background:
        #e96f50;

}


/* ---------------------------------------------------------
   STATUS PILLS
--------------------------------------------------------- */

.pill {

    display:
        inline-block;

    padding:
        5px 10px;

    border-radius:
        14px;

    font-size:
        9px;

    font-family:
        'Space Mono',
        monospace;

}


.pill-review {

    background:
        #fff0e8;

    color:
        #a45d48;

}


.pill-reviewed {

    background:
        #e9f5ef;

    color:
        #4e8871;

}


.pill-mimic {

    background:
        #edf0fb;

    color:
        #465895;

}


/* ---------------------------------------------------------
   EVIDENCE
--------------------------------------------------------- */

.evidence-stat {

    background:
        #f0f2f5;

    border-radius:
        16px;

    padding:
        18px;

}


.evidence-label {

    color:
        #808791;

    font-family:
        'Space Mono',
        monospace;

    font-size:
        8px;

    letter-spacing:
        1px;

}


.evidence-value {

    color:
        #222b3e;

    font-size:
        26px;

    font-weight:
        700;

    margin-top:
        8px;

}


/* ---------------------------------------------------------
   CLASSIFICATION
--------------------------------------------------------- */

.class-box {

    border:
        1px solid
        #d8dbdc;

    border-radius:
        14px;

    padding:
        12px;

    text-align:
        center;

    font-size:
        12px;

}


.class-box.active {

    background:
        #e8f6f7;

    border-color:
        #54aebe;

    color:
        #268596;

    font-weight:
        600;

}


/* ---------------------------------------------------------
   SEVERITY
--------------------------------------------------------- */

.severity-box {

    border:
        1px solid
        #d9dcdf;

    border-radius:
        14px;

    padding:
        12px;

    text-align:
        center;

    font-size:
        12px;

}


.severity-box.active {

    background:
        #fff0e9;

    border-color:
        #efb9a7;

    color:
        #b65c45;

    font-weight:
        700;

}


/* ---------------------------------------------------------
   ATTENTION NOTE
--------------------------------------------------------- */

.attention-note {

    border-top:
        1px solid
        #e1e2e2;

    margin-top:
        20px;

    padding-top:
        16px;

}


.attention-title {

    color:
        #3a414e;

    font-weight:
        600;

    font-size:
        12px;

}


.attention-text {

    color:
        #7c828a;

    font-size:
        11px;

    margin-top:
        7px;

}


/* ---------------------------------------------------------
   UPLOAD BOX
--------------------------------------------------------- */

.upload-card {

    background:
        #ffffff;

    border:
        1px dashed
        #aeb6bd;

    border-radius:
        18px;

    padding:
        18px;

    margin-bottom:
        20px;

}


/* ---------------------------------------------------------
   FOOTER
--------------------------------------------------------- */

.footer-note {

    color:
        #858b93;

    font-size:
        10px;

    text-align:
        center;

    padding:
        20px 0;

}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# LOAD PIPELINE
# ============================================================

@st.cache_resource
def load_pipeline():

    return CMBDetectionPipeline()


pipeline = load_pipeline()


# ============================================================
# SESSION STATE
# ============================================================

if "results" not in st.session_state:

    st.session_state.results = None


if "volume" not in st.session_state:

    st.session_state.volume = None


if "scan_name" not in st.session_state:

    st.session_state.scan_name = "No scan loaded"


if "selected_candidate" not in st.session_state:

    st.session_state.selected_candidate = 0


if "reviewed" not in st.session_state:

    st.session_state.reviewed = set()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">

            <div class="brand-icon">
                🧠
            </div>

            <div>
                <div class="brand-title">
                    CMB Review
                </div>

                <div class="brand-subtitle">
                    NEUROIMAGING LAB
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="demo-card">

            <div class="demo-label">
                <span class="demo-dot"></span>
                DEMO ANALYSIS
            </div>

            <div class="demo-title">
                VALDO Review
            </div>

            <div class="demo-sub">
                SWI · Research inference session
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="nav-label">
            WORKSPACE
        </div>

        <div class="nav-item active">
            ◫ &nbsp; Review workspace
        </div>

        <div class="nav-item">
            ⚗ &nbsp; Methodology
        </div>

        <div class="nav-item">
            ◫ &nbsp; Literature map
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="safety">

            <div class="safety-title">
                ◇ &nbsp; SAFETY FIRST
            </div>

            <div class="safety-text">
                Research interface only.
                Outputs require qualified clinical review.
            </div>

        </div>

        <div class="user-card">

            <div class="avatar">
                HV
            </div>

            <div>
                <div class="user-name">
                    Capstone Team
                </div>

                <div class="user-role">
                    Research prototype
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MAIN
# ============================================================

st.markdown(
    """
    <div class="topbar">

        <div class="breadcrumb">
            CMB &nbsp;/&nbsp; REVIEW
        </div>

        <div class="local-mode">
            <span class="local-dot"></span>
            Local demo mode
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero-row">

        <div>

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

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    '<div class="upload-card">',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload SWI brain MRI scan (.nii.gz)",
    type=["gz"],
    help="Upload a T2S/SWI NIfTI scan."
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# BUTTON ROW
# ============================================================

button_col1, button_col2, spacer = st.columns(
    [1.2, 1.2, 4]
)


with button_col1:

    rerun = st.button(
        "↻  Re-run analysis",
        use_container_width=True
    )


with button_col2:

    export_clicked = st.button(
        "⇩  Export report",
        use_container_width=True
    )


# ============================================================
# RUN ANALYSIS
# ============================================================

should_run = (
    uploaded_file is not None
    and (
        rerun
        or st.session_state.results is None
        or st.session_state.scan_name
        != uploaded_file.name
    )
)


if should_run:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".nii.gz"
    ) as temporary_file:

        temporary_file.write(
            uploaded_file.getvalue()
        )

        temporary_path = temporary_file.name


    st.info(
        "Preprocessing SWI scan and running the two-stage "
        "CMB detection pipeline..."
    )


    progress = st.progress(
        0,
        text="Starting analysis..."
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
                f"Running analysis... "
                f"{int(value * 100)}%"
            )
        )


    try:

        volume = pipeline.preprocess(
            temporary_path
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
            "severity": severity
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

        st.success(
            "Analysis completed successfully."
        )

    except Exception as error:

        st.error(
            "Pipeline error:"
        )

        st.exception(
            error
        )


# ============================================================
# NO RESULTS
# ============================================================

if st.session_state.results is None:

    st.markdown(
        """
        <div class="warning">

            <div class="warning-title">
                Demo analysis — not for clinical use
            </div>

            <div class="warning-text">
                Upload an SWI scan to begin the research inference
                workflow. This prototype does not replace
                radiologist judgment.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="footer-note">
            CMB Review · Research prototype · VALDO-based development
        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# RESULTS
# ============================================================

detections = (
    st.session_state.results["detections"]
)

severity = (
    st.session_state.results["severity"]
)


# ============================================================
# SUMMARY VALUES
# ============================================================

candidate_count = len(
    detections
)

review_count = sum(
    1
    for d in detections
    if d["confidence_label"]
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


# ============================================================
# SUMMARY CARDS
# ============================================================

st.markdown(
    f"""
    <div class="metric-grid">

        <div class="metric-card">

            <div class="metric-label">
                CANDIDATES
            </div>

            <div class="metric-value">
                {candidate_count:02d}
            </div>

            <div class="metric-small">
                {sum(
                    d["classification"] == "True microbleed"
                    for d in detections
                )} microbleed ·
                {sum(
                    d["classification"] == "Mimic"
                    for d in detections
                )} mimic
            </div>

        </div>


        <div class="metric-card">

            <div class="metric-label">
                REVIEWED
            </div>

            <div class="metric-value">
                {len(st.session_state.reviewed):02d}
            </div>

            <div class="metric-small">
                {max(
                    candidate_count
                    - len(st.session_state.reviewed),
                    0
                )} still need attention
            </div>

        </div>


        <div class="metric-card">

            <div class="metric-label">
                MEAN CONFIDENCE
            </div>

            <div class="metric-value">
                {mean_confidence * 100:.1f}%
            </div>

            <div class="metric-small">
                Candidate-level estimate
            </div>

        </div>


        <div class="metric-card">

            <div class="metric-label">
                CURRENT SCAN
            </div>

            <div class="metric-value"
                 style="font-size:17px;">
                {st.session_state.scan_name[:24]}
            </div>

            <div class="metric-small">
                Local inference
            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# WARNING
# ============================================================

st.markdown(
    """
    <div class="warning">

        <div class="warning-title">
            ⓘ &nbsp; Demo analysis — not for clinical use
        </div>

        <div class="warning-text">
            This application demonstrates a research workflow using
            sample findings. It is not a medical diagnostic device
            and does not replace qualified radiologist judgment.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TWO COLUMN REVIEW WORKSPACE
# ============================================================

left_col, right_col = st.columns(
    [1.05, 1.0],
    gap="large"
)


# ============================================================
# LEFT — CANDIDATE QUEUE
# ============================================================

with left_col:

    st.markdown(
        f"""
        <div class="section-card">

            <div class="section-kicker">
                ● CANDIDATE QUEUE
            </div>

            <div class="section-title">
                Findings
                <span style="
                    color:#8a9199;
                    font-size:13px;
                    font-weight:400;
                ">
                    &nbsp; {candidate_count}/{candidate_count}
                </span>
            </div>

        """,
        unsafe_allow_html=True
    )


    if not detections:

        st.info(
            "No candidate detections were produced."
        )

    else:

        for index, detection in enumerate(
            detections
        ):

            is_selected = (
                index
                == st.session_state.selected_candidate
            )

            center = detection["center"]

            x, y, z = center

            confidence = (
                detection["confidence"]
            )

            classification = (
                detection["classification"]
            )

            if (
                classification
                == "True microbleed"
            ):

                candidate_type = (
                    "True microbleed"
                )

            else:

                candidate_type = (
                    "Mimic"
                )


            status = (
                "REVIEWED"
                if detection["id"]
                in st.session_state.reviewed
                else "NEEDS REVIEW"
            )


            status_class = (
                "pill-reviewed"
                if status == "REVIEWED"
                else "pill-review"
            )


            card_class = (
                "candidate selected"
                if is_selected
                else "candidate"
            )


            st.markdown(
                f"""
                <div class="{card_class}">

                    <div class="candidate-header">

                        <div>

                            <div class="candidate-id">
                                {detection["id"]}
                            </div>

                            <div class="candidate-name">
                                Candidate {index + 1}
                            </div>

                            <div class="candidate-type">
                                {candidate_type}
                            </div>

                        </div>

                        <div style="
                            text-align:right;
                        ">

                            <span class="
                                pill {status_class}
                            ">
                                {status}
                            </span>

                            <div class="
                                confidence-number
                            ">
                                {confidence * 100:.0f}% conf.
                            </div>

                        </div>

                    </div>

                    <div class="
                        confidence-bar
                    ">

                        <div
                            class="
                                confidence-fill
                            "
                            style="
                                width:
                                {confidence * 100:.0f}%;
                            "
                        ></div>

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            if st.button(
                f"Open {detection['id']}",
                key=f"candidate_{index}",
                use_container_width=True
            ):

                st.session_state.selected_candidate = (
                    index
                )

                st.rerun()


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# RIGHT — EVIDENCE DETAIL
# ============================================================

with right_col:

    if detections:

        selected_index = (
            min(
                st.session_state.selected_candidate,
                len(detections) - 1
            )
        )

        selected = detections[
            selected_index
        ]

        x, y, z = selected["center"]

        confidence = (
            selected["confidence"]
        )

        classification = (
            selected["classification"]
        )


        # ----------------------------------------------------
        # VIEWER CONTROLS
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="section-card">

                <div class="section-kicker">
                    ● VIEWER CONTROLS
                </div>

                <div style="
                    color:#283145;
                    font-size:16px;
                    font-weight:700;
                    margin-bottom:15px;
                ">
                    Overlays
                </div>

            """,
            unsafe_allow_html=True
        )


        overlay_col1, overlay_col2 = st.columns(
            [4, 1]
        )


        with overlay_col1:

            st.markdown(
                """
                <div style="
                    font-size:12px;
                    color:#343b48;
                    margin-bottom:5px;
                ">
                    Attention heatmap
                </div>

                <div style="
                    font-size:9px;
                    color:#8b9199;
                    font-family:'Space Mono';
                ">
                    Grad-CAM-style
                </div>
                """,
                unsafe_allow_html=True
            )


        with overlay_col2:

            heatmap_on = st.toggle(
                "Heatmap",
                value=True,
                key="heatmap_toggle",
                label_visibility="collapsed"
            )


        marker_col1, marker_col2 = st.columns(
            [4, 1]
        )


        with marker_col1:

            st.markdown(
                """
                <div style="
                    font-size:12px;
                    color:#343b48;
                    margin-bottom:5px;
                    margin-top:15px;
                ">
                    Candidate markers
                </div>

                <div style="
                    font-size:9px;
                    color:#8b9199;
                    font-family:'Space Mono';
                ">
                    Detected candidates
                </div>
                """,
                unsafe_allow_html=True
            )


        with marker_col2:

            marker_on = st.toggle(
                "Markers",
                value=True,
                key="marker_toggle",
                label_visibility="collapsed"
            )


        cross_col1, cross_col2 = st.columns(
            [4, 1]
        )


        with cross_col1:

            st.markdown(
                """
                <div style="
                    font-size:12px;
                    color:#343b48;
                    margin-top:15px;
                ">
                    Crosshair guide
                </div>

                <div style="
                    font-size:9px;
                    color:#8b9199;
                    font-family:'Space Mono';
                ">
                    Coordinate aid
                </div>
                """,
                unsafe_allow_html=True
            )


        with cross_col2:

            crosshair_on = st.toggle(
                "Crosshair",
                value=False,
                key="crosshair_toggle",
                label_visibility="collapsed"
            )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # EVIDENCE DETAIL
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="section-card">

                <div class="section-kicker">
                    ● EVIDENCE DETAIL
                </div>

            """,
            unsafe_allow_html=True
        )


        st.markdown(
            f"""
            <div style="
                color:#202a3c;
                font-size:23px;
                font-weight:700;
            ">
                Candidate {selected_index + 1}
            </div>

            <div style="
                color:#7c838d;
                font-family:'Space Mono';
                font-size:10px;
                margin-top:6px;
            ">
                X · {x} &nbsp;/&nbsp;
                Y · {y} &nbsp;/&nbsp;
                Z · {z}
            </div>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        volume = (
            st.session_state.volume
        )


        if volume is not None:

            slice_index = int(
                np.clip(
                    z,
                    0,
                    volume.shape[2] - 1
                )
            )


            image_slice = (
                volume[:, :, slice_index]
            )


            fig, ax = plt.subplots(
                figsize=(6.5, 4.5)
            )

            ax.imshow(
                image_slice.T,
                cmap="gray",
                origin="lower"
            )


            # ------------------------------------------------
            # CANDIDATE MARKERS
            # ------------------------------------------------

            if marker_on:

                for d in detections:

                    dx, dy, dz = (
                        d["center"]
                    )

                    if dz == slice_index:

                        if (
                            d["id"]
                            == selected["id"]
                        ):

                            radius = 7

                            linewidth = 2.5

                        else:

                            radius = 5

                            linewidth = 1.5


                        circle = Circle(
                            (
                                dx,
                                dy
                            ),
                            radius=radius,
                            fill=False,
                            linewidth=linewidth,
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
                f"Slice {slice_index}",
                fontsize=10
            )

            ax.axis(
                "off"
            )


            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)


        # ----------------------------------------------------
        # STATS
        # ----------------------------------------------------

        stat1, stat2 = st.columns(2)


        with stat1:

            st.markdown(
                f"""
                <div class="evidence-stat">

                    <div class="evidence-label">
                        CONFIDENCE
                    </div>

                    <div class="evidence-value">
                        {confidence * 100:.0f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        with stat2:

            st.markdown(
                f"""
                <div class="evidence-stat">

                    <div class="evidence-label">
                        SLICE
                    </div>

                    <div class="evidence-value">
                        {slice_index}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        st.markdown(
            """
            <div style="
                color:#737a83;
                font-family:'Space Mono';
                font-size:9px;
                letter-spacing:1.5px;
                margin:18px 0 8px 0;
            ">
                CLASSIFICATION
            </div>
            """,
            unsafe_allow_html=True
        )


        class1, class2 = st.columns(2)


        with class1:

            if classification == "True microbleed":

                st.markdown(
                    """
                    <div class="
                        class-box active
                    ">
                        True microbleed
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class="
                        class-box
                    ">
                        True microbleed
                    </div>
                    """,
                    unsafe_allow_html=True
                )


        with class2:

            if classification == "Mimic":

                st.markdown(
                    """
                    <div class="
                        class-box active
                    ">
                        Mimic
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class="
                        class-box
                    ">
                        Mimic
                    </div>
                    """,
                    unsafe_allow_html=True
                )


        # ----------------------------------------------------
        # SEVERITY
        # ----------------------------------------------------

        st.markdown(
            """
            <div style="
                color:#737a83;
                font-family:'Space Mono';
                font-size:9px;
                letter-spacing:1.5px;
                margin:18px 0 8px 0;
            ">
                REVIEW SEVERITY
            </div>
            """,
            unsafe_allow_html=True
        )


        severity_values = [
            "Low",
            "Moderate",
            "High"
        ]

        severity_mapping = {
            "None": "Low",
            "Mild": "Low",
            "Moderate": "Moderate",
            "Severe": "High"
        }

        current_severity = (
            severity_mapping.get(
                severity,
                "Low"
            )
        )


        s1, s2, s3 = st.columns(3)


        for column, value in zip(
            [s1, s2, s3],
            severity_values
        ):

            with column:

                if value == current_severity:

                    st.markdown(
                        f"""
                        <div class="
                            severity-box active
                        ">
                            {value}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        f"""
                        <div class="
                            severity-box
                        ">
                            {value}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


        # ----------------------------------------------------
        # GRAD-CAM
        # ----------------------------------------------------

        if heatmap_on:

            st.markdown(
                """
                <div class="attention-note">

                    <div class="attention-title">
                        ✧ &nbsp; Explainability view
                    </div>

                    <div class="attention-text">
                        Attention heatmap indicates regions that
                        contributed to the selected model response.
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            try:

                selected_patch = (
                    pipeline._cut_patch(
                        volume,
                        selected["center"],
                        (16, 16, 8)
                    )
                )


                heatmap = generate_gradcam(
                    pipeline.stage2_model,
                    selected_patch
                )


                heat_slice = (
                    heatmap[:, :, heatmap.shape[2] // 2]
                )


                patch_slice = (
                    selected_patch[
                        :,
                        :,
                        selected_patch.shape[2] // 2
                    ]
                )


                overlay = create_attention_overlay(
                    patch_slice,
                    heat_slice
                )


                st.image(
                    overlay,
                    caption="Stage-2 model attention",
                    use_container_width=True
                )

            except Exception as error:

                st.caption(
                    "Grad-CAM could not be generated for this candidate."
                )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # REVIEW BUTTON
        # ----------------------------------------------------

        review_button_col1, review_button_col2 = (
            st.columns(2)
        )


        with review_button_col1:

            if st.button(
                "✓ Mark as reviewed",
                use_container_width=True
            ):

                st.session_state.reviewed.add(
                    selected["id"]
                )

                st.rerun()


        with review_button_col2:

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
# PDF EXPORT
# ============================================================

if export_clicked:

    pdf_bytes = generate_pdf_report(
        filename=st.session_state.scan_name,
        detections=detections,
        severity=severity,
        mean_confidence=mean_confidence,
        review_count=review_count
    )


    st.download_button(
        label="Download PDF report",
        data=pdf_bytes,
        file_name=(
            Path(
                st.session_state.scan_name
            ).stem
            + "_CMB_Report.pdf"
        ),
        mime="application/pdf",
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-note">
        CMB Review · Research interface only ·
        Attention maps do not establish causality or clinical significance.
    </div>
    """,
    unsafe_allow_html=True
)