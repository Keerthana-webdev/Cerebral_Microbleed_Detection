import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import base64
import io

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CMB Review",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #f7f8f6;
}

/* Hide Streamlit default UI */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #171b39 0%,
        #1c2143 100%
    );
    min-width: 300px;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.2rem;
}

/* Sidebar text */
.sidebar-title {
    color: white;
    font-size: 22px;
    font-weight: 700;
    margin-left: 8px;
}

.sidebar-subtitle {
    color: #7e86a5;
    font-size: 10px;
    letter-spacing: 2px;
    margin-left: 52px;
    margin-top: -4px;
}

.logo-circle {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background: #20bdd0;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 23px;
    margin-right: 8px;
}

/* Demo analysis card */

.demo-card {
    margin-top: 34px;
    padding: 18px;
    border-radius: 20px;
    background: rgba(75, 86, 140, 0.35);
    border: 1px solid rgba(255,255,255,0.08);
}

.demo-label {
    color: #ff8063;
    font-family: 'Space Mono';
    font-size: 10px;
    letter-spacing: 2px;
}

.demo-name {
    color: white;
    font-weight: 700;
    font-size: 16px;
    margin-top: 12px;
}

.demo-description {
    color: #9299b8;
    font-size: 12px;
    margin-top: 4px;
}

/* Sidebar workspace */

.workspace-label {
    color: #6f7695;
    font-family: 'Space Mono';
    font-size: 10px;
    letter-spacing: 2px;
    margin-top: 35px;
    margin-bottom: 12px;
}

.nav-active {
    background: rgba(25, 185, 211, 0.22);
    color: #27c3d6;
    padding: 13px 15px;
    border-radius: 14px;
    font-size: 14px;
    margin-bottom: 8px;
}

.nav-item {
    color: #c0c5d8;
    padding: 13px 15px;
    border-radius: 14px;
    font-size: 14px;
    margin-bottom: 4px;
}

/* Sidebar footer */

.sidebar-bottom {
    position: fixed;
    bottom: 20px;
    width: 250px;
}

.safety-title {
    color: #777e9e;
    font-family: 'Space Mono';
    font-size: 10px;
    letter-spacing: 2px;
}

.safety-text {
    color: #8d94af;
    font-size: 11px;
    line-height: 1.5;
    margin-top: 8px;
}

.user-card {
    margin-top: 18px;
    padding: 12px;
    border-radius: 18px;
    background: rgba(70, 78, 128, 0.45);
    display: flex;
    align-items: center;
}

.avatar {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background: #e97b5f;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    margin-right: 10px;
}

.user-name {
    color: white;
    font-size: 13px;
    font-weight: 600;
}

.user-role {
    color: #9299b8;
    font-size: 10px;
}

/* =========================================================
   TOP HEADER
   ========================================================= */

.top-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 4px 0 20px 0;
    border-bottom: 1px solid #e5e6e4;
}

.breadcrumb {
    color: #4f5361;
    font-family: 'Space Mono';
    font-size: 11px;
    letter-spacing: 2px;
}

.demo-status {
    color: #6f747c;
    font-size: 12px;
}

.green-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    background: #42a77a;
    border-radius: 50%;
    margin-right: 6px;
}

/* =========================================================
   HERO
   ========================================================= */

.hero-label {
    color: #e9755d;
    font-family: 'Space Mono';
    font-size: 11px;
    letter-spacing: 2px;
    margin-top: 42px;
}

.hero-title {
    font-size: 42px;
    line-height: 1.05;
    color: #20243b;
    font-weight: 700;
    margin-top: 14px;
}

.hero-title span {
    color: #168fa2;
}

.hero-description {
    color: #777b85;
    font-size: 14px;
    max-width: 680px;
    line-height: 1.6;
    margin-top: 14px;
}

/* =========================================================
   BUTTONS
   ========================================================= */

.stButton > button {
    border-radius: 25px;
    border: 1px solid #dadbdc;
    background: white;
    color: #343743;
    font-weight: 600;
    min-height: 44px;
}

.stButton > button:hover {
    border-color: #21aabd;
    color: #158fa3;
}

.primary-button button {
    background: #159caf !important;
    color: white !important;
    border: none !important;
}

/* =========================================================
   METRICS
   ========================================================= */

.metric-card {
    background: white;
    border: 1px solid #e1e2df;
    border-radius: 20px;
    padding: 22px;
    min-height: 130px;
    box-shadow: 0 3px 15px rgba(35, 40, 60, 0.025);
}

.metric-label {
    color: #747984;
    font-family: 'Space Mono';
    font-size: 10px;
    letter-spacing: 1.5px;
}

.metric-value {
    color: #25283b;
    font-size: 30px;
    font-weight: 700;
    margin-top: 14px;
}

.metric-sub {
    color: #92959c;
    font-size: 11px;
    margin-top: 5px;
}

/* =========================================================
   SAFETY BANNER
   ========================================================= */

.safety-banner {
    background: #fff0ea;
    border: 1px solid #f1c7b9;
    border-radius: 18px;
    padding: 17px 20px;
    margin: 22px 0;
}

.safety-title-main {
    color: #9b594a;
    font-weight: 700;
    font-size: 14px;
}

.safety-description {
    color: #8c756e;
    font-size: 12px;
    margin-top: 6px;
}

/* =========================================================
   SECTION CARDS
   ========================================================= */

.section-card {
    background: white;
    border: 1px solid #e0e2df;
    border-radius: 22px;
    padding: 22px;
    margin-top: 20px;
}

.section-label {
    color: #e66e57;
    font-family: 'Space Mono';
    font-size: 10px;
    letter-spacing: 2px;
}

.section-title {
    color: #292d3f;
    font-size: 22px;
    font-weight: 700;
    margin-top: 10px;
}

/* =========================================================
   CANDIDATE ROW
   ========================================================= */

.candidate-row {
    border-top: 1px solid #ececea;
    padding: 18px 4px;
}

.candidate-id {
    color: #e36d55;
    font-family: 'Space Mono';
    font-size: 10px;
    letter-spacing: 1px;
}

.candidate-name {
    color: #292c3d;
    font-size: 16px;
    font-weight: 600;
    margin-top: 5px;
}

.candidate-type {
    color: #70747d;
    font-size: 12px;
    margin-top: 6px;
}

.confidence {
    text-align: right;
    font-weight: 600;
    color: #3b3f4d;
}

.confidence-bar {
    width: 100%;
    height: 5px;
    background: #e9eaed;
    border-radius: 10px;
    margin-top: 12px;
    overflow: hidden;
}

.confidence-fill {
    height: 100%;
    border-radius: 10px;
    background: #e77759;
}

.confidence-fill.mimic {
    background: #4b5a9a;
}

/* Status pills */

.status-review {
    background: #fff0e8;
    color: #b36350;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 10px;
    font-family: 'Space Mono';
}

.status-reviewed {
    background: #e8f4ef;
    color: #51866f;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 10px;
    font-family: 'Space Mono';
}

/* =========================================================
   VIEWER
   ========================================================= */

.viewer-card {
    background: white;
    border: 1px solid #e0e2df;
    border-radius: 22px;
    padding: 24px;
    margin-top: 20px;
}

.control-label {
    color: #777c86;
    font-family: 'Space Mono';
    font-size: 10px;
    letter-spacing: 1px;
}

.control-description {
    color: #9a9da4;
    font-size: 10px;
    margin-top: 4px;
}

/* =========================================================
   EVIDENCE
   ========================================================= */

.evidence-card {
    background: white;
    border: 1px solid #e0e2df;
    border-radius: 22px;
    padding: 24px;
    margin-top: 20px;
}

.evidence-title {
    color: #25293a;
    font-size: 24px;
    font-weight: 700;
}

.coordinates {
    color: #8a8e98;
    font-family: 'Space Mono';
    font-size: 11px;
    margin-top: 5px;
}

.detail-box {
    background: #f1f3f6;
    border-radius: 17px;
    padding: 18px;
    margin-top: 20px;
}

.detail-label {
    color: #858995;
    font-family: 'Space Mono';
    font-size: 9px;
    letter-spacing: 1px;
}

.detail-value {
    color: #25283a;
    font-size: 26px;
    font-weight: 700;
    margin-top: 8px;
}

.explanation {
    border-top: 1px solid #e2e3e2;
    margin-top: 20px;
    padding-top: 18px;
}

.explanation-title {
    color: #d96c55;
    font-size: 14px;
    font-weight: 600;
}

.explanation-text {
    color: #777b84;
    font-size: 12px;
    margin-top: 8px;
}

.info-note {
    background: #f2f7f8;
    border: 1px solid #dcebed;
    color: #6f7d82;
    border-radius: 15px;
    padding: 13px 16px;
    font-size: 11px;
    margin-top: 15px;
}

/* =========================================================
   UPLOAD
   ========================================================= */

.upload-box {
    background: white;
    border: 2px dashed #d5d8d8;
    border-radius: 20px;
    padding: 20px;
    margin-top: 18px;
}

/* =========================================================
   STREAMLIT WIDGET TWEAKS
   ========================================================= */

div[data-testid="stFileUploader"] {
    background: transparent;
}

div[data-testid="stExpander"] {
    border-radius: 16px;
}

.stSlider {
    padding-top: 0;
}

</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE
# ============================================================

if "analysis_run" not in st.session_state:
    st.session_state.analysis_run = False

if "selected_candidate" not in st.session_state:
    st.session_state.selected_candidate = 0

if "reviewed" not in st.session_state:
    st.session_state.reviewed = set()

if "attention_map" not in st.session_state:
    st.session_state.attention_map = True

if "candidate_markers" not in st.session_state:
    st.session_state.candidate_markers = True

if "crosshair" not in st.session_state:
    st.session_state.crosshair = False

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="display:flex;align-items:center;">
            <div class="logo-circle">🧠</div>
            <div>
                <div class="sidebar-title">CMB Review</div>
                <div class="sidebar-subtitle">NEUROIMAGING LAB</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="demo-card">
            <div class="demo-label">● DEMO ANALYSIS</div>
            <div class="demo-name">VALDO_014</div>
            <div class="demo-description">
                Synthetic review session · SWI
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="workspace-label">WORKSPACE</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="nav-active">◫ &nbsp; Review workspace &nbsp; •</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="nav-item">♜ &nbsp; Methodology</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="nav-item">▤ &nbsp; Literature map</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br><br><br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="sidebar-bottom">
            <div class="safety-title">♙ SAFETY FIRST</div>
            <div class="safety-text">
                Research interface only. Outputs require
                qualified clinical review.
            </div>

            <div class="user-card">
                <div class="avatar">HV</div>
                <div>
                    <div class="user-name">Capstone Team</div>
                    <div class="user-role">Research workspace</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="top-header">
        <div class="breadcrumb">CMB &nbsp;/&nbsp; REVIEW</div>

        <div class="demo-status">
            <span class="green-dot"></span>
            Local demo mode
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HERO
# ============================================================

col1, col2 = st.columns([3.7, 1.2])

with col1:

    st.markdown(
        '<div class="hero-label">● REVIEW WORKSPACE / DEMO</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero-title">
            Read the evidence,<br>
            <span>not just the output.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero-description">
            A clinician-facing review surface for cerebral microbleed
            candidates, mimics, uncertainty and model attention.
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:

    st.markdown("<br><br>", unsafe_allow_html=True)

    if st.button("↻  Re-run analysis", use_container_width=True):
        st.session_state.analysis_run = True
        st.rerun()

    if st.button("⇩  Export report", use_container_width=True):
        st.success("Report export will be connected to the PDF generator.")

# ============================================================
# UPLOAD
# ============================================================

with st.expander("📁  Upload a new SWI / T2S scan", expanded=False):

    uploaded_file = st.file_uploader(
        "Upload NIfTI scan",
        type=["nii", "gz"],
        help="Upload a .nii or .nii.gz MRI scan.",
    )

    if uploaded_file is not None:

        st.success(
            f"Loaded: {uploaded_file.name}"
        )

        if st.button(
            "▶ Run CMB Analysis",
            use_container_width=True,
        ):
            st.session_state.analysis_run = True
            st.session_state.uploaded_name = uploaded_file.name
            st.rerun()

# ============================================================
# DEMO DATA
# ============================================================

candidates = [
    {
        "id": "CMB-01",
        "location": "Right temporal",
        "classification": "True microbleed",
        "confidence": 0.94,
        "slice": 82,
        "coords": (18, 42, 82),
        "status": "Needs review",
        "severity": "Moderate",
        "explanation": "Compact hypointense focus with a high attention response.",
    },
    {
        "id": "CMB-02",
        "location": "Left parietal",
        "classification": "True microbleed",
        "confidence": 0.87,
        "slice": 76,
        "coords": (42, 61, 76),
        "status": "Reviewed",
        "severity": "Moderate",
        "explanation": "Compact candidate consistent with a microbleed-like appearance.",
    },
    {
        "id": "CMB-03",
        "location": "Right occipital",
        "classification": "Mimic",
        "confidence": 0.79,
        "slice": 64,
        "coords": (71, 43, 64),
        "status": "Needs review",
        "severity": "Low",
        "explanation": "Candidate pattern is more consistent with a potential mimic.",
    },
    {
        "id": "CMB-04",
        "location": "Left frontal",
        "classification": "True microbleed",
        "confidence": 0.72,
        "slice": 58,
        "coords": (39, 37, 58),
        "status": "Needs review",
        "severity": "Moderate",
        "explanation": "Candidate detected by the two-stage classifier.",
    },
    {
        "id": "CMB-05",
        "location": "Right frontal",
        "classification": "Mimic",
        "confidence": 0.67,
        "slice": 49,
        "coords": (82, 29, 49),
        "status": "Reviewed",
        "severity": "Low",
        "explanation": "Candidate demonstrates mimic-like characteristics.",
    },
]

# ============================================================
# METRICS
# ============================================================

true_count = sum(
    1 for c in candidates
    if c["classification"] == "True microbleed"
)

mimic_count = sum(
    1 for c in candidates
    if c["classification"] == "Mimic"
)

reviewed_count = sum(
    1 for c in candidates
    if c["status"] == "Reviewed"
)

mean_confidence = (
    np.mean([c["confidence"] for c in candidates]) * 100
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">CANDIDATES</div>
            <div class="metric-value">{len(candidates):02d}</div>
            <div class="metric-sub">
                {true_count} microbleed · {mimic_count} mimic
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">REVIEWED</div>
            <div class="metric-value">{reviewed_count:02d}</div>
            <div class="metric-sub">
                {len(candidates) - reviewed_count} still need attention
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">MEAN CONFIDENCE</div>
            <div class="metric-value">{mean_confidence:.1f}%</div>
            <div class="metric-sub">
                Candidate-level estimate
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">LAST RUN</div>
            <div class="metric-value" style="font-size:18px;">
                Today
            </div>
            <div class="metric-sub">
                {datetime.now().strftime("%I:%M %p")} · Demo inference
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# SAFETY BANNER
# ============================================================

st.markdown(
    """
    <div class="safety-banner">
        <div class="safety-title-main">
            ⓘ &nbsp; Demo analysis — not for clinical use
        </div>
        <div class="safety-description">
            This interface demonstrates a research workflow using sample
            findings. It is not a medical diagnostic device and does not
            replace qualified radiologist judgment.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# CANDIDATE QUEUE
# ============================================================

st.markdown(
    """
    <div class="section-card">
        <div class="section-label">● CANDIDATE QUEUE</div>
        <div class="section-title">
            Findings
            <span style="font-size:12px;color:#999;font-weight:400;">
                5/5
            </span>
        </div>
    """,
    unsafe_allow_html=True,
)

for idx, candidate in enumerate(candidates):

    col1, col2, col3 = st.columns([4.5, 1.5, 1.2])

    with col1:

        classification_class = (
            "mimic"
            if candidate["classification"] == "Mimic"
            else ""
        )

        st.markdown(
            f"""
            <div class="candidate-row">

                <div class="candidate-id">
                    ● &nbsp; {candidate["id"]}
                </div>

                <div class="candidate-name">
                    {candidate["location"]}
                </div>

                <div class="candidate-type">
                    {candidate["classification"]}
                </div>

                <div class="confidence-bar">
                    <div
                        class="confidence-fill {classification_class}"
                        style="width:{candidate["confidence"]*100}%;">
                    </div>
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown("<br>", unsafe_allow_html=True)

        if candidate["status"] == "Reviewed":
            st.markdown(
                '<span class="status-reviewed">✓ REVIEWED</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-review">● NEEDS REVIEW</span>',
                unsafe_allow_html=True,
            )

    with col3:

        st.markdown(
            f"""
            <div class="confidence">
                {candidate["confidence"]*100:.0f}% conf.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Open",
            key=f"open_{idx}",
        ):
            st.session_state.selected_candidate = idx
            st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# SELECTED CANDIDATE
# ============================================================

selected = candidates[
    st.session_state.selected_candidate
]

# ============================================================
# VIEWER CONTROLS
# ============================================================

st.markdown(
    """
    <div class="viewer-card">

        <div class="section-label">
            ● VIEWER CONTROLS
        </div>

        <div class="section-title">
            Overlays
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:

    attention = st.toggle(
        "Attention heatmap",
        value=st.session_state.attention_map,
    )

    st.session_state.attention_map = attention

    st.caption("Grad-CAM-style")

with col2:

    markers = st.toggle(
        "Candidate markers",
        value=st.session_state.candidate_markers,
    )

    st.session_state.candidate_markers = markers

    st.caption(f"{len(candidates)} detected")

with col3:

    crosshair = st.toggle(
        "Crosshair guide",
        value=st.session_state.crosshair,
    )

    st.session_state.crosshair = crosshair

    st.caption("Coordinate aid")

# ============================================================
# EVIDENCE DETAIL
# ============================================================

st.markdown(
    f"""
    <div class="evidence-card">

        <div class="section-label">
            ● EVIDENCE DETAIL
        </div>

        <div class="evidence-title">
            {selected["location"]}
        </div>

        <div class="coordinates">
            R · {selected["coords"][0]} /
            {selected["coords"][1]} /
            {selected["coords"][2]}
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DETAILS
# ============================================================

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        f"""
        <div class="detail-box">
            <div class="detail-label">CONFIDENCE</div>
            <div class="detail-value">
                {selected["confidence"]*100:.0f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:

    st.markdown(
        f"""
        <div class="detail-box">
            <div class="detail-label">SLICE</div>
            <div class="detail-value">
                {selected["slice"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# CLASSIFICATION
# ============================================================

st.markdown(
    """
    <div style="
        color:#777c86;
        font-family:'Space Mono';
        font-size:10px;
        letter-spacing:1.5px;
        margin-top:22px;
    ">
        CLASSIFICATION
    </div>
    """,
    unsafe_allow_html=True,
)

class_col1, class_col2 = st.columns(2)

with class_col1:

    if selected["classification"] == "True microbleed":
        st.success("✓  True microbleed")
    else:
        st.button(
            "True microbleed",
            key="true_classification",
            use_container_width=True,
        )

with class_col2:

    if selected["classification"] == "Mimic":
        st.info("●  Mimic")
    else:
        st.button(
            "Mimic",
            key="mimic_classification",
            use_container_width=True,
        )

# ============================================================
# SEVERITY
# ============================================================

st.markdown(
    """
    <div style="
        color:#777c86;
        font-family:'Space Mono';
        font-size:10px;
        letter-spacing:1.5px;
        margin-top:22px;
    ">
        REVIEW SEVERITY
    </div>
    """,
    unsafe_allow_html=True,
)

sev1, sev2, sev3 = st.columns(3)

with sev1:
    st.button(
        "Low",
        use_container_width=True,
        key="severity_low",
    )

with sev2:
    if selected["severity"] == "Moderate":
        st.warning("Moderate")
    else:
        st.button(
            "Moderate",
            use_container_width=True,
            key="severity_moderate",
        )

with sev3:
    st.button(
        "High",
        use_container_width=True,
        key="severity_high",
    )

# ============================================================
# EXPLANATION
# ============================================================

st.markdown(
    f"""
    <div class="explanation">

        <div class="explanation-title">
            ✨ &nbsp; Explainability note
        </div>

        <div class="explanation-text">
            {selected["explanation"]}
        </div>

    </div>

    <div class="info-note">
        ⓘ &nbsp; Attention maps show where the model focused;
        they do not establish causality or clinical significance.
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# MRI VIEWER PLACEHOLDER
# ============================================================

st.markdown(
    """
    <div class="section-card">

        <div class="section-label">
            ● MRI EVIDENCE VIEW
        </div>

        <div class="section-title">
            Candidate slice
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)

# Create a simple demo MRI-like image
size = 256

x = np.linspace(-1, 1, size)
y = np.linspace(-1, 1, size)
xx, yy = np.meshgrid(x, y)

brain = np.exp(
    -((xx / 0.72) ** 2 + (yy / 0.85) ** 2) * 3
)

noise = np.random.default_rng(42).normal(
    0,
    0.04,
    (size, size),
)

mri = brain + noise
mri = np.clip(mri, 0, 1)

# Candidate marker
cx = int(
    (selected["coords"][0] / 100) * size
)

cy = int(
    (selected["coords"][1] / 100) * size
)

if st.session_state.candidate_markers:

    yy2, xx2 = np.ogrid[
        :size,
        :size
    ]

    radius = 6

    marker = (
        (xx2 - cx) ** 2 +
        (yy2 - cy) ** 2
        <= radius ** 2
    )

    mri[marker] = 1.0

# Crosshair
if st.session_state.crosshair:

    if 0 <= cx < size:
        mri[:, cx] = 1.0

    if 0 <= cy < size:
        mri[cy, :] = 1.0

st.image(
    mri,
    caption=(
        f"Slice {selected['slice']} · "
        f"{selected['location']}"
    ),
    use_container_width=True,
)

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#9a9da4;
        font-size:10px;
        padding:30px 0;
    ">
        CMB Review · Research prototype ·
        Outputs require qualified clinical review
    </div>
    """,
    unsafe_allow_html=True,
)