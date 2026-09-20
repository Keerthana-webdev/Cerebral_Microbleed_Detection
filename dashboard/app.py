"""
CMB Review — clinician-style review dashboard, wired to the real detection pipeline.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src" / "pipeline"))
sys.path.append(str(PROJECT_ROOT / "src" / "severity_gradcam"))
from final_pipeline import CMBDetectionPipeline
from gradcam_3d import GradCAM3D

# =========================================================
# PAGE CONFIG + CSS (this block gives the whole dark/light clinical look)
# =========================================================
st.set_page_config(page_title="CMB Review", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #f7f7f5; }
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
    }
    section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; }

    .sidebar-logo {
        display:flex; align-items:center; gap:10px; padding: 4px 0 18px 0;
    }
    .sidebar-logo .brain { font-size: 26px; }
    .sidebar-logo .title { font-weight: 700; font-size: 17px; color:#f1f5f9; }
    .sidebar-logo .subtitle { font-size: 10px; letter-spacing: 1.5px; color:#64748b; }

    .demo-card {
        background: #1e293b; border-radius: 10px; padding: 14px 16px; margin-bottom: 18px;
        border-left: 3px solid #f97316;
    }
    .demo-card .tag { font-size: 10px; letter-spacing: 1px; color: #fb923c; font-weight:600; }
    .demo-card .name { font-size: 16px; font-weight: 700; color: #f1f5f9; margin-top: 4px;}
    .demo-card .desc { font-size: 11px; color: #94a3b8; margin-top:2px; }

    .safety-box {
        background: #1e293b; border-radius: 10px; padding: 12px 14px; font-size: 11px;
        color: #94a3b8; margin-top: 24px; line-height:1.4;
    }
    .safety-box .head { color:#e2e8f0; font-weight:600; font-size:11px; margin-bottom:4px;}

    .stat-card {
        background: white; border-radius: 12px; padding: 16px 18px;
        border: 1px solid #e5e7eb; height: 100%;
    }
    .stat-label { font-size: 10px; letter-spacing: 1px; color: #94a3b8; font-weight:600; text-transform:uppercase;}
    .stat-value { font-size: 28px; font-weight: 800; color: #0f172a; margin-top: 4px;}
    .stat-sub { font-size: 11px; color: #94a3b8; margin-top: 2px;}

    .safety-banner {
        background: #fef2f2; border: 1px solid #fecaca; border-radius: 10px;
        padding: 14px 18px; margin: 18px 0; font-size: 13px; color: #7f1d1d;
    }
    .safety-banner b { color:#991b1b; }

    .badge-review {
        background:#fff7ed; color:#c2410c; border:1px solid #fed7aa; border-radius: 20px;
        padding: 3px 10px; font-size: 11px; font-weight:600;
    }
    .badge-reviewed {
        background:#f0fdf4; color:#15803d; border:1px solid #bbf7d0; border-radius: 20px;
        padding: 3px 10px; font-size: 11px; font-weight:600;
    }

    .candidate-card {
        background: white; border: 1px solid #e5e7eb; border-radius: 12px;
        padding: 14px 18px; margin-bottom: 10px;
    }
    .candidate-card.selected { border: 1.5px solid #0d9488; background:#f0fdfa; }
    .candidate-id { font-size: 10px; color: #94a3b8; letter-spacing: 0.5px; }
    .candidate-region { font-size: 16px; font-weight: 700; color:#0f172a; margin: 2px 0 8px 0;}
    .candidate-classlabel { font-size: 12px; color:#475569; }

    .conf-bar-bg { background:#e5e7eb; border-radius:6px; height:6px; width:100%; margin-top:6px;}
    .conf-bar-fill-cmb { background: linear-gradient(90deg,#fb923c,#ea580c); height:6px; border-radius:6px; }
    .conf-bar-fill-mimic { background: linear-gradient(90deg,#818cf8,#4f46e5); height:6px; border-radius:6px; }

    .section-label {
        font-size: 11px; letter-spacing: 1.5px; color: #0d9488; font-weight: 700;
        text-transform: uppercase; margin-bottom: 6px;
    }

    div.stButton > button {
        border-radius: 8px; border: 1px solid #d1d5db; background: white; color:#0f172a;
    }
    div.stButton > button:hover { border-color:#0d9488; color:#0d9488; }
</style>
""", unsafe_allow_html=True)


# =========================================================
# PIPELINE + STATE
# =========================================================
@st.cache_resource
def load_pipeline():
    return CMBDetectionPipeline()


pipeline = load_pipeline()

if "candidates" not in st.session_state:
    st.session_state.candidates = None
if "reviewed" not in st.session_state:
    st.session_state.reviewed = {}
if "classification_override" not in st.session_state:
    st.session_state.classification_override = {}
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = 0
if "volume" not in st.session_state:
    st.session_state.volume = None
if "subject_name" not in st.session_state:
    st.session_state.subject_name = None


def region_label(center, shape):
    """Rough anatomical-style label from voxel coordinates — heuristic only, for
    demo/UI purposes, not a clinically validated atlas mapping."""
    x, y, z = center
    side = "L" if x < shape[0] / 2 else "R"
    side_full = "Left" if side == "L" else "Right"
    if y > shape[1] * 0.62:
        lobe = "frontal"
    elif y < shape[1] * 0.35:
        lobe = "occipital"
    elif x < shape[0] * 0.3 or x > shape[0] * 0.7:
        lobe = "temporal"
    else:
        lobe = "parietal"
    return f"{side_full} {lobe}", f"{side} · {x}/{y}/{z}"


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span class="brain">🧠</span>
        <div>
            <div class="title">CMB Review</div>
            <div class="subtitle">NEUROIMAGING LAB</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.subject_name:
        st.markdown(f"""
        <div class="demo-card">
            <div class="tag">● ANALYSIS</div>
            <div class="name">{st.session_state.subject_name}</div>
            <div class="desc">SWI scan review session</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("**WORKSPACE**")
    page = st.radio("nav", ["Review workspace", "Methodology", "Literature map"],
                     label_visibility="collapsed")

    st.markdown("""
    <div class="safety-box">
        <div class="head">🛡️ SAFETY FIRST</div>
        Research interface only. Outputs require qualified clinical review.
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# METHODOLOGY / LITERATURE PAGES (simple static content)
# =========================================================
if page == "Methodology":
    st.title("Methodology")
    st.markdown("""
    **Pipeline:** N4 bias correction → resampling → z-score normalization → sliding-window
    Stage 1 3D CNN (candidate detection) → Non-Maximum Suppression → Stage 2 3D CNN
    (mimic-aware classification) → confidence scoring → rule-based severity grading →
    Grad-CAM explainability.

    **Training data:** VALDO Task 2 (72 subjects), subject-wise train/val/test split.

    **Key results:** 93.2% reduction in false positives per subject (patch-level),
    100% sensitivity on held-out test patches, 29.3% lesion-level sensitivity at
    ~107.6 FP/scan on whole-volume sliding-window evaluation.
    """)
    st.stop()

if page == "Literature map":
    st.title("Literature Map")
    st.markdown("30 papers reviewed (2015–2026) covering CMB detection architectures, "
                "mimic rejection, and confidence-aware clinical deployment. "
                "See your Literature Survey document for the full table.")
    st.stop()

# =========================================================
# MAIN — HEADER
# =========================================================
top_l, top_r = st.columns([5, 1])
with top_l:
    st.caption("CMB / REVIEW")
with top_r:
    st.caption("🟢 Local demo mode")

st.markdown("### Review workspace")
st.markdown("# Read the evidence,  \n:teal[not just the output.]")
st.write("A clinician-style review surface for cerebral microbleed candidates, "
         "mimics, uncertainty, and model attention.")

col_upload, col_btn = st.columns([3, 1])
with col_upload:
    uploaded_file = st.file_uploader("Upload SWI scan (.nii.gz)", type=["nii.gz", "gz"],
                                      label_visibility="collapsed")
with col_btn:
    run_clicked = st.button("🔄  Run analysis", use_container_width=True)

if uploaded_file is not None and run_clicked:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Preprocessing + running two-stage detection pipeline..."):
        volume = pipeline.preprocess(tmp_path)
        detections = pipeline.detect(volume)

    candidates = []
    for i, d in enumerate(detections):
        region, coord_str = region_label(d["center"], volume.shape)
        candidates.append({
            "id": f"CMB-{i+1:02d}",
            "region": region,
            "coord_str": coord_str,
            "center": d["center"],
            "confidence": d["confidence"],
            "confidence_label": d["confidence_label"],
        })

    st.session_state.candidates = candidates
    st.session_state.volume = volume
    st.session_state.subject_name = uploaded_file.name.replace(".nii.gz", "")
    st.session_state.reviewed = {}
    st.session_state.classification_override = {}
    st.session_state.selected_idx = 0

# =========================================================
# EMPTY STATE
# =========================================================
if st.session_state.candidates is None:
    st.info("👆 Upload an SWI scan and click **Run analysis** to begin a review session.")
    st.stop()

candidates = st.session_state.candidates
volume = st.session_state.volume
n_total = len(candidates)
n_reviewed = len(st.session_state.reviewed)
mean_conf = np.mean([c["confidence"] for c in candidates]) * 100 if candidates else 0

# =========================================================
# STAT CARDS
# =========================================================
s1, s2, s3, s4 = st.columns(4)
n_cmb_default = sum(1 for c in candidates if c["confidence_label"] != "REVIEW RECOMMENDED")
with s1:
    st.markdown(f"""<div class="stat-card"><div class="stat-label">CANDIDATES</div>
    <div class="stat-value">{n_total:02d}</div>
    <div class="stat-sub">{n_cmb_default} high-conf · {n_total - n_cmb_default} review</div></div>""",
    unsafe_allow_html=True)
with s2:
    st.markdown(f"""<div class="stat-card"><div class="stat-label">REVIEWED</div>
    <div class="stat-value">{n_reviewed:02d}</div>
    <div class="stat-sub">{n_total - n_reviewed} still need attention</div></div>""",
    unsafe_allow_html=True)
with s3:
    st.markdown(f"""<div class="stat-card"><div class="stat-label">MEAN CONFIDENCE</div>
    <div class="stat-value">{mean_conf:.1f}%</div>
    <div class="stat-sub">Candidate-level estimate</div></div>""",
    unsafe_allow_html=True)
with s4:
    st.markdown(f"""<div class="stat-card"><div class="stat-label">SUBJECT</div>
    <div class="stat-value" style="font-size:18px;">{st.session_state.subject_name}</div>
    <div class="stat-sub">Pipeline v1 · CPU inference</div></div>""",
    unsafe_allow_html=True)

st.markdown("""
<div class="safety-banner">
⚠️ <b>Demo analysis — not for clinical use.</b><br>
This capstone interface demonstrates a research workflow using real model outputs on
sample scans. It is not a medical diagnostic device and does not replace radiologist judgment.
</div>
""", unsafe_allow_html=True)

# =========================================================
# CANDIDATE QUEUE + EVIDENCE DETAIL (two-column layout)
# =========================================================
left, right = st.columns([1.1, 1.4])

with left:
    st.markdown(f'<div class="section-label">CANDIDATE QUEUE — FINDINGS {n_total}/{n_total}</div>',
                unsafe_allow_html=True)

    for i, c in enumerate(candidates):
        is_selected = (i == st.session_state.selected_idx)
        is_reviewed = c["id"] in st.session_state.reviewed
        override = st.session_state.classification_override.get(c["id"])
        classification = override or ("True microbleed" if c["confidence_label"] != "REVIEW RECOMMENDED" else "Uncertain")
        bar_class = "conf-bar-fill-cmb" if classification != "Mimic" else "conf-bar-fill-mimic"
        badge_html = ('<span class="badge-reviewed">✓ REVIEWED</span>' if is_reviewed
                      else '<span class="badge-review">● NEEDS REVIEW</span>')

        card_class = "candidate-card selected" if is_selected else "candidate-card"
        st.markdown(f"""
        <div class="{card_class}">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="candidate-id">{c['id']}</span>
                {badge_html}
            </div>
            <div class="candidate-region">{c['region']}</div>
            <div class="candidate-classlabel">{classification} &nbsp;·&nbsp; {c['confidence']*100:.0f}% conf.</div>
            <div class="conf-bar-bg"><div class="{bar_class}" style="width:{c['confidence']*100:.0f}%;"></div></div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(f"Open {c['id']}", key=f"open_{c['id']}", use_container_width=True):
            st.session_state.selected_idx = i
            st.rerun()

with right:
    selected = candidates[st.session_state.selected_idx]

    st.markdown('<div class="section-label">VIEWER CONTROLS</div>', unsafe_allow_html=True)
    vc1, vc2, vc3 = st.columns(3)
    show_heatmap = vc1.toggle("Grad-CAM overlay", value=True)
    show_markers = vc2.toggle("Candidate markers", value=True)
    show_crosshair = vc3.toggle("Crosshair guide", value=False)

    # ---- Build the slice image ----
    z = selected["center"][2]
    slice_data = volume[:, :, z].T
    vmin, vmax = np.percentile(slice_data, (1, 99))

    fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=100)
    ax.imshow(slice_data, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)

    if show_heatmap:
        try:
            patch = pipeline._cut_patch(volume, selected["center"], (16, 16, 8))
            patch_tensor = __import__("torch").tensor(patch, dtype=__import__("torch").float32).unsqueeze(0).unsqueeze(0)
            patch_tensor.requires_grad_(True)
            target_layer = pipeline.stage2_model.features[6]
            cam = GradCAM3D(pipeline.stage2_model, target_layer)
            heatmap = cam.generate(patch_tensor)
            x, y, _ = selected["center"]
            half = (8, 8, 4)
            hx0, hx1 = x - half[0], x + half[0]
            hy0, hy1 = y - half[1], y + half[1]
            mid_z_local = heatmap.shape[2] // 2
            ax.imshow(heatmap[:, :, mid_z_local].T, cmap="jet", alpha=0.45, origin="lower",
                       extent=[hx0, hx1, hy0, hy1])
        except Exception:
            pass

    if show_markers:
        for c in candidates:
            if c["center"][2] == z:
                cx, cy, _ = c["center"]
                is_sel = c["id"] == selected["id"]
                color = "#ea580c" if is_sel else "#facc15"
                circle = plt.Circle((cx, cy), 4, color=color, fill=False, linewidth=2 if is_sel else 1.2)
                ax.add_patch(circle)

    if show_crosshair:
        x, y, _ = selected["center"]
        ax.axhline(y, color="cyan", linewidth=0.5, alpha=0.6)
        ax.axvline(x, color="cyan", linewidth=0.5, alpha=0.6)

    ax.axis("off")
    fig.tight_layout()
    img_c1, img_c2, img_c3 = st.columns([1, 3, 1])
    with img_c2:
        st.pyplot(fig, use_container_width=False)

    st.markdown('<div class="section-label" style="margin-top:18px;">EVIDENCE DETAIL</div>',
                unsafe_allow_html=True)
    st.markdown(f"### {selected['region']}")
    st.caption(selected["coord_str"])

    ec1, ec2 = st.columns(2)
    ec1.markdown(f"""<div class="stat-card"><div class="stat-label">CONFIDENCE</div>
    <div class="stat-value">{selected['confidence']*100:.0f}%</div></div>""", unsafe_allow_html=True)
    ec2.markdown(f"""<div class="stat-card"><div class="stat-label">SLICE</div>
    <div class="stat-value">{z}</div></div>""", unsafe_allow_html=True)

    st.write("")
    st.markdown("**CLASSIFICATION**")
    cls1, cls2 = st.columns(2)
    if cls1.button("True microbleed", key="cls_cmb", use_container_width=True):
        st.session_state.classification_override[selected["id"]] = "True microbleed"
        st.session_state.reviewed[selected["id"]] = True
        st.rerun()
    if cls2.button("Mimic", key="cls_mimic", use_container_width=True):
        st.session_state.classification_override[selected["id"]] = "Mimic"
        st.session_state.reviewed[selected["id"]] = True
        st.rerun()

    st.markdown("**REVIEW SEVERITY**")
    sv1, sv2, sv3 = st.columns(3)
    sv1.button("Low", key="sev_low", use_container_width=True)
    sv2.button("Moderate", key="sev_mod", use_container_width=True)
    sv3.button("High", key="sev_high", use_container_width=True)

    conf_pct = selected["confidence"] * 100
    if conf_pct > 80:
        note = "Compact hypointense focus with a high attention response."
    elif conf_pct > 60:
        note = "Moderate hypointense focus; attention response is present but less concentrated."
    else:
        note = "Borderline signal — attention response is diffuse. Manual review recommended."

    st.markdown(f"""
    **✨ Explainability note**  
    <span style="color:#475569; font-size:13px;">{note}</span>
    """, unsafe_allow_html=True)

    st.caption("ℹ️ Attention maps show where the model focused; they do not establish "
               "causality or clinical significance. Region labels are coordinate-based "
               "heuristics, not a validated anatomical atlas.")