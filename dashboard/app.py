"""
CMB Review — clinician-style review dashboard, wired to the real detection pipeline.
"""

import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src" / "pipeline"))
sys.path.append(str(PROJECT_ROOT / "src" / "severity_gradcam"))
from final_pipeline import CMBDetectionPipeline
from gradcam_3d import GradCAM3D

# =========================================================
# PAGE CONFIG + CSS
# =========================================================
st.set_page_config(page_title="CMB Review", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header[data-testid="stHeader"] {visibility: hidden; height: 0;}
    .stApp { background-color: #f7f7f5; }

    section[data-testid="stSidebar"] { background-color: #0f172a; }
    section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

    .sidebar-logo { display:flex; align-items:center; gap:10px; padding: 4px 0 20px 0; }
    .sidebar-logo .icon-circle {
        background:#0d9488; border-radius:50%; width:36px; height:36px;
        display:flex; align-items:center; justify-content:center; font-size:16px; flex-shrink:0;
    }
    .sidebar-logo .title { font-weight: 700; font-size: 17px; color:#f1f5f9; }
    .sidebar-logo .subtitle { font-size: 10px; letter-spacing: 1.5px; color:#64748b; }

    .demo-card {
        background: #1e293b; border-radius: 10px; padding: 14px 16px; margin-bottom: 20px;
        border-left: 3px solid #f97316;
    }
    .demo-card .tag { font-size: 10px; letter-spacing: 1px; color: #fb923c; font-weight:700; }
    .demo-card .name { font-size: 15px; font-weight: 700; color: #f1f5f9; margin-top: 6px; line-height:1.3;}
    .demo-card .desc { font-size: 11px; color: #94a3b8; margin-top:4px; }

    .nav-header { font-size:11px; letter-spacing:1.5px; color:#64748b; font-weight:700; margin: 4px 0 8px 2px;}

    .safety-box {
        background: #1e293b; border-radius: 10px; padding: 12px 14px; font-size: 11px;
        color: #94a3b8; margin-top: 20px; line-height:1.5;
    }
    .safety-box .head { color:#e2e8f0; font-weight:700; font-size:11px; margin-bottom:5px;}

    .stat-card {
        background: white; border-radius: 12px; padding: 16px 18px;
        border: 1px solid #e5e7eb; height: 100%; position: relative;
    }
    .stat-label { font-size: 10px; letter-spacing: 1px; color: #94a3b8; font-weight:700; text-transform:uppercase;}
    .stat-value { font-size: 26px; font-weight: 800; color: #0f172a; margin-top: 4px;}
    .stat-sub { font-size: 11px; color: #94a3b8; margin-top: 2px;}
    .stat-icon { position:absolute; top:16px; right:16px; font-size:14px; color:#94a3b8; }

    .breadcrumb-label {
        color:#0d9488; font-weight:700; font-size:11px; letter-spacing:1.5px;
        text-transform:uppercase; margin-bottom:14px;
    }

    .user-card {
        display:flex; align-items:center; gap:10px; margin-top:24px;
        padding:10px 12px; background:#1e293b; border-radius:10px;
    }
    .user-avatar {
        width:34px; height:34px; border-radius:50%; background:#f97316;
        display:flex; align-items:center; justify-content:center;
        color:white; font-weight:700; font-size:12px; flex-shrink:0;
    }
    .user-name { font-size:13px; font-weight:700; color:#f1f5f9; line-height:1.3; }
    .user-role { font-size:11px; color:#94a3b8; line-height:1.3; }

    .safety-banner {
        background: #fef2f2; border: 1px solid #fecaca; border-radius: 10px;
        padding: 14px 18px; margin: 18px 0; font-size: 13px; color: #7f1d1d;
    }
    .safety-banner b { color:#991b1b; }

    .badge-review {
        background:#fff7ed; color:#c2410c; border:1px solid #fed7aa; border-radius: 20px;
        padding: 3px 10px; font-size: 11px; font-weight:700; white-space:nowrap;
    }
    .badge-reviewed {
        background:#f0fdf4; color:#15803d; border:1px solid #bbf7d0; border-radius: 20px;
        padding: 3px 10px; font-size: 11px; font-weight:700; white-space:nowrap;
    }

    .candidate-card {
        background: white; border: 1px solid #e5e7eb; border-radius: 12px;
        padding: 14px 18px; margin-bottom: 8px;
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
        text-transform: uppercase; margin: 4px 0 10px 0;
    }

    .viewer-row {
        display:flex; justify-content:space-between; align-items:center;
        padding: 10px 0; border-bottom: 1px solid #f1f5f9;
    }
    .viewer-row .vlabel { font-size:13px; font-weight:600; color:#0f172a; }
    .viewer-row .vsub { font-size:11px; color:#94a3b8; }

    div.stButton > button {
        border-radius: 8px; border: 1px solid #d1d5db; background: white; color:#0f172a;
        font-size: 13px;
    }
    div.stButton > button:hover { border-color:#0d9488; color:#0d9488; }
    div.stButton > button[kind="primary"] {
        background:#0d9488; border-color:#0d9488; color:white;
    }

    /* Segmented-control look for classification / severity radios */
    div[role="radiogroup"] { display:flex; gap:8px; flex-wrap:wrap; }
    div[role="radiogroup"] label {
        border:1px solid #d1d5db; border-radius:8px; padding:8px 16px !important;
        background:white; margin:0 !important; flex:1; justify-content:center;
    }
    div[role="radiogroup"] label div:first-child { display:none; }
    div[role="radiogroup"] label[aria-checked="true"] {
        border:1.5px solid #0d9488 !important; background:#f0fdfa !important;
    }
    div[role="radiogroup"] label[aria-checked="true"] p { color:#0d9488 !important; font-weight:700; }
</style>
""", unsafe_allow_html=True)


# =========================================================
# PIPELINE + STATE
# =========================================================
@st.cache_resource
def load_pipeline():
    return CMBDetectionPipeline()


pipeline = load_pipeline()

defaults = {
    "candidates": None, "reviewed": {}, "classification_override": {},
    "severity_override": {}, "selected_idx": 0, "volume": None,
    "subject_name": None, "last_run": None, "show_all_candidates": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def region_label(center, shape):
    """Coordinate-based region heuristic — NOT a validated anatomical atlas."""
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


def build_pdf_report(path, subject_name, candidates, reviewed, overrides, severity_overrides):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    y = height - 25 * mm

    c.setFont("Helvetica-Bold", 16)
    c.drawString(20 * mm, y, "Cerebral Microbleed Detection — Review Report")
    y -= 8 * mm
    c.setFont("Helvetica", 10)
    c.drawString(20 * mm, y, f"Subject: {subject_name}")
    y -= 5 * mm
    c.drawString(20 * mm, y, f"Generated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}")
    y -= 5 * mm
    c.drawString(20 * mm, y, "Demo analysis — not for clinical use.")
    y -= 10 * mm

    n_mimic = sum(1 for cid in overrides if overrides[cid] == "Mimic")
    n_cmb = len(candidates) - n_mimic
    c.setFont("Helvetica-Bold", 11)
    c.drawString(20 * mm, y, f"Total candidates: {len(candidates)}   |   True microbleed: {n_cmb}   |   Mimic: {n_mimic}   |   Reviewed: {len(reviewed)}")
    y -= 10 * mm

    c.setFont("Helvetica-Bold", 9)
    c.drawString(20 * mm, y, "ID")
    c.drawString(35 * mm, y, "Region")
    c.drawString(80 * mm, y, "Confidence")
    c.drawString(105 * mm, y, "Classification")
    c.drawString(140 * mm, y, "Severity")
    c.drawString(170 * mm, y, "Status")
    y -= 4 * mm
    c.line(20 * mm, y, 190 * mm, y)
    y -= 5 * mm

    c.setFont("Helvetica", 8)
    for cand in candidates:
        if y < 20 * mm:
            c.showPage()
            y = height - 20 * mm
            c.setFont("Helvetica", 8)
        classification = overrides.get(cand["id"], "True microbleed" if cand["confidence_label"] != "REVIEW RECOMMENDED" else "Uncertain")
        severity = severity_overrides.get(cand["id"], "—")
        status = "Reviewed" if cand["id"] in reviewed else "Needs review"
        c.drawString(20 * mm, y, cand["id"])
        c.drawString(35 * mm, y, cand["region"])
        c.drawString(80 * mm, y, f"{cand['confidence']*100:.0f}%")
        c.drawString(105 * mm, y, classification)
        c.drawString(140 * mm, y, severity)
        c.drawString(170 * mm, y, status)
        y -= 5.5 * mm

    c.save()


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="icon-circle">🧠</div>
        <div>
            <div class="title">CMB Review</div>
            <div class="subtitle">NEUROIMAGING LAB</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.subject_name:
        st.markdown(f"""
        <div class="demo-card">
            <div class="tag">● DEMO ANALYSIS</div>
            <div class="name">{st.session_state.subject_name}</div>
            <div class="desc">SWI scan review session</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="nav-header">WORKSPACE</div>', unsafe_allow_html=True)
    page = st.radio("nav", ["Review workspace", "Methodology", "Literature map"],
                     label_visibility="collapsed")

    st.markdown('<div class="nav-header" style="margin-top:20px;">UPLOAD SCAN</div>', unsafe_allow_html=True)
    sidebar_upload = st.file_uploader("Upload SWI scan (.nii.gz)", type=["nii.gz", "gz"],
                                       label_visibility="collapsed")
    st.session_state["_sidebar_uploaded_file"] = sidebar_upload

    st.markdown("""
    <div class="safety-box">
        <div class="head">🛡️ SAFETY FIRST</div>
        Research interface only. Outputs require qualified clinical review.
    </div>
    """, unsafe_allow_html=True)

    # ---- Team member profile card (edit TEAM_MEMBER_NAME / TEAM_ROLE below) ----
    TEAM_MEMBER_NAME = "Harshitha V"
    TEAM_ROLE = "Capstone team"
    initials = "".join(w[0] for w in TEAM_MEMBER_NAME.split()[:2]).upper()
    st.markdown(f"""
    <div class="user-card">
        <div class="user-avatar">{initials}</div>
        <div>
            <div class="user-name">{TEAM_MEMBER_NAME}</div>
            <div class="user-role">{TEAM_ROLE}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# METHODOLOGY / LITERATURE PAGES
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
                "mimic rejection, and confidence-aware clinical deployment.")
    st.stop()

# =========================================================
# MAIN — HEADER
# =========================================================
top_l, top_r = st.columns([5, 1])
with top_l:
    st.caption("CMB / REVIEW")
with top_r:
    st.caption("🟢 Local demo mode   ❓")

st.markdown('<div class="breadcrumb-label">● REVIEW WORKSPACE / DEMO</div>', unsafe_allow_html=True)

header_left, header_right = st.columns([3, 1.1])
with header_left:
    st.markdown("# Read the evidence,")
    st.markdown("## :teal[not just the output.]")
    st.write("A clinician-style review surface for cerebral microbleed candidates, "
             "mimics, uncertainty, and model attention.")
with header_right:
    st.write("")
    st.write("")
    run_clicked = st.button("🔄 Re-run analysis", use_container_width=True)
    export_clicked = st.button("⬇ Export report", use_container_width=True, type="primary",
                                disabled=(st.session_state.candidates is None))

# File uploader lives in the sidebar (see below) — this reads whatever is
# currently selected there, so the header row stays clean like the reference.
uploaded_file = st.session_state.get("_sidebar_uploaded_file")

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
            "id": f"CMB-{i+1:02d}", "region": region, "coord_str": coord_str,
            "center": d["center"], "confidence": d["confidence"],
            "confidence_label": d["confidence_label"],
        })

    st.session_state.candidates = candidates
    st.session_state.volume = volume
    st.session_state.subject_name = uploaded_file.name.replace(".nii.gz", "")
    st.session_state.reviewed = {}
    st.session_state.classification_override = {}
    st.session_state.severity_override = {}
    st.session_state.selected_idx = 0
    st.session_state.show_all_candidates = False
    st.session_state.last_run = datetime.now()
    st.session_state["slice_slider"] = candidates[0]["center"][2] if candidates else 0

if export_clicked and st.session_state.candidates:
    out_path = PROJECT_ROOT / "reports" / "cmb_review_report.pdf"
    out_path.parent.mkdir(exist_ok=True)
    build_pdf_report(out_path, st.session_state.subject_name, st.session_state.candidates,
                      st.session_state.reviewed, st.session_state.classification_override,
                      st.session_state.severity_override)
    with open(out_path, "rb") as f:
        st.download_button("📄 Download PDF report", f, file_name="cmb_review_report.pdf",
                            mime="application/pdf")

# =========================================================
# EMPTY STATE
# =========================================================
if st.session_state.candidates is None:
    st.info("👆 Upload an SWI scan in the sidebar and click **Re-run analysis** to begin a review session.")
    st.stop()

candidates = st.session_state.candidates
volume = st.session_state.volume
n_total = len(candidates)
n_reviewed = len(st.session_state.reviewed)
mean_conf = np.mean([c["confidence"] for c in candidates]) * 100 if candidates else 0
n_mimic = sum(1 for cid, v in st.session_state.classification_override.items() if v == "Mimic")
n_cmb = n_total - n_mimic
last_run_str = st.session_state.last_run.strftime("Today, %I:%M %p") if st.session_state.last_run else "—"

# =========================================================
# STAT CARDS
# =========================================================
s1, s2, s3, s4 = st.columns(4)
with s1:
    st.markdown(f"""<div class="stat-card"><div class="stat-icon">▦</div><div class="stat-label">CANDIDATES</div>
    <div class="stat-value">{n_total:02d}</div>
    <div class="stat-sub">{n_cmb} microbleed · {n_mimic} mimic</div></div>""", unsafe_allow_html=True)
with s2:
    st.markdown(f"""<div class="stat-card"><div class="stat-icon">✓</div><div class="stat-label">REVIEWED</div>
    <div class="stat-value">{n_reviewed:02d}</div>
    <div class="stat-sub">{n_total - n_reviewed} still need attention</div></div>""", unsafe_allow_html=True)
with s3:
    st.markdown(f"""<div class="stat-card"><div class="stat-icon">◔</div><div class="stat-label">MEAN CONFIDENCE</div>
    <div class="stat-value">{mean_conf:.1f}%</div>
    <div class="stat-sub">Candidate-level estimate</div></div>""", unsafe_allow_html=True)
with s4:
    st.markdown(f"""<div class="stat-card"><div class="stat-icon">⏱</div><div class="stat-label">LAST RUN</div>
    <div class="stat-value" style="font-size:18px;">{last_run_str}</div>
    <div class="stat-sub">Demo inference only</div></div>""", unsafe_allow_html=True)

st.markdown("""
<div class="safety-banner">
⚠️ <b>Demo analysis — not for clinical use.</b><br>
This capstone interface demonstrates a research workflow using real model outputs on
sample scans. It is not a medical diagnostic device and does not replace radiologist judgment.
</div>
""", unsafe_allow_html=True)

# =========================================================
# CANDIDATE QUEUE + EVIDENCE DETAIL
# =========================================================
left, right = st.columns([1.1, 1.4])

with left:
    st.markdown(f'<div class="section-label">CANDIDATE QUEUE — FINDINGS {n_total}/{n_total}</div>',
                unsafe_allow_html=True)

    DISPLAY_LIMIT = 8
    show_all = st.session_state.show_all_candidates
    visible_candidates = candidates if show_all else candidates[:DISPLAY_LIMIT]

    for i, c in enumerate(candidates):
        if not show_all and i >= DISPLAY_LIMIT:
            break
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
            st.session_state["slice_slider"] = c["center"][2]
            st.rerun()

    if n_total > DISPLAY_LIMIT:
        if not show_all:
            if st.button(f"Show all {n_total} candidates", use_container_width=True):
                st.session_state.show_all_candidates = True
                st.rerun()
        else:
            if st.button("Show fewer", use_container_width=True):
                st.session_state.show_all_candidates = False
                st.rerun()

with right:
    selected = candidates[st.session_state.selected_idx]

    vc_head_l, vc_head_r = st.columns([5, 1])
    with vc_head_l:
        st.markdown('<div class="section-label">VIEWER CONTROLS — OVERLAYS</div>', unsafe_allow_html=True)

    show_heatmap = st.toggle("Attention heatmap  ·  *Grad-CAM-style*", value=True)
    show_markers = st.toggle(f"Candidate markers  ·  *{n_total} detected*", value=True)
    show_crosshair = st.toggle("Crosshair guide  ·  *Coordinate aid*", value=False)

    max_slice = volume.shape[2] - 1
    if "slice_slider" not in st.session_state:
        st.session_state["slice_slider"] = selected["center"][2]
    st.session_state["slice_slider"] = int(np.clip(st.session_state["slice_slider"], 0, max_slice))

    slices_with_detections = sorted(set(c["center"][2] for c in candidates))
    z = st.slider(
        f"Slice  ({len(slices_with_detections)} slice(s) have detections)",
        0, max_slice, key="slice_slider",
    )
    n_on_this_slice = sum(1 for c in candidates if c["center"][2] == z)
    st.caption(f"📍 {n_on_this_slice} detection(s) on slice {z}" if n_on_this_slice
               else f"No detections on slice {z} — try a slice from: {slices_with_detections[:8]}{'...' if len(slices_with_detections) > 8 else ''}")

    slice_data = volume[:, :, z].T
    vmin, vmax = np.percentile(slice_data, (1, 99))

    fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=100)
    ax.imshow(slice_data, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)

    candidates_on_slice = [c for c in candidates if c["center"][2] == z]
    heatmap_target = selected if selected["center"][2] == z else (candidates_on_slice[0] if candidates_on_slice else None)

    if show_heatmap and heatmap_target is not None:
        try:
            patch = pipeline._cut_patch(volume, heatmap_target["center"], (16, 16, 8))
            patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            patch_tensor.requires_grad_(True)
            target_layer = pipeline.stage2_model.features[6]
            cam = GradCAM3D(pipeline.stage2_model, target_layer)
            heatmap = cam.generate(patch_tensor)
            x, y, _ = heatmap_target["center"]
            half = (8, 8, 4)
            hx0, hx1 = x - half[0], x + half[0]
            hy0, hy1 = y - half[1], y + half[1]
            mid_z_local = heatmap.shape[2] // 2
            ax.imshow(heatmap[:, :, mid_z_local].T, cmap="jet", alpha=0.45, origin="lower",
                       extent=[hx0, hx1, hy0, hy1])
        except Exception:
            pass

    if show_markers:
        for c in candidates_on_slice:
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
    <div class="stat-value">{z}</div><div class="stat-sub">Viewing slice {z} of {max_slice}</div></div>""",
    unsafe_allow_html=True)

    st.write("")
    st.markdown("**CLASSIFICATION**")
    current_class = st.session_state.classification_override.get(
        selected["id"], "True microbleed" if selected["confidence_label"] != "REVIEW RECOMMENDED" else "Mimic")
    new_class = st.radio("classification", ["True microbleed", "Mimic"],
                          index=0 if current_class == "True microbleed" else 1,
                          horizontal=True, label_visibility="collapsed", key=f"radio_class_{selected['id']}")
    if new_class != current_class:
        st.session_state.classification_override[selected["id"]] = new_class
        st.session_state.reviewed[selected["id"]] = True
        st.rerun()

    st.markdown("**REVIEW SEVERITY**")
    current_sev = st.session_state.severity_override.get(selected["id"], "Moderate")
    new_sev = st.radio("severity", ["Low", "Moderate", "High"],
                        index=["Low", "Moderate", "High"].index(current_sev),
                        horizontal=True, label_visibility="collapsed", key=f"radio_sev_{selected['id']}")
    if new_sev != current_sev:
        st.session_state.severity_override[selected["id"]] = new_sev
        st.session_state.reviewed[selected["id"]] = True
        st.rerun()

    conf_pct = selected["confidence"] * 100
    if conf_pct > 80:
        note = "Compact hypointense focus with a high attention response."
    elif conf_pct > 60:
        note = "Moderate hypointense focus; attention response is present but less concentrated."
    else:
        note = "Borderline signal — attention response is diffuse. Manual review recommended."

    st.markdown(f"**✨ Explainability note**  \n<span style='color:#475569; font-size:13px;'>{note}</span>",
                unsafe_allow_html=True)
    st.caption("ℹ️ Attention maps show where the model focused; they do not establish "
               "causality or clinical significance. Region labels are coordinate-based "
               "heuristics, not a validated anatomical atlas.")