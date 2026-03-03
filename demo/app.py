import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use('Agg')

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection
import contextlib
import io
import warnings

from generate_plots import make_circles_np, make_moons_np, make_spirals
from cluster import spectral_clustering
from knn_cluster import kmeans_clustering

st.set_page_config(page_title="Spectral Clustering", layout="wide")

# ── Theme ─────────────────────────────────────────────────────────────────────

BG     = '#020617'
SURFACE = '#0f172a'
TEXT   = '#e2e8f0'
MUTED  = '#64748b'
ACCENT = '#818cf8'
PINK   = '#f472b6'
CMAP   = ListedColormap([ACCENT, PINK])

# ── CSS — constrain plots so everything fits in one viewport ──────────────────

st.markdown(f"""<style>
.block-container {{padding-top:1rem; padding-bottom:0;}}
hr {{border-color:{SURFACE} !important; margin:0.4rem 0 !important;}}
div.stButton > button {{width:100%;}}
.block-container img {{max-height:60vh; object-fit:contain;}}
.stAppDeployButton {{display:none;}}
header[data-testid="stHeader"] {{visibility:hidden; height:0; padding:0;}}
.stAppViewBlockContainer {{padding-top:1rem;}}
[data-testid="manage-app-button"] {{display:none;}}
.viewerBadge_container__r5tak {{display:none;}}

/* ── Mobile ─────────────────────────────────────────────────────── */
@media (max-width: 768px) {{
    [data-testid="stHorizontalBlock"] {{
        flex-wrap: wrap !important;
    }}
    [data-testid="column"] {{
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }}
    .block-container {{
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
    }}
    .mobile-subtitle {{
        display: block !important;
        margin-left: 0 !important;
        margin-top: 0.25rem;
    }}
    .katex-display {{
        overflow-x: auto !important;
        overflow-y: hidden !important;
        -webkit-overflow-scrolling: touch;
    }}
    div.stButton > button {{
        min-height: 2.8rem;
        font-size: 0.95rem;
    }}
}}
</style>""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def dark_fig(w=6, h=6):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUTED, which='both', labelsize=9)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return fig, ax

def show(fig):
    st.pyplot(fig, width="stretch")
    plt.close(fig)

def col_label(text, color=TEXT):
    st.markdown(
        f"<p style='text-align:center;color:{color};font-weight:600;"
        f"font-size:0.85rem;margin-bottom:0.2rem;'>{text}</p>",
        unsafe_allow_html=True,
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────

NOISE_CAPS = {"Rings": 0.09, "Moons": 0.10, "Spirals": 0.22}

with st.sidebar:
    st.markdown(
        f'<p style="font-size:0.75rem;color:{MUTED};letter-spacing:0.1em;'
        f'text-transform:uppercase;margin-bottom:0.5rem;">Dataset</p>',
        unsafe_allow_html=True,
    )
    shape = st.radio("Shape", list(NOISE_CAPS), label_visibility="collapsed")
    st.markdown("")
    st.markdown(
        f'<p style="font-size:0.75rem;color:{MUTED};letter-spacing:0.1em;'
        f'text-transform:uppercase;margin-bottom:0.5rem;">Noise level</p>',
        unsafe_allow_html=True,
    )
    cap = NOISE_CAPS[shape]
    noise = st.slider("Noise", 0.005, cap, cap, 0.005, "%.3f", label_visibility="collapsed")
    st.markdown("")
    st.markdown("---")
    st.markdown(
        f'<p style="color:{MUTED};font-size:0.8rem;line-height:1.6;">'
        f'Spectral clustering implemented from scratch using only NumPy. '
        f'No scikit-learn.<br><br>'
        f'<span style="color:{TEXT};font-size:1.1rem;font-weight:600;">Eren Menges, 2026</span><br>'
        f'<a href="https://erenmenges.com" target="_blank" '
        f'style="color:{ACCENT};text-decoration:none;font-size:1rem;">erenmenges.com</a></p>',
        unsafe_allow_html=True,
    )

k_neighbors = 10

# ── Data & algorithms (cached) ───────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_clustering(shape_name, noise_val, k_nn):
    generators = {
        "Rings":   lambda n: make_circles_np(500, factor=0.5, noise=n, random_state=42)[0],
        "Moons":   lambda n: make_moons_np(500, noise=n, random_state=42)[0],
        "Spirals": lambda n: make_spirals(500, noise=n, random_state=42)[0],
    }
    data = generators[shape_name](noise_val)

    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        labels_km, _, fig_km = kmeans_clustering(data.copy(), k=2)
        labels_sc, data_sc, fig_knn, fig_sc, info = spectral_clustering(
            data.copy(), k_neighbors=k_nn,
        )

    plt.close(fig_km)
    plt.close(fig_sc)
    plt.close(fig_knn)

    return data, labels_km, labels_sc, data_sc, info

data, labels_km, labels_sc, data_sc, info = run_clustering(shape, noise, k_neighbors)

A     = info['adjacency_matrix']
evals = info['eigenvalues']
evecs = info['k_eigenvectors']
deg   = info['degrees']

# ── Steps ─────────────────────────────────────────────────────────────────────

STEPS = [
    "Input Data",
    "Similarity & KNN Graph",
    "Laplacian & Eigendecomposition",
    "Spectral Embedding",
    "Final Comparison",
]
N = len(STEPS)

if 'step' not in st.session_state:
    st.session_state.step = 0
if st.session_state.get('prev_shape') != shape:
    st.session_state.step = 0
    st.session_state.prev_shape = shape

def go_next():
    st.session_state.step = min(st.session_state.step + 1, N - 1)

def go_back():
    st.session_state.step = max(st.session_state.step - 1, 0)

def go_end():
    st.session_state.step = N - 1

s = st.session_state.step

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE
# ══════════════════════════════════════════════════════════════════════════════

# ── Title (compact) ───────────────────────────────────────────────────────────

st.markdown(
    f'<div style="margin-bottom:0.2rem;">'
    f'<span style="font-size:1.8rem;font-weight:700;letter-spacing:-0.03em;">'
    f'The Beauty of Spectral Clustering</span>'
    f'<span class="mobile-subtitle" style="color:{MUTED};font-size:0.85rem;margin-left:12px;">'
    f'from scratch with NumPy &mdash; '
    f'{shape} · noise {noise:.3f} · {data.shape[0]} pts'
    f'</span></div>',
    unsafe_allow_html=True,
)

# ── Step header + LaTeX ───────────────────────────────────────────────────────

st.markdown(
    f'<div style="display:flex;align-items:baseline;gap:14px;'
    f'margin:0.6rem 0 0.3rem;">'
    f'<span style="font-size:2.2rem;font-weight:800;color:{ACCENT};'
    f'opacity:.15;line-height:1;font-variant-numeric:tabular-nums;">'
    f'{s + 1:02d}</span>'
    f'<span style="font-size:1.1rem;font-weight:600;color:{TEXT};">'
    f'{STEPS[s]}</span></div>',
    unsafe_allow_html=True,
)

if s == 0:
    st.markdown(
        "Classical K-means clustering **fails** with these shapes. It assigns each point to its **nearest centroid**, which only works "
        "when clusters are convex blobs."
    )
    

elif s == 1:
    st.markdown(
        "Compute pairwise distances via the Gram matrix, then build a "
        "binary *k*-NN graph and symmetrize via mutual k-NN."
    )
    st.latex(
        r"d_{ij}^{\,2} = \|x_i\|^2 + \|x_j\|^2 - 2\,x_i^\top x_j"
        r"\qquad A \leftarrow \min(A, A^\top)"
    )

elif s == 2:
    st.markdown(
        "We build the degree matrix, form the normalized Laplacian, then "
        "decompose it. The **spectral gap** suggests the number of clusters."
    )
    st.latex(
        r"d_i = \textstyle\sum_j A_{ij},"
        r"\quad L_{\mathrm{sym}} = I - D^{-1/2} A \, D^{-1/2},"
        r"\quad L_{\mathrm{sym}}\,v_i = \lambda_i\,v_i"
    )

elif s == 3:
    st.markdown(
        "We take the *k* eigenvectors with the smallest eigenvalues and row-normalize. "
        "Non-convex clusters become **linearly separable**."
    )
    st.latex(
        r"U = [\, v_1 \;\; v_2 \;\; \cdots \;\; v_k \,],"
        r"\qquad \hat{u}_i = u_i \,/\, \|u_i\|"
    )

elif s == 4:
    st.markdown(
        "Beautiful results: K-means in the spectral embedding vs. K-means applied directly."
    )

# ── Two-column layout ────────────────────────────────────────────────────────

left, right = st.columns(2, gap="large")

with left:
    col_label("K-Means (direct)", MUTED)
    fig, ax = dark_fig(5, 5)
    ax.scatter(
        data[:, 0], data[:, 1],
        c=labels_km, cmap=CMAP, s=14, alpha=0.85, edgecolors='none',
    )
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    show(fig)

with right:
    if s == 0:
        col_label("Raw Data")
        fig, ax = dark_fig(5, 5)
        ax.scatter(
            data[:, 0], data[:, 1],
            c=MUTED, s=12, alpha=0.7, edgecolors='none',
        )
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        show(fig)

    elif s == 1:
        col_label(f"k-NN Graph (k = {k_neighbors})")
        fig, ax = dark_fig(5, 5)
        rows, cols = np.where(np.triu(A, k=1) > 0)
        segs = np.stack([data_sc[rows], data_sc[cols]], axis=1)
        ax.add_collection(
            LineCollection(segs, colors=ACCENT, alpha=0.35, linewidths=0.8)
        )
        ax.scatter(
            data_sc[:, 0], data_sc[:, 1],
            c=TEXT, s=10, alpha=0.95, edgecolors='none', zorder=2,
        )
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.autoscale_view()
        fig.tight_layout()
        show(fig)

    elif s == 2:
        col_label("Eigenvalue Spectrum")
        K = 2
        n_show = min(5, len(evals))
        fig, ax = dark_fig(5, 4.5)
        bar_c = [PINK if i < K else ACCENT for i in range(n_show)]
        ax.bar(range(n_show), evals[:n_show], color=bar_c, alpha=0.85, width=0.7)
        gap_x = K - 0.5
        gap_v = evals[K] - evals[K - 1]
        ax.axvline(gap_x, color=PINK, ls='--', alpha=0.45, lw=1.5)
        mid_y = (evals[K - 1] + evals[K]) / 2
        target_y = evals[n_show - 1] * 0.7
        ax.annotate(
            f'spectral gap\nΔλ = {gap_v:.4f}',
            xy=(gap_x, mid_y),
            xytext=(gap_x + 1.8, target_y),
            fontsize=9, color=PINK, fontweight='500',
            arrowprops=dict(arrowstyle='->', color=PINK, lw=1.2),
        )
        ax.set_xlabel('index', color=MUTED, fontsize=10)
        ax.set_ylabel('λ', color=MUTED, fontsize=11)
        ax.set_xticks(range(n_show))
        ax.tick_params(axis='x', labelsize=8)
        fig.tight_layout()
        show(fig)

    elif s == 3:
        col_label("Eigenvector Space")
        fig, ax = dark_fig(5, 5)
        ax.scatter(
            evecs[:, 0], evecs[:, 1],
            c=labels_sc, cmap=CMAP, s=14, alpha=0.85, edgecolors='none',
        )
        ax.set_xlabel('û₁', color=MUTED, fontsize=12)
        ax.set_ylabel('û₂', color=MUTED, fontsize=12)
        ax.set_aspect('equal')
        fig.tight_layout()
        show(fig)

    elif s == 4:
        col_label("Spectral Clustering")
        fig, ax = dark_fig(5, 5)
        ax.scatter(
            data_sc[:, 0], data_sc[:, 1],
            c=labels_sc, cmap=CMAP, s=14, alpha=0.85, edgecolors='none',
        )
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        show(fig)

# ── Navigation ────────────────────────────────────────────────────────────────

st.markdown("---")

nav_l, nav_m, nav_skip, nav_r = st.columns([1, 2, 1, 1])

with nav_l:
    st.button("◀  Back", on_click=go_back, disabled=(s == 0))

with nav_m:
    dots = ''.join(
        f'<span style="color:{ACCENT if i == s else MUTED};'
        f'font-size:0.6rem;margin:0 5px;">●</span>'
        for i in range(N)
    )
    st.markdown(
        f'<div style="text-align:center;padding-top:0.45rem;">'
        f'{dots}<br>'
        f'<span style="color:{MUTED};font-size:0.8rem;">'
        f'Step {s + 1} of {N}</span></div>',
        unsafe_allow_html=True,
    )

with nav_skip:
    st.button("⏭  Result", on_click=go_end, disabled=(s == N - 1))

with nav_r:
    st.button("Next  ▶", on_click=go_next, disabled=(s == N - 1))
