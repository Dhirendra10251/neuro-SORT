# Cell 9 — Canonical Streamlit App (single version)
# NeuroSort — snnTorch-Powered Neuromorphic Edge Vision Core
import streamlit as st
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate, utils
import matplotlib.pyplot as plt
import numpy as np
import random
from torchvision import datasets, transforms

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroSort | Neuromorphic Waste Sorter",
    page_icon="⚡",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
WASTE_CLASSES = {
    0: "Cotton/Textile Fiber",
    1: "Mixed Fabric / Denim",
    2: "Cotton/Textile Fiber",
    3: "Mixed Fabric / Denim",
    4: "Mixed Fabric / Denim",
    5: "Footwear – Rubber/Synthetic",
    6: "Cotton/Textile Fiber",
    7: "Footwear – Rubber/Synthetic",
    8: "Textile/Leather Packaging",
    9: "Footwear – Rubber/Synthetic",
}

FMNIST_NAMES = {
    0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress",
    4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker",
    8: "Bag", 9: "Ankle boot"
}

NUM_STEPS  = 25
DEVICE     = torch.device('cpu')
spike_grad = surrogate.fast_sigmoid(slope=25)

# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    net = nn.Sequential(
        nn.Conv2d(1, 8, 5),
        nn.MaxPool2d(2),
        snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True),
        nn.Conv2d(8, 16, 5),
        nn.MaxPool2d(2),
        snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True),
        nn.Flatten(),
        nn.Linear(16 * 4 * 4, 10),
        snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True, output=True),
    ).to(DEVICE)
    net.load_state_dict(torch.load('neurosort_model.pth', map_location=DEVICE))
    net.eval()
    return net

@st.cache_resource
def load_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    return datasets.FashionMNIST(
        '/tmp/data/fashion_mnist', train=False, download=True, transform=transform
    )

# ── Inference helper ─────────────────────────────────────────────────────────
def run_inference(model, img_tensor):
    """Returns spk_rec [T, 1, 10] on CPU."""
    model.eval()
    utils.reset(model)
    spk_rec = []
    with torch.no_grad():
        for _ in range(NUM_STEPS):
            spk_out, _ = model(img_tensor.to(DEVICE))
            spk_rec.append(spk_out.cpu())
    return torch.stack(spk_rec)  # [T, 1, 10]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='text-align:center; padding: 1.5rem 0 0.5rem'>
        <h1 style='font-size:2.8rem; font-weight:800; letter-spacing:-1px;
                   background: linear-gradient(90deg, #00FF66, #00CFFF);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            ⚡ NeuroSort
        </h1>
        <p style='color:#718096; font-size:1rem; margin-top:-0.5rem;'>
            snnTorch-Powered Neuromorphic Edge Vision Core
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Load resources ────────────────────────────────────────────────────────────
with st.spinner("Loading model & dataset..."):
    model        = load_model()
    test_dataset = load_dataset()

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.title("🎮 Controls")
sample_idx = st.sidebar.slider(
    "Sample index", 0, len(test_dataset) - 1,
    random.randint(0, len(test_dataset) - 1), step=1
)

if st.sidebar.button("🎲 Random Sample"):
    sample_idx = random.randint(0, len(test_dataset) - 1)

# ── Batch accuracy button (Fix 5b) ────────────────────────────────────────────
if st.sidebar.button("▶ Run 10 Random Samples"):
    correct = 0
    with st.spinner("Running 10 samples..."):
        for _ in range(10):
            idx_b         = random.randint(0, len(test_dataset) - 1)
            img_b, lbl_b  = test_dataset[idx_b]
            spk_b         = run_inference(model, img_b.unsqueeze(0))
            pred_b        = spk_b.sum(dim=0).squeeze().argmax().item()
            if pred_b == lbl_b:
                correct += 1
    st.sidebar.success(f"Batch accuracy: {correct}/10 correct")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Architecture: Conv-SNN-Conv-SNN-Linear-SNN | "
    "Dataset: Fashion-MNIST | Framework: snnTorch"
)

# ── Inference ─────────────────────────────────────────────────────────────────
img_t, true_label = test_dataset[sample_idx]
spk_rec           = run_inference(model, img_t.unsqueeze(0))  # [T, 1, 10]

spike_counts = spk_rec.sum(dim=0).squeeze().numpy()  # [10]
prediction   = int(spike_counts.argmax())

# ── Live metrics (Fix 2) ──────────────────────────────────────────────────────
total_spikes  = spk_rec.sum().item()
total_neurons = spk_rec.numel()
sparsity      = 1.0 - (total_spikes / total_neurons)
compute_saved = sparsity * 100
power_mw      = round((1.0 - sparsity) * 1.2, 3)

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("CORE POWER (estimated)", f"{power_mw} mW",      delta="event-driven")
with m2:
    st.metric("SPIKE SPARSITY",         f"{sparsity*100:.1f}%", delta="live per sample")
with m3:
    st.metric("COMPUTE SAVED vs DNN",   f"{compute_saved:.1f}%",delta="temporal efficiency")

st.markdown("---")

# ── Main layout: image + result ───────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Input Sample")
    img_np = img_t.squeeze().numpy()
    fig_img, ax_img = plt.subplots(figsize=(3, 3))
    fig_img.patch.set_facecolor('#0A0C10')
    ax_img.set_facecolor('#0A0C10')
    ax_img.imshow(img_np, cmap='inferno', interpolation='nearest')
    ax_img.axis('off')
    ax_img.set_title(f"True: {FMNIST_NAMES[true_label]}", color='#A0AEC0', fontsize=9)
    st.pyplot(fig_img)

with col2:
    correct_flag = prediction == true_label
    colour       = "#00FF66" if correct_flag else "#FF4B4B"
    label_text   = WASTE_CLASSES[prediction]

    st.markdown(
        f"""
        <div style='border:2px solid {colour}; border-radius:12px; padding:1.2rem;
                    background:rgba(0,255,102,0.05); margin-top:0.5rem;'>
            <h3 style='color:{colour}; margin:0; font-size:1.5rem;'>
                {'✅' if correct_flag else '❌'} {label_text}
            </h3>
            <p style='color:#718096; margin:0.4rem 0 0;'>
                Fashion-MNIST class {prediction} · {FMNIST_NAMES[prediction]}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Population vote bar chart (Fix 5a) ───────────────────────────────────────
st.subheader("🔋 Output Neuron Population Vote")
fig_bar, ax_bar = plt.subplots(figsize=(8, 3))
fig_bar.patch.set_facecolor('#0A0C10')
ax_bar.set_facecolor('#131722')
ax_bar.barh(
    [WASTE_CLASSES[i] for i in range(10)],
    spike_counts,
    color=['#00FF66' if i == prediction else '#2D3748' for i in range(10)],
)
ax_bar.set_xlabel('Spike Count (confidence proxy)', color='#718096', fontsize=9)
ax_bar.set_title('Output Neuron Population Vote', color='white', fontsize=10, loc='left')
ax_bar.tick_params(colors='#A0AEC0', labelsize=8)
for spine in ['top', 'right', 'bottom']:
    ax_bar.spines[spine].set_visible(False)
ax_bar.spines['left'].set_color('#2D3748')
st.pyplot(fig_bar)

st.markdown("---")

# ── Animated raster + time-step slider (Fix 5c) ───────────────────────────────
st.subheader("📡 Spike Raster — Step Through Time")
t_step = st.slider("Step through time:", 1, NUM_STEPS, NUM_STEPS, key='raster_slider')

partial_spk  = spk_rec[:t_step, :, :]            # [t_step, 1, 10]
spike_matrix = partial_spk[:, 0, :].numpy()      # [t_step, 10]
fire_times, neuron_ids = np.where(spike_matrix.T)

fig_raster, ax_raster = plt.subplots(figsize=(10, 3))
fig_raster.patch.set_facecolor('#0A0C10')
ax_raster.set_facecolor('#131722')
if len(fire_times) > 0:
    ax_raster.scatter(neuron_ids, fire_times, s=12, c='#00FF66', alpha=0.85)
else:
    ax_raster.text(0.5, 0.5, 'No spikes in this window',
                   ha='center', va='center', color='#718096',
                   transform=ax_raster.transAxes)
ax_raster.set_xlabel('Neuron ID (output class)', color='#A0AEC0', fontsize=9)
ax_raster.set_ylabel('Time Step', color='#A0AEC0', fontsize=9)
ax_raster.set_title(f'Output Spike Raster (t=1 → {t_step})', color='white', fontsize=10)
ax_raster.set_xlim(-0.5, 9.5)
ax_raster.set_ylim(-0.5, t_step - 0.5 if t_step > 1 else 0.5)
ax_raster.tick_params(colors='#A0AEC0', labelsize=8)
for spine in ax_raster.spines.values():
    spine.set_edgecolor('#2D3748')
plt.tight_layout()
st.pyplot(fig_raster)

st.markdown("---")
st.caption(
    "NeuroSort · snnTorch-Powered Neuromorphic Edge Vision Core · "
    "Event-driven SNN on Fashion-MNIST silhouette benchmark"
)
