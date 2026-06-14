<div align="center">
  
# ⚡ NeuroSort 
**snnTorch-Powered Neuromorphic Edge Vision Core**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://neurosort-1.streamlit.app/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![snnTorch](https://img.shields.io/badge/snnTorch-Neuromorphic-00FF66.svg)](https://snntorch.readthedocs.io/)

[**Live Demo**](https://neurosort-1.streamlit.app/)

</div>

---

## 🧠 The Vision: Why NeuroSort?
Traditional Deep Neural Networks (DNNs) are highly accurate but notoriously power-hungry. When deploying AI to remote, edge-computing environments—such as industrial recycling plants or smart waste bins—battery life and thermal constraints become massive bottlenecks.

**NeuroSort** solves this by utilizing a **Spiking Neural Network (SNN)**. Inspired by the human brain, SNNs communicate via discrete, sparse electrical spikes rather than continuous, dense decimal values. This *event-driven* architecture ensures that power is only consumed when information is actually processed, allowing for **ultra-low-power edge vision classification**.

## ✨ Features
- **True Spiking Architecture:** Uses Leaky Integrate-and-Fire (LIF) neurons simulated over time steps.
- **Extreme Sparsity:** Achieves ~70%+ spike sparsity, translating directly to reduced power consumption (~0.3mW estimated core power) and computational savings vs traditional DNNs.
- **Live Neuromorphic Dashboard:** An interactive, dark-themed UI built to visualize real-time spike rasters, neural population votes, and temporal inference steps.
- **Surrogate Gradient Training:** Overcomes the non-differentiable nature of spikes using fast-sigmoid surrogate gradients in `snnTorch`.

## 🏗️ Architecture & Tech Stack
The model is a Convolutional Spiking Neural Network trained on the **Fashion-MNIST** silhouette benchmark (mapped to simulated material categories).

- **Frameworks:** `PyTorch`, `snnTorch`, `Streamlit`
- **Network Topology:** 
  `Conv2D(1→8) -> MaxPool2D -> snn.Leaky` <br/>
  `Conv2D(8→16) -> MaxPool2D -> snn.Leaky` <br/>
  `Linear(256→10) -> snn.Leaky (Output Vote)`
- **Temporal Dynamics:** 25 time-steps (`T=25`), $\beta=0.9$
- **Loss Function:** MSE Count Loss (Temporal spike accumulation)

## 🚀 Live Demo
Experience the event-driven inference yourself on our live Streamlit Community Cloud deployment:

👉 **[Launch NeuroSort Dashboard](https://neurosort-1.streamlit.app/)**

## 💻 Running Locally

### Prerequisites
- Python 3.9+
- Git

### Installation
1. Clone the repository:
```bash
git clone https://github.com/your-username/NeuroSort.git
cd NeuroSort
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Launch the Streamlit dashboard:
```bash
streamlit run app.py
```

## 🔬 Hackathon Impact
NeuroSort demonstrates that complex computer vision tasks do not require massive GPU clusters at the edge. By shifting the paradigm from dense tensor multiplication to sparse temporal spiking, we pave the way for sustainable, battery-powered AI in the recycling and waste-management industries.

*Note: The repository already includes the fully trained `neurosort_model.pth` weights.*
---
<div align="center">
  <em>🔋 Engineered to push the boundaries of ultra-low-power edge AI 🧠</em>
</div
