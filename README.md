# ⚡ NeuroSort

**snnTorch-Powered Neuromorphic Edge Vision Core**

NeuroSort is an event-driven Spiking Neural Network (SNN) waste sorting system. This repository contains the Streamlit dashboard and the trained model weights.

## Live Demo
The application is deployed on Streamlit Community Cloud.
[Link to live demo] *(Replace this with your Streamlit Cloud URL after deployment)*

## Architecture & Tech Stack
- **Framework:** PyTorch & snnTorch
- **Dataset:** Fashion-MNIST (28×28 grayscale silhouette benchmark)
- **Model:** Convolutional SNN with Leaky integrate-and-fire neurons
- **Dashboard:** Streamlit (Custom UI with Matplotlib Raster Visualizations)

## Running Locally
```bash
pip install -r requirements.txt
streamlit run app.py
