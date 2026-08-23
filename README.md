# 🧠 Neural Learning Behavior Laboratory

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.13%20%7C%203.14-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75.svg)](https://plotly.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced, interactive visual laboratory for studying and comparing classical and modern neural network learning rules in 2D space. Built with **Streamlit**, **Plotly**, and **NumPy**.

👉 **Live Demo:** [https://neural-learning-behavior-lab.streamlit.app/](https://neural-learning-behavior-lab.streamlit.app/)

---

## 📌 Overview

The **Neural Learning Behavior Laboratory** provides deep conceptual and visual clarity on how neural learning algorithms adapt, update weights, construct decision boundaries, and converge over time.

### 🌟 Key Highlights & Capabilities
- **⏱️ Dynamic Epoch Replay**: Scrub through training history epoch-by-epoch to watch decision surfaces rotate and weights adapt in real-time.
- **🔬 Interactive 2D Decision Surfaces**: High-resolution Plotly contour heatmaps, probability gradients, decision boundaries, and misclassified sample markers.
- **⚖️ Side-by-Side Model Comparison**: Compare any 2 or 3 learning rules simultaneously on the exact same dataset distribution and split.
- **⚡ Hyperparameter Sensitivity & Stress Testing**: Multi-learning rate sweeps ($\eta \in [0.001, 2.0]$) demonstrating convergence, stagnation, oscillations, and divergence.
- **📈 Parameter Trajectory Visualization**: Track weight vectors ($w_1, w_2, \dots$) and bias ($b$) evolution paths across training epochs.
- **🧪 Rich Benchmark Datasets**: Gaussian Blobs, XOR Problem, Two Moons, Concentric Circles, Spirals, Anisotropic Blobs, and custom CSV dataset upload with automated preprocessing & PCA.
- **💾 Weight & Experiment Exporter**: Download trained model parameters as JSON and logged experiment comparisons as CSV.
- **📖 Neural Learning Theory Guide**: In-app mathematical formulation with LaTeX equations, update mechanics, and biological analogies.

---

## 🧠 Supported Learning Rules

| Algorithm | Type | Activation / Update Condition | Objective / Loss Function |
| :--- | :--- | :--- | :--- |
| **Rosenblatt Perceptron** | Supervised | Step activation $\text{sign}(\mathbf{w}^T \mathbf{x} + b)$ | Mistake-driven discrete hinge |
| **Adaline (Widrow-Hoff LMS)** | Supervised | Linear continuous activation $z = \mathbf{w}^T \mathbf{x} + b$ | Mean Squared Error (MSE) |
| **Hebbian Learning (Oja / Classic)** | Unsupervised | Correlation $\Delta \mathbf{w} = \eta y_i (\mathbf{x}_i - y_i \mathbf{w})$ | Correlation alignment & PCA |
| **Logistic Regression** | Supervised | Logistic Sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$ | Binary Cross-Entropy Loss |
| **Multi-Layer Perceptron (MLP)** | Supervised | Hidden: Tanh / ReLU / Sigmoid + Backprop | Non-linear Binary Cross-Entropy |
| **Competitive Learning (WTA)** | Unsupervised | Winner-Take-All Euclidean distance | Minimum Quantization Error (Voronoi) |

---

## 🧪 Benchmark Datasets

1. **Linearly Separable**: Two Gaussian clusters with configurable margin and noise.
2. **XOR Problem**: Canonical non-linearly separable benchmark.
3. **Two Moons**: Interlocking non-linear crescent manifolds.
4. **Concentric Circles**: Radial non-linear distribution.
5. **Two Spirals**: High-curvature non-linear spiral manifold.
6. **Anisotropic Blobs**: Skewed, elongated Gaussian clusters.
7. **Custom CSV Upload**: Automatic numeric detection, label encoding, PCA dimensionality reduction, and standard scaling.

---

## 📁 Repository Structure

```
Neural-Learning-Behavior-Lab/
├── app.py                  # Main Streamlit web application & UI
├── requirements.txt        # Python dependency specifications
├── .gitignore              # Git ignore rules
├── README.md               # Project documentation
├── core/                   # Vectorized neural learning engine
│   ├── __init__.py         # Package initialization
│   ├── models.py           # Object-oriented neural learning models
│   ├── datasets.py         # Synthetic benchmark generators & CSV processor
│   └── visualizer.py       # Plotly decision surface & metric visualizers
└── tests/                  # Automated test suite
    └── test_models.py      # Unit tests for datasets & learning algorithms
```

---

## ⚙️ Quickstart Guide

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/vishalkoripala/Neural-Learning-Behavior-Lab.git
cd Neural-Learning-Behavior-Lab
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application
```bash
streamlit run app.py
```

The application will launch in your default web browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests

Run the automated test suite to verify algorithm convergence and dataset generation:
```bash
python -m unittest discover tests
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
