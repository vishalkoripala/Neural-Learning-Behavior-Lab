"""
Neural Learning Behavior Laboratory
A modern interactive laboratory for exploring neural network learning dynamics,
decision boundaries, convergence properties, and hyperparameter sensitivity in 2D.
"""

import streamlit as st
import numpy as np
import pandas as pd
import json
from typing import Dict, Any, Type

from core.datasets import (
    generate_dataset,
    process_uploaded_dataframe,
    AVAILABLE_DATASETS
)
from core.models import (
    PerceptronModel,
    AdalineModel,
    HebbianModel,
    LogisticRegressionModel,
    MLPModel,
    CompetitiveLearningModel,
    BaseNeuralModel,
    AVAILABLE_MODELS
)
from core.visualizer import (
    create_decision_boundary_figure,
    create_learning_curve_figure,
    create_weight_trajectory_figure,
    create_stress_test_figure,
    create_confusion_matrix_figure
)

# -------------------------------------------------------------
# Streamlit Page Config & Custom Styling
# -------------------------------------------------------------
st.set_page_config(
    page_title="Neural Learning Behavior Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Modern CSS
st.markdown("""
<style>
    /* Global Styling */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.95) 0%, rgba(14, 17, 23, 0.95) 100%);
        border: 1px solid rgba(0, 242, 254, 0.2);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 50%, #fe0979 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        color: #a0aec0;
        font-size: 1.05rem;
        margin-bottom: 0px;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(22, 27, 34, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 242, 254, 0.4);
    }
    .metric-title {
        color: #8b949e;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f0f6fc;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-badge {
        display: inline-block;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 20px;
        font-weight: 600;
        margin-top: 4px;
    }
    .badge-cyan { background: rgba(0, 242, 254, 0.15); color: #00f2fe; }
    .badge-pink { background: rgba(254, 9, 121, 0.15); color: #fe0979; }
    .badge-green { background: rgba(0, 255, 135, 0.15); color: #00ff87; }
    .badge-yellow { background: rgba(255, 189, 0, 0.15); color: #ffbd00; }

    /* Theory Block */
    .theory-card {
        background: rgba(22, 27, 34, 0.8);
        border-left: 4px solid #00f2fe;
        border-radius: 0 12px 12px 0;
        padding: 18px 22px;
        margin: 16px 0;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Session State Initialization
# -------------------------------------------------------------
if "experiments" not in st.session_state:
    st.session_state.experiments = []

# -------------------------------------------------------------
# Model Mapping Helper
# -------------------------------------------------------------
MODEL_CLASS_MAP: Dict[str, Type[BaseNeuralModel]] = {
    "Perceptron (Rosenblatt)": PerceptronModel,
    "Adaline (Widrow-Hoff LMS)": AdalineModel,
    "Hebbian Learning (Oja / Classical)": HebbianModel,
    "Logistic Regression (Gradient Descent)": LogisticRegressionModel,
    "Multi-Layer Perceptron (MLP Backprop)": MLPModel,
    "Competitive Learning (Winner-Take-All)": CompetitiveLearningModel
}

# -------------------------------------------------------------
# Sidebar: Controls & Configuration
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Laboratory Controls")
    
    st.markdown("### 1. Dataset Configuration")
    data_source = st.radio("Data Source", ["Synthetic Benchmarks", "Upload CSV Dataset"], horizontal=True)

    if data_source == "Synthetic Benchmarks":
        dataset_name = st.selectbox("Benchmark Distribution", AVAILABLE_DATASETS, index=0)
        n_samples = st.slider("Sample Count", min_value=30, max_value=300, value=120, step=10)
        noise_level = st.slider("Noise / Overlap Level", min_value=0.0, max_value=1.5, value=0.08, step=0.02)
        seed = st.number_input("Random Seed", min_value=0, max_value=9999, value=42, step=1)
        
        X, y = generate_dataset(dataset_name, n_samples=n_samples, noise=noise_level, random_state=seed)
        feature_labels = ["Feature 1", "Feature 2"]
    else:
        uploaded_file = st.file_uploader("Upload CSV (Binary Classification)", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded: `{df.shape[0]}` rows, `{df.shape[1]}` columns")
            target_col = st.selectbox("Select Target Label Column", df.columns)
            feat_mode = st.selectbox("2D Feature Mode", ["First 2 Features", "Select 2 Specific Features", "PCA (All Features)"])
            
            sel_feats = None
            if feat_mode == "Select 2 Specific Features":
                numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
                if len(numeric_cols) >= 2:
                    sel_feats = st.multiselect("Pick 2 Numeric Features", numeric_cols, default=numeric_cols[:2], max_selections=2)
                else:
                    st.error("Need at least 2 numeric features.")
                    st.stop()
            
            X, y, feature_labels, err = process_uploaded_dataframe(
                df,
                target_column=target_col,
                feature_selection=feat_mode,
                selected_features=sel_feats
            )
            if err:
                st.error(err)
                st.stop()
            dataset_name = f"Uploaded CSV ({uploaded_file.name})"
        else:
            st.info("👆 Please upload a CSV dataset or switch to Synthetic Benchmarks.")
            X, y = generate_dataset("Linearly Separable", n_samples=100, noise=0.05, random_state=42)
            dataset_name = "Linearly Separable (Fallback)"
            feature_labels = ["Feature 1", "Feature 2"]

    st.markdown("---")
    st.markdown("### 2. Algorithm & Hyperparameters")
    selected_model_name = st.selectbox("Learning Algorithm", AVAILABLE_MODELS, index=0)
    
    col_lr, col_ep = st.columns(2)
    with col_lr:
        learning_rate = st.number_input("Learning Rate (η)", min_value=0.001, max_value=5.0, value=0.1, step=0.01, format="%.3f")
    with col_ep:
        epochs = st.slider("Epochs", min_value=5, max_value=200, value=40, step=5)

    # Algorithm specific sub-settings
    mlp_hidden = 8
    mlp_activation = "tanh"
    hebbian_mode = "Normalized (Oja)"
    comp_clusters = 2

    if "Multi-Layer" in selected_model_name:
        col_h, col_act = st.columns(2)
        with col_h:
            mlp_hidden = st.slider("Hidden Neurons", min_value=2, max_value=32, value=8, step=2)
        with col_act:
            mlp_activation = st.selectbox("Activation", ["tanh", "relu", "sigmoid"], index=0)
    elif "Hebbian" in selected_model_name:
        hebbian_mode = st.selectbox("Hebbian Variant", ["Normalized (Oja)", "Classical Hebbian"], index=0)
    elif "Competitive" in selected_model_name:
        comp_clusters = st.slider("Cluster Count (K)", min_value=2, max_value=4, value=2)

    st.markdown("---")
    resolution = st.slider("Decision Mesh Resolution", min_value=50, max_value=150, value=90, step=10)

# -------------------------------------------------------------
# Hero Header
# -------------------------------------------------------------
st.markdown(f"""
<div class="hero-container">
    <div class="hero-title">🧠 Neural Learning Behavior Laboratory</div>
    <div class="hero-subtitle">
        Explore convergence, weight trajectories, decision boundary geometry, and hyperparameter sensitivity across classical and modern neural learning rules in 2D.
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Instantiate and Train Active Model
# -------------------------------------------------------------
def get_trained_model(m_name: str, lr: float, eps: int) -> BaseNeuralModel:
    if "Perceptron" in m_name:
        m = PerceptronModel()
        m.fit(X, y, epochs=eps, lr=lr)
    elif "Adaline" in m_name:
        m = AdalineModel()
        m.fit(X, y, epochs=eps, lr=lr)
    elif "Hebbian" in m_name:
        m = HebbianModel(mode=hebbian_mode)
        m.fit(X, y, epochs=eps, lr=lr)
    elif "Logistic" in m_name:
        m = LogisticRegressionModel()
        m.fit(X, y, epochs=eps, lr=lr)
    elif "Multi-Layer" in m_name:
        m = MLPModel(hidden_dim=mlp_hidden, activation=mlp_activation)
        m.fit(X, y, epochs=eps, lr=lr)
    elif "Competitive" in m_name:
        m = CompetitiveLearningModel(n_clusters=comp_clusters)
        m.fit(X, y, epochs=eps, lr=lr)
    else:
        m = PerceptronModel()
        m.fit(X, y, epochs=eps, lr=lr)
    return m

active_model = get_trained_model(selected_model_name, learning_rate, epochs)
summary_metrics = active_model.get_summary_metrics()

# -------------------------------------------------------------
# Main Tabs Navigation
# -------------------------------------------------------------
tab_lab, tab_compare, tab_stress, tab_history, tab_theory = st.tabs([
    "🔬 Live Learning Laboratory",
    "⚖️ Side-by-Side Comparison",
    "⚡ Hyperparameter Stress Test",
    "📊 Experiment Tracker & Weights",
    "📖 Neural Learning Theory"
])

# =============================================================
# TAB 1: Live Learning Laboratory
# =============================================================
with tab_lab:
    # Top Scorecards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Model Rule</div>
            <div class="metric-value" style="font-size:1.15rem; color:#00f2fe;">{active_model.name.split(' ')[0]}</div>
            <span class="metric-badge badge-cyan">Active Engine</span>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        acc_text = f"{summary_metrics['final_accuracy']}%" if summary_metrics['final_accuracy'] != "N/A" else "Unsupervised"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Final Accuracy</div>
            <div class="metric-value" style="color:#00ff87;">{acc_text}</div>
            <span class="metric-badge badge-green">Performance</span>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Final Loss / Metric</div>
            <div class="metric-value" style="color:#fe0979;">{summary_metrics['final_loss']}</div>
            <span class="metric-badge badge-pink">Loss Value</span>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        conv_text = f"Epoch {summary_metrics['converged_at_epoch']}" if summary_metrics['converged_at_epoch'] != "Did Not Converge" else "No (max ep)"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Convergence Status</div>
            <div class="metric-value" style="font-size:1.25rem; color:#ffbd00;">{conv_text}</div>
            <span class="metric-badge badge-yellow">Stability</span>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Loss Variance</div>
            <div class="metric-value" style="font-size:1.25rem; color:#a0aec0;">{summary_metrics['loss_variance']}</div>
            <span class="metric-badge badge-cyan">Dynamics</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Epoch Scrubbing Controls
    col_scrub_label, col_scrub_slider = st.columns([1, 4])
    with col_scrub_label:
        st.markdown("#### ⏱️ Epoch Replay")
        st.caption("Scrub slider to watch the boundary evolve step-by-step:")
    with col_scrub_slider:
        selected_epoch = st.slider("Select Replay Epoch", min_value=1, max_value=epochs, value=epochs, step=1, label_visibility="collapsed")

    # Main Visualizations Grid
    v_col1, v_col2 = st.columns([1.3, 1.0])

    with v_col1:
        fig_boundary = create_decision_boundary_figure(
            active_model,
            X,
            y,
            epoch_idx=selected_epoch - 1,
            grid_resolution=resolution,
            title=f"{active_model.name} — Decision Space (Epoch {selected_epoch}/{epochs})"
        )
        st.plotly_chart(fig_boundary, use_container_width=True)

    with v_col2:
        fig_curves = create_learning_curve_figure(active_model)
        st.plotly_chart(fig_curves, use_container_width=True)

        if active_model.is_supervised and len(y) > 0:
            preds_curr = active_model.predict(X, epoch_idx=selected_epoch - 1)
            fig_cm = create_confusion_matrix_figure(y, preds_curr)
            st.plotly_chart(fig_cm, use_container_width=True)

    # Weight Trajectories Row
    if active_model.weight_history:
        st.markdown("### 📈 Parameter Trajectories & Weight Space Evolution")
        fig_weights = create_weight_trajectory_figure(active_model)
        st.plotly_chart(fig_weights, use_container_width=True)


# =============================================================
# TAB 2: Side-by-Side Model Comparison
# =============================================================
with tab_compare:
    st.markdown("### ⚖️ Multi-Model Simultaneous Comparison")
    st.caption("Evaluate how different neural learning rules behave on the exact same dataset and distribution.")

    comp_models_selected = st.multiselect(
        "Select 2 or 3 Learning Rules to Compare",
        AVAILABLE_MODELS,
        default=["Perceptron (Rosenblatt)", "Logistic Regression (Gradient Descent)", "Multi-Layer Perceptron (MLP Backprop)"],
        max_selections=3
    )

    if len(comp_models_selected) >= 2:
        cols_comp = st.columns(len(comp_models_selected))
        trained_comp_models = []

        for idx, m_name in enumerate(comp_models_selected):
            with cols_comp[idx]:
                m_inst = get_trained_model(m_name, learning_rate, epochs)
                trained_comp_models.append(m_inst)
                m_summary = m_inst.get_summary_metrics()

                st.markdown(f"**{m_name}**")
                st.caption(f"Final Loss: `{m_summary['final_loss']}` | Accuracy: `{m_summary['final_accuracy']}`%")
                fig_comp_b = create_decision_boundary_figure(m_inst, X, y, grid_resolution=resolution, title=m_name)
                st.plotly_chart(fig_comp_b, use_container_width=True)

        # Overlaid Learning Curves Comparison
        st.markdown("#### 📊 Comparative Learning Curves")
        import plotly.graph_objects as go
        fig_overlaid = go.Figure()
        colors_list = ["#00f2fe", "#fe0979", "#00ff87"]

        for idx, m_inst in enumerate(trained_comp_models):
            fig_overlaid.add_trace(go.Scatter(
                x=list(range(1, len(m_inst.loss_history) + 1)),
                y=m_inst.loss_history,
                mode='lines',
                name=m_inst.name,
                line=dict(width=2.5, color=colors_list[idx % len(colors_list)])
            ))

        fig_overlaid.update_layout(
            title="Overlaid Loss / Convergence Trajectories",
            xaxis=dict(title="Epoch", gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(title="Loss Metric", gridcolor="rgba(255,255,255,0.08)"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,20,30,0.7)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=360
        )
        st.plotly_chart(fig_overlaid, use_container_width=True)
    else:
        st.info("Select at least 2 models to compare.")


# =============================================================
# TAB 3: Hyperparameter Sensitivity & Stress Test
# =============================================================
with tab_stress:
    st.markdown("### ⚡ Learning Rate (η) Stability & Stress Sweep")
    st.caption("Demonstrates how too high a learning rate causes divergence / oscillation, while too small causes stagnation.")

    stress_model_name = st.selectbox("Select Learning Rule for Stress Test", AVAILABLE_MODELS, index=0)
    stress_lrs = [0.005, 0.02, 0.1, 0.5, 1.0, 2.0]
    
    custom_lrs = st.multiselect("Selected Learning Rates to Sweep", [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0], default=[0.01, 0.05, 0.1, 0.5, 1.0])

    if custom_lrs:
        fig_stress = create_stress_test_figure(
            MODEL_CLASS_MAP[stress_model_name],
            X,
            y,
            learning_rates=sorted(custom_lrs),
            epochs=epochs
        )
        st.plotly_chart(fig_stress, use_container_width=True)

        st.info(f"""
        **Observations on Learning Rate Dynamics for {stress_model_name}**:
        - **Low Learning Rates (e.g. η ≤ 0.01)**: Smooth and monotonic gradient steps, but slower convergence speed.
        - **Moderate Learning Rates (e.g. 0.05 ≤ η ≤ 0.2)**: Fast, stable convergence without destructive overshooting.
        - **High Learning Rates (e.g. η ≥ 1.0)**: Severe loss oscillations, bouncing across the loss surface, and potential weight divergence.
        """)


# =============================================================
# TAB 4: Experiment Tracker & Weights Exporter
# =============================================================
with tab_history:
    st.markdown("### 📊 Experiment Laboratory & Parameter Downloader")

    c_log, c_clear = st.columns([3, 1])
    with c_log:
        if st.button("📝 Log Current Run to Experiment Tracker", use_container_width=True):
            st.session_state.experiments.append({
                "Dataset": dataset_name,
                "Model": active_model.name,
                "Learning Rate": learning_rate,
                "Epochs": epochs,
                "Final Loss": summary_metrics["final_loss"],
                "Final Accuracy (%)": summary_metrics["final_accuracy"],
                "Converged Epoch": summary_metrics["converged_at_epoch"],
                "Loss Variance": summary_metrics["loss_variance"]
            })
            st.success(f"Logged {active_model.name} run successfully!")
    with c_clear:
        if st.button("🗑️ Clear Experiment History", use_container_width=True):
            st.session_state.experiments = []
            st.rerun()

    if st.session_state.experiments:
        st.markdown("#### 📋 Logged Runs History")
        exp_df = pd.DataFrame(st.session_state.experiments)
        st.dataframe(exp_df, use_container_width=True)

        csv_data = exp_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Experiment History (CSV)",
            data=csv_data,
            file_name="neural_learning_experiments.csv",
            mime="text/csv"
        )

    st.markdown("---")
    st.markdown("#### 💾 Export Trained Model Weights & Architecture")
    weights_json = active_model.export_weights()
    st.json(weights_json)

    st.download_button(
        label="⬇️ Download Model Parameters (JSON)",
        data=json.dumps(weights_json, indent=2),
        file_name=f"{active_model.name.lower().replace(' ', '_')}_weights.json",
        mime="application/json"
    )


# =============================================================
# TAB 5: Neural Learning Theory & Mathematics
# =============================================================
with tab_theory:
    st.markdown("### 📖 Foundations of Neural Learning Rules")
    st.caption("Mathematical foundations, biological inspirations, and algorithmic update mechanics.")

    with st.expander("1. Rosenblatt Perceptron (1958)", expanded=True):
        st.markdown(r"""
        **Biological & Historical Concept**:
        Frank Rosenblatt's Perceptron was one of the earliest models of supervised artificial neurons. It uses a **mistake-driven** discrete threshold update rule.
        
        **Mathematical Formulation**:
        $$\hat{y} = \text{sign}(\mathbf{w}^T \mathbf{x} + b)$$
        $$\mathbf{w} \leftarrow \mathbf{w} + \eta \, y_i \, \mathbf{x}_i, \quad b \leftarrow b + \eta \, y_i \quad \text{iff } y_i(\mathbf{w}^T \mathbf{x}_i + b) \le 0$$
        
        **Convergence Theorem (Novikoff Bound)**:
        If the dataset is linearly separable with geometric margin $\gamma > 0$ and bounded radius $\|\mathbf{x}\| \le R$, the perceptron is guaranteed to converge in at most:
        $$k \le \left(\frac{R}{\gamma}\right)^2 \text{ mistake updates}$$
        *(Limitation: Cannot converge on non-linearly separable data like XOR without non-linear transforms.)*
        """)

    with st.expander("2. Adaline (Adaptive Linear Neuron / Widrow-Hoff LMS, 1960)"):
        st.markdown(r"""
        **Continuous Delta Rule**:
        Unlike the Perceptron which uses step activation before learning, Adaline updates weights based on the continuous linear activation $z = \mathbf{w}^T \mathbf{x} + b$, minimizing Mean Squared Error (MSE).
        
        **Cost Function & Gradient Update**:
        $$J(\mathbf{w}, b) = \frac{1}{2N} \sum_{i=1}^N (y_i - (\mathbf{w}^T \mathbf{x}_i + b))^2$$
        $$\mathbf{w} \leftarrow \mathbf{w} + \frac{\eta}{N} X^T (\mathbf{y} - \hat{\mathbf{z}}), \quad b \leftarrow b + \eta \, \text{mean}(\mathbf{y} - \hat{\mathbf{z}})$$
        """)

    with st.expander("3. Hebbian Learning & Oja's Normalization Rule (1949 / 1982)"):
        st.markdown(r"""
        **Biological Inspiration ("Neurons that fire together, wire together")**:
        Donald Hebb postulated that synaptic efficacy increases proportionally to simultaneous pre-synaptic and post-synaptic activity.
        
        **Classical Hebb Rule**:
        $$\Delta \mathbf{w} = \eta \, y_i \, \mathbf{x}_i$$
        *(Classical Hebbian learning causes weights to grow unbounded without an upper limit.)*
        
        **Oja's Normalized Rule (Principal Component Analyzer)**:
        Erkki Oja added weight decay normalization to guarantee stable convergence to the first principal component eigenvector:
        $$\Delta \mathbf{w} = \eta \, y_i (\mathbf{x}_i - y_i \mathbf{w})$$
        """)

    with st.expander("4. Logistic Regression (Maximum Likelihood & Cross-Entropy)"):
        st.markdown(r"""
        **Probabilistic Classification**:
        Maps linear activations through the logistic Sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$ into calibrated class probabilities $P(y=1|\mathbf{x}) \in (0, 1)$.
        
        **Binary Cross-Entropy Loss**:
        $$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$
        $$\mathbf{w} \leftarrow \mathbf{w} - \frac{\eta}{N} X^T (\hat{\mathbf{p}} - \mathbf{y})$$
        """)

    with st.expander("5. Multi-Layer Perceptron & Error Backpropagation (Rumelhart et al., 1986)"):
        st.markdown(r"""
        **Solving Non-Linear Manifolds (XOR, Moons, Spirals)**:
        By introducing hidden layers with non-linear activation functions (e.g. $\tanh$, $\text{ReLU}$), MLPs learn hierarchical non-linear feature embeddings that can warp non-linearly separable inputs into linearly separable hidden representations.
        
        **Backpropagation Chain Rule**:
        $$\delta^{(2)} = \hat{\mathbf{y}} - \mathbf{y}_{\text{target}}$$
        $$\frac{\partial \mathcal{L}}{\partial W_2} = \mathbf{a}_1^T \delta^{(2)}, \quad \delta^{(1)} = (\delta^{(2)} W_2^T) \odot \sigma'(z_1)$$
        $$\frac{\partial \mathcal{L}}{\partial W_1} = X^T \delta^{(1)}$$
        """)

    with st.expander("6. Competitive Learning (Winner-Take-All Clustering)"):
        st.markdown(r"""
        **Unsupervised Vector Quantization & Self-Organization**:
        In Competitive Learning, output prototype neurons compete for the right to represent input patterns. The winning neuron adjusts its weight vector closer to the input vector.
        
        **Winner Selection & Update**:
        $$k^* = \arg\min_k \|\mathbf{x}_i - \mathbf{w}_k\|$$
        $$\mathbf{w}_{k^*} \leftarrow \mathbf{w}_{k^*} + \eta (\mathbf{x}_i - \mathbf{w}_{k^*})$$
        Creates dynamic **Voronoi partitions** dividing the 2D input space into distinct receptive fields.
        """)

# -------------------------------------------------------------
# Footer
# -------------------------------------------------------------
st.markdown("""
<div style="text-align: center; color: #6e7681; font-size: 0.85rem; margin-top: 50px; padding: 20px;">
    Neural Learning Behavior Laboratory • Built with Streamlit, Plotly & NumPy • Interactive ML Research & Education
</div>
""", unsafe_allow_html=True)
