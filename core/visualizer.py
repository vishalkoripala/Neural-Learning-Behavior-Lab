"""
Interactive Visualizations using Plotly for the Neural Learning Behavior Laboratory
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Dict, Any
from .models import BaseNeuralModel, CompetitiveLearningModel


# Sleek, harmonious color palettes
COLORS = {
    "pos_class": "#00f2fe",      # Neon Cyan
    "neg_class": "#fe0979",      # Vibrant Pink/Magenta
    "misclassified": "#ffbd00",  # Amber Yellow
    "boundary": "#ffffff",       # Crisp White
    "prototypes": ["#00f2fe", "#fe0979", "#00ff87"],
    "bg_dark": "#0e1117",
    "card_dark": "#161b22",
    "grid": "rgba(255, 255, 255, 0.08)"
}


def create_decision_boundary_figure(
    model: BaseNeuralModel,
    X: np.ndarray,
    y: np.ndarray,
    epoch_idx: Optional[int] = None,
    grid_resolution: int = 100,
    title: Optional[str] = None
) -> go.Figure:
    """
    Generate an interactive 2D decision boundary plot with contours, scatter points, and misclassifications.
    """
    fig = go.Figure()

    # Determine 2D bounds with margin
    x_min, x_max = X[:, 0].min() - 0.7, X[:, 0].max() + 0.7
    y_min, y_max = X[:, 1].min() - 0.7, X[:, 1].max() + 0.7

    xx = np.linspace(x_min, x_max, grid_resolution)
    yy = np.linspace(y_min, y_max, grid_resolution)
    grid_x, grid_y = np.meshgrid(xx, yy)
    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]

    # Predict raw activations/probabilities on grid
    z_raw = model.predict_raw(grid_points, epoch_idx=epoch_idx)
    z_grid = z_raw.reshape(grid_x.shape)

    # 1. Background Contour Heatmap
    if isinstance(model, CompetitiveLearningModel):
        # Cluster zone distance difference
        colorscale = [[0.0, "rgba(254, 9, 121, 0.25)"], [0.5, "rgba(30, 35, 45, 0.1)"], [1.0, "rgba(0, 242, 254, 0.25)"]]
    elif "Multi-Layer" in model.name or "Logistic" in model.name:
        # Probability contours [0, 1]
        colorscale = [[0.0, "rgba(254, 9, 121, 0.25)"], [0.5, "rgba(30, 35, 45, 0.1)"], [1.0, "rgba(0, 242, 254, 0.25)"]]
    else:
        # Linear margins
        colorscale = [[0.0, "rgba(254, 9, 121, 0.25)"], [0.5, "rgba(30, 35, 45, 0.1)"], [1.0, "rgba(0, 242, 254, 0.25)"]]

    fig.add_trace(go.Contour(
        x=xx,
        y=yy,
        z=z_grid,
        showscale=False,
        colorscale=colorscale,
        contours_coloring='fill',
        line=dict(width=0),
        hoverinfo='none',
        name="Decision Region"
    ))

    # 2. Decision Boundary Contour Line (Z = 0 or P = 0.5)
    boundary_threshold = 0.5 if ("Multi-Layer" in model.name or "Logistic" in model.name) else 0.0
    fig.add_trace(go.Contour(
        x=xx,
        y=yy,
        z=z_grid,
        showscale=False,
        contours=dict(
            start=boundary_threshold,
            end=boundary_threshold,
            size=1,
            coloring='none'
        ),
        line=dict(color="#ffffff", width=2.5, dash="solid"),
        hoverinfo='none',
        name="Decision Boundary"
    ))

    # 3. Data Scatter Points
    pos_mask = (y == 1)
    neg_mask = (y == -1)

    fig.add_trace(go.Scatter(
        x=X[pos_mask, 0],
        y=X[pos_mask, 1],
        mode='markers',
        marker=dict(
            size=11,
            color=COLORS["pos_class"],
            line=dict(width=1.5, color='#ffffff'),
            symbol='circle'
        ),
        name='Class +1',
        hovertemplate='<b>Class +1</b><br>X1: %{x:.2f}<br>X2: %{y:.2f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=X[neg_mask, 0],
        y=X[neg_mask, 1],
        mode='markers',
        marker=dict(
            size=11,
            color=COLORS["neg_class"],
            line=dict(width=1.5, color='#ffffff'),
            symbol='diamond'
        ),
        name='Class -1',
        hovertemplate='<b>Class -1</b><br>X1: %{x:.2f}<br>X2: %{y:.2f}<extra></extra>'
    ))

    # 4. Highlight Misclassified Points (Supervised models)
    if model.is_supervised:
        preds = model.predict(X, epoch_idx=epoch_idx)
        mis_mask = (preds != y)
        if np.any(mis_mask):
            fig.add_trace(go.Scatter(
                x=X[mis_mask, 0],
                y=X[mis_mask, 1],
                mode='markers',
                marker=dict(
                    size=16,
                    color='rgba(0,0,0,0)',
                    line=dict(width=3, color=COLORS["misclassified"]),
                    symbol='x'
                ),
                name=f'Misclassified ({np.sum(mis_mask)})',
                hovertemplate='<b>Misclassified Point</b><br>X1: %{x:.2f}<br>X2: %{y:.2f}<extra></extra>'
            ))

    # 5. Prototypes for Competitive Learning
    if isinstance(model, CompetitiveLearningModel):
        prototypes = model._get_prototypes(epoch_idx=epoch_idx)
        fig.add_trace(go.Scatter(
            x=prototypes[:, 0],
            y=prototypes[:, 1],
            mode='markers+text',
            marker=dict(
                size=22,
                color='#ffe600',
                line=dict(width=3, color='#000000'),
                symbol='star'
            ),
            text=[f"Cluster {i+1}" for i in range(len(prototypes))],
            textposition="top center",
            textfont=dict(color="#ffffff", size=11, family="sans-serif"),
            name='Prototype Centroids'
        ))

    ep_label = f" (Epoch {epoch_idx + 1}/{len(model.loss_history)})" if epoch_idx is not None and model.loss_history else ""
    fig_title = title or f"{model.name} — 2D Decision Space{ep_label}"

    fig.update_layout(
        title=dict(text=fig_title, font=dict(size=15, color="#f0f2f6")),
        xaxis=dict(
            title="Feature 1 (Normalized)",
            gridcolor=COLORS["grid"],
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.15)",
            range=[x_min, x_max]
        ),
        yaxis=dict(
            title="Feature 2 (Normalized)",
            gridcolor=COLORS["grid"],
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.15)",
            range=[y_min, y_max],
            scaleanchor="x",
            scaleratio=1
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,20,30,0.7)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10, color="#ffffff")
        ),
        margin=dict(l=40, r=40, t=50, b=40),
        height=480
    )

    return fig


def create_learning_curve_figure(model: BaseNeuralModel) -> go.Figure:
    """
    Plot interactive Loss and Accuracy evolution across training epochs.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    epochs = list(range(1, len(model.loss_history) + 1))

    # Loss curve
    loss_name = "Reconstruction Loss" if isinstance(model, CompetitiveLearningModel) else ("Weight Norm" if not model.is_supervised else "Loss / Misclassifications")
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=model.loss_history,
            mode='lines+markers',
            name=loss_name,
            line=dict(color="#fe0979", width=2.5),
            marker=dict(size=5),
            hovertemplate='Epoch %{x}: Loss = %{y:.4f}<extra></extra>'
        ),
        secondary_y=False
    )

    # Accuracy curve (if available)
    if model.accuracy_history:
        acc_label = "Prototype Distance" if isinstance(model, CompetitiveLearningModel) else "Classification Accuracy (%)"
        acc_vals = [a * 100 if model.is_supervised else a for a in model.accuracy_history]
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=acc_vals,
                mode='lines+markers',
                name=acc_label,
                line=dict(color="#00f2fe", width=2.5, dash='dash'),
                marker=dict(size=5),
                hovertemplate='Epoch %{x}: ' + ('Acc = %{y:.2f}%' if model.is_supervised else 'Dist = %{y:.3f}') + '<extra></extra>'
            ),
            secondary_y=True
        )

    fig.update_layout(
        title=dict(text=f"{model.name} — Learning Convergence Dynamics", font=dict(size=14, color="#f0f2f6")),
        xaxis=dict(title="Training Epoch", gridcolor=COLORS["grid"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,20,30,0.7)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10, color="#ffffff")),
        margin=dict(l=40, r=40, t=50, b=40),
        height=340
    )

    fig.update_yaxes(title_text=loss_name, gridcolor=COLORS["grid"], secondary_y=False)
    if model.accuracy_history:
        fig.update_yaxes(title_text=acc_label, gridcolor="rgba(0,0,0,0)", secondary_y=True)

    return fig


def create_weight_trajectory_figure(model: BaseNeuralModel) -> go.Figure:
    """
    Plot evolution of individual weights/parameters over training epochs.
    """
    fig = go.Figure()
    if not model.weight_history:
        return fig

    w_arr = np.array(model.weight_history)
    epochs = list(range(1, len(w_arr) + 1))
    colors = ["#00f2fe", "#fe0979", "#ffbd00", "#00ff87", "#9d4edd", "#ff7b00", "#70e000", "#38b000"]

    for i in range(min(w_arr.shape[1], 8)):
        fig.add_trace(go.Scatter(
            x=epochs,
            y=w_arr[:, i],
            mode='lines',
            name=f'Weight w{i+1}',
            line=dict(width=2, color=colors[i % len(colors)]),
            hovertemplate=f'w{i+1}: %{{y:.4f}}<extra></extra>'
        ))

    if model.bias_history and any(b != 0 for b in model.bias_history):
        fig.add_trace(go.Scatter(
            x=epochs,
            y=model.bias_history,
            mode='lines',
            name='Bias (b)',
            line=dict(width=2, color='#ffffff', dash='dot'),
            hovertemplate='Bias: %{y:.4f}<extra></extra>'
        ))

    fig.update_layout(
        title=dict(text=f"{model.name} — Parameter Trajectories Over Time", font=dict(size=14, color="#f0f2f6")),
        xaxis=dict(title="Training Epoch", gridcolor=COLORS["grid"]),
        yaxis=dict(title="Weight Value", gridcolor=COLORS["grid"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,20,30,0.7)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10, color="#ffffff")),
        margin=dict(l=40, r=40, t=50, b=40),
        height=340
    )

    return fig


def create_stress_test_figure(
    model_class: type,
    X: np.ndarray,
    y: np.ndarray,
    learning_rates: List[float],
    epochs: int = 40
) -> go.Figure:
    """
    Run multi-learning rate sweep to visualize stability, oscillation, and divergence.
    """
    fig = go.Figure()
    palette = ["#00f2fe", "#00ff87", "#ffbd00", "#ff7b00", "#fe0979", "#9d4edd"]

    for idx, lr in enumerate(learning_rates):
        m = model_class()
        m.fit(X, y, epochs=epochs, lr=lr)
        color = palette[idx % len(palette)]
        fig.add_trace(go.Scatter(
            x=list(range(1, epochs + 1)),
            y=m.loss_history,
            mode='lines',
            name=f'η = {lr}',
            line=dict(color=color, width=2.2),
            hovertemplate=f'η={lr} | Epoch %{{x}}: Loss=%{{y:.4f}}<extra></extra>'
        ))

    fig.update_layout(
        title=dict(text="Learning Rate Sensitivity & Stress Dynamics", font=dict(size=15, color="#f0f2f6")),
        xaxis=dict(title="Epoch", gridcolor=COLORS["grid"]),
        yaxis=dict(title="Loss Metric", gridcolor=COLORS["grid"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,20,30,0.7)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11, color="#ffffff")),
        margin=dict(l=40, r=40, t=50, b=40),
        height=400
    )

    return fig


def create_confusion_matrix_figure(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """
    Generate an annotated Confusion Matrix heatmap.
    """
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])

    labels = ["Class -1", "Class +1"]
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale=[[0.0, "#161b22"], [1.0, "#00f2fe"]],
        text=cm,
        texttemplate="<b>%{text}</b>",
        textfont={"size": 18, "color": "white"},
        hoverinfo="none",
        showscale=False
    ))

    fig.update_layout(
        title=dict(text="Confusion Matrix", font=dict(size=14, color="#f0f2f6")),
        xaxis=dict(title="Predicted Label", side="bottom"),
        yaxis=dict(title="Actual Label", autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=40, b=40),
        height=260,
        width=300
    )

    return fig
