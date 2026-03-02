import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
st.set_page_config(layout="wide")
st.markdown("""
### Neural Learning Behavior Laboratory
Explore convergence, stability, hyperparameter sensitivity and nonlinear separability.
""")

def plot_decision_surface(ax, predict_func, X):
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_func(grid)
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z > 0, alpha=0.2)

st.set_page_config(layout="wide")


# -------------------------
# Session State
# -------------------------
if "experiments" not in st.session_state:
    st.session_state.experiments = []

# -------------------------
# Controls
# -------------------------
dataset_type = st.selectbox(
    "Dataset Type",
    ["Linearly Separable", "XOR"]
)

model_type = st.selectbox(
    "Learning Rule",
    ["Perceptron", "Hebbian", "Logistic", "MLP (1 Hidden Layer)", "Competitive"]
)

lr = st.slider("Learning Rate", 0.01, 1.0, 0.1)
epochs = st.slider("Epochs", 5, 100, 30)
noise = st.slider("Noise Level", 0.0, 2.0, 0.0)
# -------------------------
# Upload Dataset Section
# -------------------------

# -------------------------
# Upload Dataset Section
# -------------------------


st.subheader("Upload Your Dataset (Optional CSV)")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview")
    st.dataframe(df.head())

    target_column = st.selectbox("Select Target Column", df.columns)

    if df[target_column].dtype != object and df[target_column].nunique() > 20:
        st.warning("Selected target looks numeric with many unique values. This may not be a classification label.")

    # Separate X and y
    X_full = df.drop(columns=[target_column])
    y_full = df[target_column]

    # Keep only numeric features
    numeric_cols = X_full.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) < 2:
        st.error("Dataset must contain at least 2 numeric feature columns for 2D visualization.")
        st.stop()

    X_full = X_full[numeric_cols]
    st.info(f"Using numeric features: {list(numeric_cols)}")

    # Encode target if categorical
    if y_full.dtype == object:
        y_full = pd.factorize(y_full)[0]

    unique_classes = np.unique(y_full)

    if len(unique_classes) > 2:
        st.warning("More than 2 classes detected.")
        selected_classes = st.multiselect(
            "Select exactly 2 classes",
            unique_classes,
            max_selections=2
        )

        if len(selected_classes) < 2:
            st.info("Select 2 classes to continue.")
            st.stop()

        mask = np.isin(y_full, selected_classes)
        X_full = X_full[mask]
        y_full = y_full[mask]

    unique_classes = np.unique(y_full)

    if len(unique_classes) < 2:
        st.error("Target must contain at least 2 classes.")
        st.stop()

    y = np.where(y_full == unique_classes[0], -1, 1)

    # Select 2 features for visualization
    if X_full.shape[1] > 2:
        feature_cols = st.multiselect(
            "Select exactly 2 numeric features",
            X_full.columns,
            max_selections=2
        )

        if len(feature_cols) < 2:
            st.info("Select 2 features to continue.")
            st.stop()

        X = X_full[feature_cols].values
    else:
        X = X_full.values

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
# -------------------------
# Dataset Generation
# -------------------------
# -------------------------
# Dataset Generation (Fallback if no upload)
# -------------------------

# -------------------------
# Dataset Generation (Fallback)
# -------------------------

if uploaded_file is None:

    np.random.seed(0)

    if dataset_type == "Linearly Separable":
        class1 = np.random.randn(50, 2) + np.array([2, 2])
        class2 = np.random.randn(50, 2) + np.array([-2, -2])
        X = np.vstack((class1, class2))
        y = np.hstack((np.ones(50), -np.ones(50)))
        X += np.random.randn(*X.shape) * noise

    else:
        X = np.array([[0,0],[0,1],[1,0],[1,1]])
        y = np.array([-1,1,1,-1])
        st.warning("XOR is not linearly separable.")
# -------------------------
# Model Training
# -------------------------
w = np.zeros(X.shape[1])
b = 0
loss_history = []

if model_type == "Hebbian":
    for epoch in range(epochs):
        for i in range(len(X)):
            w += lr * y[i] * X[i]
        loss_history.append(np.linalg.norm(w))

elif model_type == "Perceptron":
    for epoch in range(epochs):
        errors = 0
        for i in range(len(X)):
            if y[i] * (np.dot(w, X[i]) + b) <= 0:
                w += lr * y[i] * X[i]
                b += lr * y[i]
                errors += 1
        loss_history.append(errors)

elif model_type == "Logistic":
    y_log = (y == 1).astype(int)
    for epoch in range(epochs):
        preds = 1 / (1 + np.exp(-(X @ w + b)))
        gradient_w = X.T @ (preds - y_log)
        gradient_b = np.sum(preds - y_log)
        w -= lr * gradient_w
        b -= lr * gradient_b
        loss = np.mean(-(y_log*np.log(preds+1e-9) + (1-y_log)*np.log(1-preds+1e-9)))
        loss_history.append(loss)

elif model_type == "MLP (1 Hidden Layer)":
    hidden_size = 4
    W1 = np.random.randn(2, hidden_size)
    b1 = np.zeros(hidden_size)
    W2 = np.random.randn(hidden_size, 1)
    b2 = 0

    y_mlp = (y == 1).astype(int).reshape(-1,1)

    for epoch in range(epochs):
        z1 = X @ W1 + b1
        a1 = np.tanh(z1)
        z2 = a1 @ W2 + b2
        preds = 1 / (1 + np.exp(-z2))

        dz2 = preds - y_mlp
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2)

        dz1 = dz2 @ W2.T * (1 - np.tanh(z1)**2)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

        loss = np.mean(-(y_mlp*np.log(preds+1e-9) + (1-y_mlp)*np.log(1-preds+1e-9)))
        loss_history.append(loss)

elif model_type == "Competitive":
    w1 = np.random.randn(2)
    w2 = np.random.randn(2)
    for epoch in range(epochs):
        for x in X:
            if np.linalg.norm(x-w1) < np.linalg.norm(x-w2):
                w1 += lr * (x-w1)
            else:
                w2 += lr * (x-w2)
    loss_history.append(np.linalg.norm(w1-w2))
if model_type in ["Perceptron", "Logistic"]:
    def predict(grid):
        return grid @ w + b
    

# -------------------------
# Visualization
# -------------------------
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()

    # Plot data generically
    class_pos = X[y == 1]
    class_neg = X[y == -1]

    ax.scatter(class_pos[:, 0], class_pos[:, 1], label="Class 1")
    ax.scatter(class_neg[:, 0], class_neg[:, 1], label="Class -1")


    # Decision surface
    if model_type in ["Perceptron", "Logistic"]:
        def predict(grid):
            return grid @ w + b

        plot_decision_surface(ax, predict, X)

    ax.legend()
    ax.set_title("Decision Visualization")
    st.pyplot(fig)
stress_test = st.checkbox("Run Learning Rate Stress Test")
compare_mode = st.checkbox("Enable Side-by-Side Model Comparison")
if stress_test:
    test_lrs = [0.01, 0.1, 0.5, 1.0]
    fig_stress, ax_stress = plt.subplots()

    for test_lr in test_lrs:
        temp_w = np.zeros(X.shape[1])
        temp_b = 0
        temp_loss = []

        for epoch in range(epochs):
            errors = 0
            for i in range(len(X)):
                if y[i]*(np.dot(temp_w,X[i])+temp_b)<=0:
                    temp_w += test_lr*y[i]*X[i]
                    temp_b += test_lr*y[i]
                    errors+=1
            temp_loss.append(errors)

        ax_stress.plot(temp_loss, label=f"LR={test_lr}")

    ax_stress.legend()
    ax_stress.set_title("Learning Rate Stress Test")
    st.pyplot(fig_stress)
if model_type == "Hebbian":
    st.info("Hebbian Learning: Unsupervised correlation-based weight update.")
elif model_type == "Perceptron":
    st.info("Perceptron: Linear classifier using mistake-driven updates.")
elif model_type == "Logistic":
    st.info("Logistic: Gradient descent on cross-entropy loss.")
elif model_type == "MLP (1 Hidden Layer)":
    st.info("MLP: Nonlinear feature transformation + backpropagation.")
elif model_type == "Competitive":
    st.info("Competitive Learning: Winner-take-all unsupervised clustering.")
with col2:
    fig2, ax2 = plt.subplots()
    ax2.plot(loss_history)
    ax2.set_title("Learning Curve")
    ax2.set_xlabel("Epoch")
    st.pyplot(fig2)

# -------------------------
# Metrics
# -------------------------
st.subheader("Final Observation")
st.write("Convergence Epoch:", 
         next((i for i, v in enumerate(loss_history) if v == 0), "Did not converge"))
st.write("Loss Variance:", np.var(loss_history))
if len(loss_history) > 0:
    st.write("Final Metric:", float(loss_history[-1]))
if len(loss_history) > 0:
    threshold = 0.01
    convergence = next((i for i,v in enumerate(loss_history) if v < threshold), None)
    st.write("Convergence Epoch:", convergence if convergence is not None else "Did Not Converge")
    st.write("Stability Score (Loss Variance):", np.var(loss_history))
# -------------------------
# Experiment Logging
# -------------------------
if st.button("Run Experiment & Log Results", key="log_btn"):
    if len(loss_history) > 0:
        st.session_state.experiments.append({
            "Dataset": dataset_type,
            "Model": model_type,
            "LR": lr,
            "Epochs": epochs,
            "Final Metric": float(loss_history[-1])
        })
if st.button("Download Model Weights"):
    st.json({"weights": w.tolist()})
if st.session_state.experiments:
    st.subheader("Experiment Comparison Table")
    df = pd.DataFrame(st.session_state.experiments)
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Results as CSV",
        csv,
        "experiment_results.csv",
        "text/csv",
        key="download_csv"
    )
predictions = np.sign(X @ w + b)
accuracy = np.mean(predictions == y)
st.write("Final Accuracy:", accuracy)
