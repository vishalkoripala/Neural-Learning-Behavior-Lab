import tkinter as tk
from tkinter import ttk, filedialog, simpledialog
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import hebbian
import error_correction
import competitive

X=None
y=None

def clean_dataset(df,label):
    y_raw=df[label]
    X_raw=df.drop(columns=[label])

    for col in X_raw.columns:
        if X_raw[col].nunique()==len(X_raw):
            X_raw=X_raw.drop(columns=[col])

    for col in X_raw.columns:
        if X_raw[col].dtype==object:
            X_raw[col]=LabelEncoder().fit_transform(X_raw[col].astype(str))

    X_raw=X_raw.apply(pd.to_numeric,errors="coerce").fillna(0)

    y_enc=LabelEncoder().fit_transform(y_raw.astype(str))

    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X_raw.values)

    return X_scaled,y_enc

def load_data():
    global X,y
    file=filedialog.askopenfilename(filetypes=[("CSV","*.csv")])
    if not file: return

    df=pd.read_csv(file)

    label=simpledialog.askstring("Label Column",f"Columns:\n{list(df.columns)}")

    if label not in df.columns:
        status.config(text="Invalid label")
        return

    X,y=clean_dataset(df,label)

    preview.delete("1.0",tk.END)
    preview.insert(tk.END,df.head().to_string())

    status.config(text=f"{X.shape[0]} samples | {X.shape[1]} features loaded")
    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.title("Dataset Preview (First 2 Features)")
    print("Dataset shape:", X.shape)
    plt.show()

def run_single():
    if X is None:
        status.config(text="Load dataset first")
        return

    rule = combo.get()
    epochs = epoch_scale.get()

    v = None   # IMPORTANT

    if rule == "Hebbian":
        v = hebbian.run(X, y, epochs)
        desc.config(text="Hebbian – Unsupervised correlation learning")
        formula.config(text="w = w + ηxy")

    elif rule == "Error Correction":
        v = error_correction.run(X, y, epochs)
        desc.config(text="Error Correction – Supervised learning")
        formula.config(text="w = w + η(target − output)x")

    elif rule == "Competitive":
        v = competitive.run(X, epochs)
        desc.config(text="Competitive – Winner-take-all clustering")
        formula.config(text="Winner: w = w + η(x − w)")

    else:
        status.config(text="Please select a rule")
        return

    status.config(text=f"{rule} metric: {v:.3f}")
    save(rule, v)

def run_all():
    def run_all():
        if X is None:
            status.config(text="Load dataset first")
            return

    e = epoch_scale.get()

    h = hebbian.run(X,y,e)
    er = error_correction.run(X,y,e)
    c = competitive.run(X,e)

    summary = f"""
Comparison Summary:
--------------------------
Hebbian Weight Norm     : {h:.3f}
Error Correction Loss   : {er:.3f}
Competitive Separation  : {c:.3f}

Observation:
Hebbian shows correlation strength.
Error Correction shows supervised convergence.
Competitive shows clustering separation.
"""
    if status.winfo_exists():
        status.config(text="All Rules Executed")
    preview.delete("1.0", tk.END)
    preview.insert(tk.END, summary)

    save("ALL", summary)

def save(rule,val):
    with open("results.txt","a") as f:
        f.write(f"{rule}: {val}\n")

def reset():
    global X,y
    X=None; y=None
    preview.delete("1.0",tk.END)
    status.config(text="Reset Done")

root=tk.Tk()
root.title("Interactive Neural Learning Rule Analysis System")
root.geometry("750x600")

tk.Label(root,text="Neural Learning Rule Analyzer",font=("Arial",16,"bold")).pack()

tk.Button(root,text="Load Dataset",command=load_data).pack()

combo=ttk.Combobox(root,values=["Hebbian","Error Correction","Competitive"])
combo.pack()

epoch_scale=tk.Scale(root,from_=10,to=200,orient="horizontal",label="Epochs")
epoch_scale.set(50)
epoch_scale.pack()

tk.Button(root,text="Run Selected Rule",command=run_single).pack(pady=5)
tk.Button(root,text="Run ALL Rules",command=run_all).pack(pady=5)
tk.Button(root,text="Reset",command=reset).pack()

desc=tk.Label(root,text="",fg="green")
desc.pack()
formula = tk.Label(root, text="", fg="purple", font=("Arial",10,"italic"))
formula.pack()

status=tk.Label(root,text="Waiting",fg="blue")
status.pack()

preview=tk.Text(root,height=8,width=90)
preview.pack()

root.mainloop()