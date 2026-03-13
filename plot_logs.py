import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# =========================
# COLORS (PAPER STYLE)
# =========================
COLOR_ACC   = "#2F6FA3"   # blue
COLOR_F1    = "#E6863B"   # orange
COLOR_PEAR  = "#3A7D44"   # dark green (Pearson)
COLOR_SPEAR = "#6A5ACD"   # purple/blue (Spearman)

# =========================
# LOAD DATA
# =========================
single = pd.concat([
    pd.read_csv("logs/single_sst2.csv"),
    pd.read_csv("logs/single_qqp.csv"),
    pd.read_csv("logs/single_stsb.csv")
], ignore_index=True)

multi = pd.read_csv("logs/multi_task.csv")

# =========================
# SAFE GET FUNCTION
# =========================
def get_val(df, task, col):
    if col not in df.columns:
        return None
    row = df[df["task"] == task]
    if row.empty:
        return None
    v = row[col].values[0]
    if pd.isna(v):
        return None
    return float(v)

def nz(x):
    return 0.0 if x is None else x

# =========================
# CLASSIFICATION TASKS
# =========================
cls_tasks = ["sst2", "qqp"]

acc_single = [nz(get_val(single,t,"accuracy")) for t in cls_tasks]
acc_multi  = [nz(get_val(multi,t,"accuracy"))  for t in cls_tasks]

f1_single  = [nz(get_val(single,t,"f1")) for t in cls_tasks]
f1_multi   = [nz(get_val(multi,t,"f1"))  for t in cls_tasks]

# =========================
# STS-B (REGRESSION)
# =========================
pear_single = get_val(single,"stsb","pearson")
if pear_single is None:
    pear_single = get_val(single,"stsb","accuracy")

pear_multi = get_val(multi,"stsb","pearson")
if pear_multi is None:
    pear_multi = get_val(multi,"stsb","accuracy")

spear_single = get_val(single,"stsb","spearman")
if spear_single is None:
    spear_single = get_val(single,"stsb","f1")

spear_multi = get_val(multi,"stsb","spearman")
if spear_multi is None:
    spear_multi = get_val(multi,"stsb","f1")

pear_single, pear_multi = nz(pear_single), nz(pear_multi)
spear_single, spear_multi = nz(spear_single), nz(spear_multi)

# =========================
# PLOT
# =========================
labels = ["SST-2","QQP","STS-B"]
x = np.arange(len(labels))
w = 0.18

plt.figure(figsize=(10,6))
bars=[]

# SST-2 & QQP
bars += plt.bar(x[:2]-w*1.5, acc_single, w,
                label="Accuracy (Single)",
                color=COLOR_ACC, alpha=0.65,
                edgecolor="black", linewidth=0.6)

bars += plt.bar(x[:2]-w*0.5, acc_multi, w,
                label="Accuracy (Multi)",
                color=COLOR_ACC,
                edgecolor="black", linewidth=0.6)

bars += plt.bar(x[:2]+w*0.5, f1_single, w,
                label="F1 (Single)",
                color=COLOR_F1, alpha=0.65,
                edgecolor="black", linewidth=0.6)

bars += plt.bar(x[:2]+w*1.5, f1_multi, w,
                label="F1 (Multi)",
                color=COLOR_F1,
                edgecolor="black", linewidth=0.6)

# STS-B
bars += plt.bar(x[2]-w*1.5, pear_single, w,
                label="Pearson (Single)",
                color=COLOR_PEAR, alpha=0.65,
                edgecolor="black", linewidth=0.6)

bars += plt.bar(x[2]-w*0.5, pear_multi, w,
                label="Pearson (Multi)",
                color=COLOR_PEAR,
                edgecolor="black", linewidth=0.6)

bars += plt.bar(x[2]+w*0.5, spear_single, w,
                label="Spearman (Single)",
                color=COLOR_SPEAR, alpha=0.65,
                edgecolor="black", linewidth=0.6)

bars += plt.bar(x[2]+w*1.5, spear_multi, w,
                label="Spearman (Multi)",
                color=COLOR_SPEAR,
                edgecolor="black", linewidth=0.6)

# =========================
# VALUE LABELS
# =========================
for b in bars:
    h = b.get_height()
    if h > 0:
        plt.text(b.get_x()+b.get_width()/2,
                 h+0.01,
                 f"{h*100:.1f}%",
                 ha="center",
                 fontsize=9)

plt.xticks(x,labels)
plt.ylabel("Score")
plt.title("Single-task vs Multi-task Learning Performance")
plt.legend(ncol=2)
plt.grid(axis="y",linestyle="--",alpha=0.4)
plt.tight_layout()

os.makedirs("logs",exist_ok=True)
plt.savefig("logs/final_results.png",dpi=300)
plt.close()

print("✓ Saved: logs/final_results.png")
