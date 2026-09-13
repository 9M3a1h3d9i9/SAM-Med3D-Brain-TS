"""
Compare evaluation results between two models.
Usage: python compare_results.py results_A.csv results_B.csv label_A label_B output_prefix
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------------- CONFIG ----------------
CSV_A = sys.argv[1] if len(sys.argv) > 1 else "results_pretrained.csv"
CSV_B = sys.argv[2] if len(sys.argv) > 2 else "results_finetuned.csv"
LABEL_A = sys.argv[3] if len(sys.argv) > 3 else "Pretrained"
LABEL_B = sys.argv[4] if len(sys.argv) > 4 else "Fine-tuned (Smoke)"
OUT_PREFIX = sys.argv[5] if len(sys.argv) > 5 else "comparison"

OUT_TABLE = f"{OUT_PREFIX}_table.csv"
OUT_BAR = f"{OUT_PREFIX}_bar.png"
OUT_BOX = f"{OUT_PREFIX}_box.png"
# ----------------------------------------

def load_csv(path):
    df = pd.read_csv(path)
    # Ensure dice is numeric
    df["dice"] = pd.to_numeric(df["dice"], errors="coerce")
    return df

def summarize(df, label):
    """Group by ROI and compute mean, std, count of dice."""
    grp = df.groupby("roi")["dice"].agg(["mean", "std", "count"]).reset_index()
    grp["model"] = label
    return grp

def build_table(sumA, sumB):
    """Build comparison table with delta."""
    rows = []
    rois = sorted(set(sumA["roi"]) | set(sumB["roi"]))
    for roi in rois:
        a = sumA[sumA["roi"] == roi]
        b = sumB[sumB["roi"] == roi]
        mean_a = a["mean"].values[0] if len(a) else np.nan
        std_a = a["std"].values[0] if len(a) else np.nan
        mean_b = b["mean"].values[0] if len(b) else np.nan
        std_b = b["std"].values[0] if len(b) else np.nan
        delta = mean_b - mean_a if not (np.isnan(mean_a) or np.isnan(mean_b)) else np.nan
        rows.append({
            "ROI": roi,
            f"{LABEL_A}_mean": round(mean_a, 4) if not np.isnan(mean_a) else "N/A",
            f"{LABEL_A}_std": round(std_a, 4) if not np.isnan(std_a) else "N/A",
            f"{LABEL_B}_mean": round(mean_b, 4) if not np.isnan(mean_b) else "N/A",
            f"{LABEL_B}_std": round(std_b, 4) if not np.isnan(std_b) else "N/A",
            "Delta": round(delta, 4) if not np.isnan(delta) else "N/A",
        })
    return pd.DataFrame(rows)

def plot_bar(df, out_path):
    rois = sorted(set(df["roi"]))
    x = np.arange(len(rois))
    width = 0.35
    means_a = [df[(df["model"] == LABEL_A) & (df["roi"] == r)]["mean"].values[0]
               if len(df[(df["model"] == LABEL_A) & (df["roi"] == r)]) else np.nan
               for r in rois]
    stds_a = [df[(df["model"] == LABEL_A) & (df["roi"] == r)]["std"].values[0]
              if len(df[(df["model"] == LABEL_A) & (df["roi"] == r)]) else 0.0
              for r in rois]
    means_b = [df[(df["model"] == LABEL_B) & (df["roi"] == r)]["mean"].values[0]
               if len(df[(df["model"] == LABEL_B) & (df["roi"] == r)]) else np.nan
               for r in rois]
    stds_b = [df[(df["model"] == LABEL_B) & (df["roi"] == r)]["std"].values[0]
              if len(df[(df["model"] == LABEL_B) & (df["roi"] == r)]) else 0.0
              for r in rois]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, means_a, width, yerr=stds_a, label=LABEL_A, capsize=4)
    ax.bar(x + width/2, means_b, width, yerr=stds_b, label=LABEL_B, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(rois, rotation=15)
    ax.set_ylabel("Dice Score")
    ax.set_title("Dice Comparison per ROI")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved bar plot → {out_path}")

def plot_box(dfA, dfB, out_path):
    rois = sorted(set(dfA["roi"]) | set(dfB["roi"]))
    data = []
    labels = []
    for r in rois:
        a_vals = dfA[dfA["roi"] == r]["dice"].dropna().values
        b_vals = dfB[dfB["roi"] == r]["dice"].dropna().values
        if len(a_vals):
            data.append(a_vals); labels.append(f"{r}\n({LABEL_A})")
        if len(b_vals):
            data.append(b_vals); labels.append(f"{r}\n({LABEL_B})")

    if not data:
        print("No data for box plot. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_ylabel("Dice Score")
    ax.set_title("Dice Distribution per ROI and Model")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved box plot → {out_path}")

def wilcoxon_test(dfA, dfB):
    """Pair-wise Wilcoxon per ROI if at least 5 samples."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        print("scipy not installed. Skipping Wilcoxon.")
        return
    rois = sorted(set(dfA["roi"]) | set(dfB["roi"]))
    print("\n=== Wilcoxon signed-rank test (per ROI) ===")
    for r in rois:
        a = dfA[dfA["roi"] == r].set_index("case")["dice"].dropna()
        b = dfB[dfB["roi"] == r].set_index("case")["dice"].dropna()
        common = a.index.intersection(b.index)
        if len(common) < 5:
            print(f"  {r}: skipped (only {len(common)} pairs, need ≥5)")
            continue
        try:
            stat, p = wilcoxon(a.loc[common], b.loc[common])
            sig = "✅ significant" if p < 0.05 else "❌ not significant"
            print(f"  {r}: stat={stat:.3f}, p={p:.4f}  {sig}")
        except Exception as e:
            print(f"  {r}: error: {e}")

def main():
    dfA = load_csv(CSV_A)
    dfB = load_csv(CSV_B)
    sumA = summarize(dfA, LABEL_A)
    sumB = summarize(dfB, LABEL_B)

    # Build merged summary for plotting
    summary = pd.concat([sumA, sumB], ignore_index=True)

    print("=" * 60)
    print(f"Model A: {LABEL_A}  ({CSV_A})")
    print(f"Model B: {LABEL_B}  ({CSV_B})")
    print("=" * 60)

    # Build table
    table = build_table(sumA, sumB)
    print("\n=== Comparison Table ===")
    print(table.to_string(index=False))
    table.to_csv(OUT_TABLE, index=False)
    print(f"\nSaved table → {OUT_TABLE}")

    # Bar plot
    plot_bar(summary, OUT_BAR)

    # Box plot
    plot_box(dfA, dfB, OUT_BOX)

    # Wilcoxon
    wilcoxon_test(dfA, dfB)

    print("\n✅ Comparison done.")

if __name__ == "__main__":
    main()
