import os
import json
import glob
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

# -----------------------------
# 1. LOAD ALL ITERATIONS
# -----------------------------
output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
files = sorted(glob.glob(os.path.join(output_dir, "scores_*.jsonl")))

if len(files) < 2:
    raise Exception("Need at least 2 iterations.")

print("Files used:")
for f in files:
    print(f)

def load_jsonl(path):
    return pd.DataFrame([json.loads(line) for line in open(path, encoding="utf-8")])

dfs = [load_jsonl(f) for f in files]
iterations = [f"Iteration {i+1}" for i in range(len(dfs))]

metrics = ["faithfulness", "coverage", "relevance", "coherence", "conciseness"]

# Ensure same length
n = len(dfs[0])
for df in dfs:
    if len(df) != n:
        raise Exception("Mismatch in row count across iterations")

# Merge all iterations on summary_id for paired analysis
base_df = dfs[0][['summary_id']].copy()
for i, df in enumerate(dfs):
    base_df = base_df.merge(df[metrics + ['summary_id']], 
                           on='summary_id', 
                           suffixes=('', f'_iter{i+1}'))

# -----------------------------
# GRAPH 1: BAR GRAPH (PER ITERATION) - Average Scores
# -----------------------------
fig_bar = go.Figure()

for i, df in enumerate(dfs):
    avg_scores = [df[m].mean() / 5 for m in metrics]

    fig_bar.add_trace(go.Bar(
        name=iterations[i],
        x=metrics,
        y=avg_scores
    ))

fig_bar.update_layout(
    title="Graph 1: Metric Scores per Iteration (Normalized 0-1)",
    yaxis=dict(title="Score (0–1)", range=[0,1]),
    barmode='group',
    height=500
)
fig_bar.show()

# -----------------------------
# GRAPH 2: CATEGORY COMPARISON (Factual / Style / Overall)
# -----------------------------
categories = {
    "Factual": ["faithfulness", "coverage"],
    "Style": ["relevance", "coherence", "conciseness"]
}

fig_cat = go.Figure()

for i, df in enumerate(dfs):
    factual = df[categories["Factual"]].mean().mean()
    style = df[categories["Style"]].mean().mean()
    overall = df[metrics].mean().mean()

    fig_cat.add_trace(go.Bar(
        name=iterations[i],
        x=["Factual", "Style", "Overall"],
        y=[factual, style, overall]
    ))

fig_cat.update_layout(
    title="Graph 2: Category Comparison (Raw Scores)",
    yaxis=dict(title="Score (0-5)", range=[0,5]),
    barmode='group',
    height=500
)
fig_cat.show()

# -----------------------------
# GRAPH 3: CORRELATION HEATMAP (vs Iteration 1 Baseline)
# -----------------------------
correlation_data = []

base_df_iter1 = dfs[0][metrics].values

for i, df in enumerate(dfs):
    if i == 0:
        corr_row = [1.0] * len(metrics)
    else:
        corr_row = [stats.pearsonr(base_df_iter1[:, j], df[metrics[j]].values)[0] 
                   for j in range(len(metrics))]
    correlation_data.append(corr_row)

corr_df = pd.DataFrame(correlation_data, columns=metrics, index=iterations)

fig_corr_heatmap = px.imshow(
    corr_df,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="RdYlGn",
    zmin=0,
    zmax=1
)

fig_corr_heatmap.update_layout(
    title="Graph 3: Correlation Heatmap vs Iteration 1 (Baseline)<br>"
          "<sub>1.0 = Perfect consistency | Closer to 1.0 = More consistent</sub>",
    height=500
)
fig_corr_heatmap.show()

# -----------------------------
# GRAPH 4: ABSOLUTE DIFFERENCES
# -----------------------------
pair_labels = [f"{iterations[i]} vs {iterations[i+1]}" for i in range(len(dfs)-1)]

fig_diff = go.Figure()

for i in range(len(dfs)-1):
    diff_data = []
    for metric in metrics:
        col1 = f"{metric}_iter{i+1}" if i > 0 else metric
        col2 = f"{metric}_iter{i+2}"
        mean_abs_diff = np.mean(np.abs(base_df[col1] - base_df[col2]))
        diff_data.append(mean_abs_diff)
    
    fig_diff.add_trace(go.Bar(
        name=pair_labels[i],
        x=metrics,
        y=diff_data
    ))

fig_diff.update_layout(
    title="Graph 4: Mean Absolute Score Differences<br><sub>Lower = More Consistent</sub>",
    yaxis=dict(title="Mean |Diff|", range=[0, 3]),
    barmode='group',
    height=500
)
fig_diff.show()

# -----------------------------
# GRAPH 5: OVERALL CONSISTENCY SCORE
# -----------------------------
fig_summary = go.Figure()

consistency_scores = []
for i in range(len(dfs)-1):
    total_corr = sum(stats.pearsonr(base_df[f"{m}_iter{i+1}" if i > 0 else m], 
                                   base_df[f"{m}_iter{i+2}"])[0] 
                     for m in metrics)
    avg_consistency = total_corr / len(metrics)
    consistency_scores.append(avg_consistency)

colors = ['green' if x > 0.7 else 'orange' if x > 0.5 else 'red' 
          for x in consistency_scores]

fig_summary.add_trace(go.Bar(
    x=pair_labels,
    y=consistency_scores,
    marker_color=colors,
    text=[f'{x:.3f}' for x in consistency_scores],
    textposition='auto'
))

fig_summary.update_layout(
    title="Graph 5: Overall Consistency Score<br>"
          "<sub>🟢 >0.7 Excellent | 🟠 0.5-0.7 Good | 🔴 <0.5 Poor</sub>",
    yaxis=dict(title="Avg Correlation", range=[0, 1]),
    height=500
)
fig_summary.show()

# -----------------------------
# 6. CONSISTENCY SUMMARY TABLE
# -----------------------------
print("\n" + "="*60)
print("CONSISTENCY ANALYSIS SUMMARY")
print("="*60)

for i in range(len(dfs)-1):
    print(f"\n{pair_labels[i]}:")
    print("-" * 40)
    total_corr = 0
    for metric in metrics:
        col1 = f"{metric}_iter{i+1}" if i > 0 else metric
        col2 = f"{metric}_iter{i+2}"
        corr, _ = stats.pearsonr(base_df[col1], base_df[col2])
        mean_diff = np.mean(np.abs(base_df[col1] - base_df[col2]))
        print(f"  {metric:12}: Corr={corr:.3f} | Δ={mean_diff:.2f}")
        total_corr += corr
    
    overall = total_corr / len(metrics)
    status = "✅ EXCELLENT" if overall > 0.7 else "⚠️ GOOD" if overall > 0.5 else "❌ POOR"
    print(f"  OVERALL:       {overall:.3f} | {status}")

global_consistency = np.mean(consistency_scores)
print(f"\n🌍 GLOBAL CONSISTENCY: {global_consistency:.3f}")
print("   " + ("✅ EXCELLENT" if global_consistency > 0.7 else "⚠️ GOOD" if global_consistency > 0.5 else "❌ POOR"))

print("\n" + "="*60)