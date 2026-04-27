import os
import json
import glob
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------
# 1. LOAD LAST 2 RUNS
# -----------------------------
output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")

files = sorted(glob.glob(os.path.join(output_dir, "scores_*.jsonl")))

if len(files) < 2:
    raise Exception("Need at least 2 score files to compare.")

file1, file2 = files[-2], files[-1]

print(f"Comparing:\n{file1}\n{file2}")

def load_jsonl(path):
    return pd.DataFrame([json.loads(line) for line in open(path, encoding="utf-8")])

df1 = load_jsonl(file1)
df2 = load_jsonl(file2)

# -----------------------------
# 2. PREPARE METRICS
# -----------------------------
metrics = ["faithfulness", "coverage", "relevance", "coherence", "conciseness"]

# completeness = same as coverage (for now)
df1["completeness"] = df1["coverage"]
df2["completeness"] = df2["coverage"]

metrics_full = ["faithfulness", "coverage", "completeness", "relevance", "coherence", "conciseness"]

# overall
df1["overall"] = df1[metrics].mean(axis=1)
df2["overall"] = df2[metrics].mean(axis=1)

# averages
avg1 = df1[metrics_full + ["overall"]].mean()
avg2 = df2[metrics_full + ["overall"]].mean()

# -----------------------------
# 3. GROUPED BAR CHART
# -----------------------------
categories = {
    "Factual": ["faithfulness", "coverage", "completeness"],
    "Style": ["relevance", "coherence", "conciseness"],
    "Overall": ["overall"]
}

def category_avg(avg, keys):
    return np.mean([avg[k] for k in keys])

bar_x = []
bar_iter1 = []
bar_iter2 = []

for cat, keys in categories.items():
    bar_x.append(cat)
    bar_iter1.append(category_avg(avg1, keys))
    bar_iter2.append(category_avg(avg2, keys))

fig_bar = go.Figure(data=[
    go.Bar(name="Iteration 1", x=bar_x, y=bar_iter1),
    go.Bar(name="Iteration 2", x=bar_x, y=bar_iter2)
])

fig_bar.update_layout(
    title="Category Comparison (Iteration 1 vs 2)",
    yaxis=dict(title="Score", range=[0,5]),
    barmode='group'
)

fig_bar.show()

# -----------------------------
# 4. RADAR CHART
# -----------------------------
radar_metrics = metrics_full + ["overall"]

fig_radar = go.Figure()

fig_radar.add_trace(go.Scatterpolar(
    r=[avg1[m] for m in radar_metrics],
    theta=radar_metrics,
    fill='toself',
    name='Iteration 1'
))

fig_radar.add_trace(go.Scatterpolar(
    r=[avg2[m] for m in radar_metrics],
    theta=radar_metrics,
    fill='toself',
    name='Iteration 2'
))

fig_radar.update_layout(
    title="Radar Comparison of Metrics",
    polar=dict(radialaxis=dict(visible=True, range=[0,5]))
)

fig_radar.show()

# -----------------------------
# 5. LINE CHART (TREND)
# -----------------------------
fig_line = go.Figure()

fig_line.add_trace(go.Scatter(
    x=radar_metrics,
    y=[avg1[m] for m in radar_metrics],
    mode='lines+markers',
    name='Iteration 1'
))

fig_line.add_trace(go.Scatter(
    x=radar_metrics,
    y=[avg2[m] for m in radar_metrics],
    mode='lines+markers',
    name='Iteration 2'
))

fig_line.update_layout(
    title="Metric Trend Comparison",
    yaxis=dict(range=[0,5])
)

fig_line.show()

# -----------------------------
# 6. HEATMAP
# -----------------------------
heatmap_data = pd.DataFrame({
    "Iteration 1": avg1,
    "Iteration 2": avg2
}).loc[radar_metrics]

fig_heatmap = px.imshow(
    heatmap_data,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="RdYlGn"
)

fig_heatmap.update_layout(title="Metric Heatmap (Green=High, Red=Low)")
fig_heatmap.show()

# -----------------------------
# 7. CONSISTENCY SCORE
# -----------------------------
diff = abs(avg2 - avg1)
consistency = 1 - (diff / 5)

consistency_df = pd.DataFrame({
    "Metric": radar_metrics,
    "Consistency": [consistency[m] for m in radar_metrics]
})

fig_consistency = px.bar(
    consistency_df,
    x="Metric",
    y="Consistency",
    title="LLM Consistency (1 = Perfectly Stable)",
    range_y=[0,1]
)

fig_consistency.show()