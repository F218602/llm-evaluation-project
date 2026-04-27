import os
import pandas as pd

# -----------------------------
# 1. SETUP
# -----------------------------

base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, "..", "data")

output_txt_path = os.path.join(base_dir, "..", "outputs", "output.txt")
output_jsonl_path = os.path.join(base_dir, "..", "outputs", "scores.jsonl")

os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)

# -----------------------------
# 2. LOAD DATA
# -----------------------------

articles = pd.read_json(os.path.join(data_path, "articles.jsonl"), lines=True)
summaries = pd.read_json(os.path.join(data_path, "summaries.jsonl"), lines=True)

df = summaries.merge(articles, on="article_id")

print("Merged shape:", df.shape)
# -----------------------------
# 3. SAMPLE OUTPUT
# -----------------------------

first_ids = df["article_id"].unique()[:3]

with open(output_txt_path, "w", encoding="utf-8") as f:

    for article_id in first_ids:
        sample = df[df["article_id"] == article_id]

        f.write("=" * 80 + "\n")
        f.write(f"ARTICLE ID: {article_id}\n\n")

        f.write("ARTICLE TITLE:\n")
        f.write(sample["title"].iloc[0] + "\n\n")

        f.write("ARTICLE TEXT:\n")
        f.write(sample["text"].iloc[0] + "\n\n")

        f.write("ARTICLE URL:\n")
        f.write(sample["url"].iloc[0] + "\n\n")

        f.write("REFERENCE SUMMARY:\n")
        f.write(sample["reference_summary"].iloc[0] + "\n\n")

        f.write("=" * 80 + "\n")
        f.write("GENERATED SUMMARIES:\n\n")

        for i, (_, row) in enumerate(sample.iterrows(), 1):
            f.write(f"Summary {i} (ID: {row['summary_id']}):\n")
            f.write(row["summary"] + "\n")
            f.write("-" * 50 + "\n")

        f.write("\n\n")

print("Output written to output.txt")