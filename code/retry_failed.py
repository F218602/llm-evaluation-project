import os
import json
import time
import glob
import pandas as pd
from openai import OpenAI, APIError, APITimeoutError
from dotenv import load_dotenv
from rouge_score import rouge_scorer

# -----------------------------
# 1. ENV + CLIENT
# -----------------------------
load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"]
)

# -----------------------------
# 2. SAFE CALL
# -----------------------------
def safe_call(fn, retries=3):
    for i in range(retries):
        try:
            return fn()
        except (APIError, APITimeoutError) as e:
            print(f"[Retry {i+1}] API failed: {e}")
            time.sleep(2)
    return None

# -----------------------------
# 3. LOAD LATEST SCORE FILE
# -----------------------------
base_dir = os.path.dirname(__file__)
output_dir = os.path.join(base_dir, "..", "outputs")

files = sorted(glob.glob(os.path.join(output_dir, "scores_*.jsonl")))
if not files:
    raise Exception("No score files found.")

latest_file = files[-1]
print(f"\nUsing file: {latest_file}")

df = pd.DataFrame([json.loads(line) for line in open(latest_file, encoding="utf-8")])

# -----------------------------
# 4. LOAD ORIGINAL DATA
# -----------------------------
data_path = os.path.join(base_dir, "..", "data")
articles = pd.read_json(os.path.join(data_path, "articles.jsonl"), lines=True)
summaries = pd.read_json(os.path.join(data_path, "summaries.jsonl"), lines=True)

full_df = summaries.merge(articles, on="article_id")

# -----------------------------
# 5. ROUGE + CONCISENESS
# -----------------------------
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def map_length_to_conciseness(summary_text, article_text):
    summary_len = len(summary_text.split())
    article_len = len(article_text.split())

    if article_len == 0:
        return 3  # safe fallback

    ratio = summary_len / article_len

    # Tuned for long articles + ultra-short summaries (matches your 5/4/3/5/5 example)
    if 0.02 <= ratio <= 0.06:
        return 5   # Optimal ultra-concise (30-90 words: dense TL;DR)
    elif 0.06 < ratio <= 0.10:
        return 4   # Good concise (90-150 words)
    elif 0.10 < ratio <= 0.20:
        return 3   # Adequate but bulkier
    elif ratio < 0.02:
        return 4   # Very short but dense potential (brevity rewarded)
    else:
        return 2   # Too verbose (>20%)

def compute_basic_metrics(reference_summary, generated_summary, article_text):
    # ROUGE-L F1 vs reference (pre-compute scorer outside if batching)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference_summary, generated_summary)
    rouge_l = rouge_scores['rougeL'].fmeasure

    # Conciseness: FULL ARTICLE ratio (not ref!)
    conciseness = map_length_to_conciseness(generated_summary, article_text)

    return {
        "rougeL": round(rouge_l, 3),
        "conciseness": conciseness 
    }

# -----------------------------
# 6. FINAL SCORE
# -----------------------------
def compute_final_score(llm_scores, basic_scores):
    if llm_scores.get("error"):
        return None, None

    faithfulness = llm_scores["faithfulness"]
    coverage = llm_scores["coverage"]
    relevance = llm_scores["relevance"]
    coherence = llm_scores["coherence"]
    conciseness = basic_scores["conciseness"]

    overall = (
        0.3 * faithfulness +
        0.3 * coverage +
        0.15 * relevance +
        0.15 * coherence +
        0.1 * conciseness
    )

    metric_values = {
        "faithfulness": faithfulness,
        "coverage": coverage,
        "relevance": relevance,
        "coherence": coherence,
        "conciseness": conciseness
    }

    dominating = min(metric_values, key=metric_values.get)

    return round(overall, 2), dominating

# -----------------------------
# 7. LLM EVALUATION
# -----------------------------
def evaluate_summary(article_text, summary):
    prompt = f"""
Evaluate SUMMARY against ARTICLE on 4 criteria (1–5).
Return ONLY JSON:
{{"faithfulness":int,"coverage":int,"relevance":int,"coherence":int}}

ARTICLE:
{article_text}

SUMMARY:
{summary}
"""

    response = safe_call(lambda: client.chat.completions.create(
        model="deepseek-ai/deepseek-v3.2",
        messages=[
            {"role": "system", "content": "Strict evaluator. Output JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        timeout=60
    ))

    if response is None:
        return {"error": "api_failed"}

    content = response.choices[0].message.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(content)
        for k in ["faithfulness", "coverage", "relevance", "coherence"]:
            parsed[k] = max(1, min(5, int(parsed.get(k, 3))))
        return parsed
    except:
        return {"error": "invalid_json"}

# -----------------------------
# 8. RETRY FAILED ROWS
# -----------------------------
failed_indices = df[df["error"] == "api_failed"].index.tolist()

print(f"\nFound {len(failed_indices)} failed rows")

for idx in failed_indices:
    row = df.loc[idx]
    summary_id = row["summary_id"]

    print(f"\nRetrying: {summary_id}")

    source_row = full_df[full_df["summary_id"] == summary_id].iloc[0]

    article_text = source_row["text"]
    summary_text = source_row["summary"]
    reference_summary = source_row["reference_summary"]

    # LLM retry
    llm_scores = evaluate_summary(article_text, summary_text)

    if llm_scores.get("error"):
        print("❌ Still failed")
        continue

    # recompute basic metrics
    basic_scores = compute_basic_metrics(reference_summary, summary_text)

    # recompute final score
    overall_score, dominating_metric = compute_final_score(llm_scores, basic_scores)

    # UPDATE ROW FULLY
    updated_row = {
        "summary_id": summary_id,
        "article_id": row["article_id"],
        **llm_scores,
        "conciseness": basic_scores["conciseness"],
        "overall_score": overall_score,
        "dominating_metric": dominating_metric,
        **basic_scores
    }

    # replace row
    df.loc[idx] = updated_row

    print("✅ Updated")

# -----------------------------
# 9. SAVE BACK (overwrite)
# -----------------------------
with open(latest_file, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

print("\n✅ Retry complete — file fully updated")