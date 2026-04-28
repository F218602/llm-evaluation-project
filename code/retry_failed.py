import os
import time
import json
import glob
import pandas as pd
from openai import OpenAI, APIError, APITimeoutError
from rouge_score import rouge_scorer

# -----------------------------
# 1. SAFE API CALL WRAPPER
# -----------------------------
def safe_call(fn, retries=3):
    for i in range(retries):
        try:
            return fn()
        except (APIError, APITimeoutError) as e:
            print(f"[Retry {i+1}] API failed: {e}")
            time.sleep(2)
    raise Exception("API failed after retries")

# -----------------------------
# 2. SETUP
# -----------------------------
base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, "..", "data")
output_dir = os.path.join(base_dir, "..", "outputs")
output_jsonl_path = os.path.join(output_dir, "scores.jsonl")
os.makedirs(output_dir, exist_ok=True)

# Clear file before run

from dotenv import load_dotenv
load_dotenv()

from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_jsonl_path = os.path.join(output_dir, f"scores_{timestamp}.jsonl")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"]
)

# -----------------------------
# 3. INIT MODELS
# -----------------------------
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# -----------------------------
# 4. LOAD DATA
# -----------------------------
articles = pd.read_json(os.path.join(data_path, "articles.jsonl"), lines=True)
summaries = pd.read_json(os.path.join(data_path, "summaries.jsonl"), lines=True)

df = summaries.merge(articles, on="article_id")
print("Merged shape:", df.shape)

# -----------------------------
# 5. LLM EVALUATION (Refined: Balanced, pure isolation)
# -----------------------------
def evaluate_summary(article_text, summary):
    prompt = f"""
You are a strict and consistent news summary evaluator.

Evaluate SUMMARY against ARTICLE (a BBC news article) on 4 independent criteria (1–5).

IMPORTANT:
- Evaluate each criterion COMPLETELY INDEPENDENTLY. Do not let scores influence each other.
- For EVERY criterion, reason step-by-step internally first (but do not output reasoning).
- Output ONLY valid JSON: {{"faithfulness":1,"coverage":1,"relevance":1,"coherence":1}}
- Scores must be integers 1-5 only.

FAITHFULNESS (factual accuracy of SUMMARY's stated content ONLY):
- Examine ONLY claims explicitly made in SUMMARY. IGNORE ALL omissions/missing info from ARTICLE (penalize omissions ONLY in coverage).
- Check if EVERY claim in SUMMARY is directly supported by ARTICLE (exact match not required; paraphrases OK if truthful to BBC's reported facts).
- Do NOT penalize for brevity, rephrasing, or lack of detail—ONLY for hallucinations, inventions, contradictions, or misrepresentations of quotes/context.
Scoring:
5 = Every statement fully supported (zero hallucinations)
4 = One trivial phrasing/interpretation mismatch, but no factual error
3 = Minor weak support or one small unsupported detail
2 = Multiple unsupported/inaccurate claims
1 = Major hallucinations or factual errors dominate

COVERAGE (how well SUMMARY captures ARTICLE's key content):
- BBC news articles follow an inverted pyramid: lead para has top 5Ws (who/what/when/where/why); body adds details, quotes, context, outcomes.
- Internally list 5-10 MOST IMPORTANT facts from ARTICLE (prioritize: core event(s) from lead, key names (people/officials/orgs/victims), locations, dates/times, critical numbers (casualties, figures), official statements/outcomes, essential background/implications. Ignore minor examples, anecdotes, or filler).
- Count how many of these key facts appear in SUMMARY (direct or clear paraphrase counts; exact wording irrelevant).
- Reward concise summaries that hit BBC's high-impact essence (news summaries are ~20-30% of article length—focus on vital points, not full reproduction).
Scoring (proportion of your key facts list covered):
5 = All or nearly all (90-100%)
4 = Most (70-89%)
3 = About half (40-69%)
2 = Few (20-39%)
1 = Very few (<20%)

RELEVANCE (focus on ARTICLE's important content):
- Assess if SUMMARY sticks to BBC ARTICLE's core news topics (events, facts, implications) without fluff, speculation, or irrelevancies.
Scoring:
5 = Exclusively important ARTICLE content
4 = Mostly important, minimal minor details
3 = Balanced mix of important + some minor details
2 = Mostly minor/unimportant ARTICLE details
1 = Off-topic or dominated by extraneous content

COHERENCE (clarity and flow of SUMMARY alone):
- Ignore ARTICLE; evaluate SUMMARY's standalone readability/structure/logic as a news summary.
- Penalise partial or incomplete sentences (like but, and, or etc..) heavily.
Scoring:
5 = Exceptionally clear, logical, well-structured (flows like pro news)
4 = Clear with minor issues
3 = Mostly understandable but some awkwardness/jumps
2 = Often hard to follow (choppy/fragmented)
1 = Confusing, disjointed, or unreadable

ARTICLE:
{article_text}

SUMMARY:
{summary}
"""

    try:
        response = safe_call(lambda: client.chat.completions.create(
            model="deepseek-ai/deepseek-v3.2",
            messages=[
                {
                    "role": "system",
                    "content": "Strict evaluator. Think step-by-step per prompt. Output ONLY JSON: no explanations, no text."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            timeout=60
        ))
    except Exception as e:
        print("❌ API failed after retries:", e)
        return {"error": "api_failed"} 

    # Safety check (rare but good practice)
    if response is None:
        return {"error": "api_failed"}

    # Safety check (rare but good practice)
    if response is None:
        return {"error": "api_failed"}

    content = response.choices[0].message.content.strip()
    # Robust JSON cleanup: remove code fences only
    content = content.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(content)
        for k in ["faithfulness", "coverage", "relevance", "coherence"]:
            parsed[k] = max(1, min(5, int(parsed.get(k, 3))))
        return parsed
    except Exception as e:
        print("JSON parse error:", e)
        return {"error": "invalid_json"}

# -----------------------------
# 6. BASIC METRICS (Conciseness calculation via ROUGE)
# -----------------------------

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

    # Conciseness: FULL ARTICLE ratio 
    conciseness = map_length_to_conciseness(generated_summary, article_text)

    return {
        "rougeL": round(rouge_l, 3),
        "conciseness": conciseness  
    }

# -----------------------------
# 7. FINAL SCORING
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

    weights = {
        "faithfulness": 0.3,
        "coverage": 0.3,
        "relevance": 0.15,
        "coherence": 0.15,
        "conciseness": 0.1
    }

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
# 8. RERUN ERRORS FROM LATEST FILE
# -----------------------------
print("\nRerunning error records from latest scores file...")

# Find latest scores_*.jsonl
files = [f for f in os.listdir(output_dir) if f.startswith("scores_") and f.endswith(".jsonl")]
if not files:
    print("No previous scores file found. Skipping.")
else:
    # Sort by modification time (latest first)
    files_with_mtime = [(f, os.path.getmtime(os.path.join(output_dir, f))) for f in files]
    files_with_mtime.sort(key=lambda x: x[1], reverse=True)
    latest_file = os.path.join(output_dir, files_with_mtime[0][0])
    print(f"Latest file: {latest_file}")

    # Load all records
    records = []
    with open(latest_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    # Find error records
    error_recs = [r for r in records if r.get('error')]
    if not error_recs:
        print("No errors found. Done.")
    else:
        print(f"Found {len(error_recs)} errors to rerun.")

        updated_count = 0
        skipped_count = 0
        for rec in error_recs:
            summary_id = rec['summary_id']
            article_id = rec['article_id']

            # Find row in df
            matching_rows = df[df['summary_id'] == summary_id]
            if matching_rows.empty:
                print(f"⚠️ Row not found for {summary_id}. Skipping.")
                skipped_count += 1
                continue

            row = matching_rows.iloc[0]
            print(f"\n→ Rerunning: {summary_id} (article: {article_id})")

            summary_text = row["summary"]
            article_text = row["text"]
            reference_summary = row["reference_summary"]

            # STEP 1: basic metrics
            basic_scores = compute_basic_metrics(reference_summary, summary_text, article_text)

            # STEP 2: LLM evaluation
            print("→ Calling LLM...")
            llm_scores = evaluate_summary(article_text, summary_text)

            # STEP 3: final scoring
            overall_score, dominating_metric = compute_final_score(llm_scores, basic_scores)

            new_result = {
                "summary_id": summary_id,
                "article_id": article_id,
                **llm_scores,
                "conciseness": basic_scores["conciseness"],
                "overall_score": overall_score,
                "dominating_metric": dominating_metric,
                **basic_scores
            }

            # Replace in records
            replaced = False
            for i in range(len(records)):
                if records[i].get('summary_id') == summary_id:
                    records[i] = new_result
                    replaced = True
                    updated_count += 1
                    print("✅ UPDATED:", {k: v for k, v in new_result.items() if k not in ["gen_length", "ref_length"]})
                    break

            if not replaced:
                print(f"⚠️ Could not replace record for {summary_id}")

        # Write back the entire file
        print(f"\nWriting back to {latest_file} ({updated_count}/{len(error_recs)} updated, {skipped_count} skipped)")
        with open(latest_file, 'w', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

        print("\n✅ RERUN DONE")