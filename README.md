# Assignment Data

This directory contains the data you will evaluate. All files are
UTF-8 encoded JSONL (one JSON object per line). Japanese text is
preserved in native script; no escaping is applied.

## Files

### `articles.jsonl` — 50 news articles from XL-Sum (Japanese)

Each row:

| Field | Type | Description |
|---|---|---|
| `article_id` | string | Stable identifier (may contain hyphens). |
| `url` | string | Original BBC Japanese article URL. |
| `title` | string | Article title. |
| `text` | string | Full article body. |
| `reference_summary` | string | XL-Sum's reference summary. **Note:** XL-Sum reference quality is known to be uneven; treat these as one signal among many, not ground truth. |

Sampled deterministically from the XL-Sum `test` split using seed=42,
filtered to articles between 300 and 4000 characters with a non-trivial
reference summary.

### `summaries.jsonl` — 250 summaries to evaluate (5 per article)

Each row:

| Field | Type | Description |
|---|---|---|
| `summary_id` | string | Stable identifier, formatted as `<article_id>_<opaque_suffix>`. The suffix is **intentionally opaque** and carries no grading information. Do not derive features from it. |
| `article_id` | string | The source article this summary was written for. Join to `articles.jsonl` to get the source text. |
| `summary` | string | The summary text itself. |

Each article has exactly 5 summaries of varying quality produced by
different methods. You do **not** know which summary was produced by
which method — discovering that is part of the evaluation problem.

## What a good evaluator should be able to do

There is no hidden answer key for individual summaries. The
summaries vary in quality by construction, and a well-designed
evaluation should be able to:

- **Rank** the 5 summaries per article in roughly the correct quality
  order.
- **Distinguish** high-quality summaries from degraded ones.
- **Detect** specific failure modes (faithfulness violations,
  coverage failures, etc.) that are present in some of the 250
  summaries.

Use these capabilities as a lens for self-evaluating your own
evaluation system.

## Recommended loading

Both files can be loaded with one line in Python:

~~~python
import json
articles  = [json.loads(l) for l in open("articles.jsonl",  encoding="utf-8")]
summaries = [json.loads(l) for l in open("summaries.jsonl", encoding="utf-8")]
~~~

Join on `article_id` when you need article text for a given summary.
