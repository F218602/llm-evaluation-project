# LLM Evaluation Project

---

## Overview

This project implements a **multi-dimensional evaluation system** to assess the quality of generated summaries.

Instead of relying on a single metric (e.g., ROUGE), summaries are evaluated across five independent dimensions:

- Faithfulness – factual correctness / hallucination detection  
- Coverage – completeness of key information  
- Relevance – importance of selected content  
- Coherence – structure and readability  
- Conciseness – compression efficiency  

The system combines:
- LLM-based evaluation (for semantic understanding)
- Heuristic scoring (for conciseness)

---

## Project Structure

```
LLM-EVALUATION-PROJECT/
│
├── src/ or scripts/
│   ├── exploration.py        → Main execution file
│   ├── retry_failed.py       → Re-run failed API calls from score files
│   ├── sampleArticles.py     → Sample articles + summaries for manual inspection
│   └── visualize_results.py  → Visualize iterations (requires ≥2 score files)
│
├── data/
│   ├── articles.jsonl
│   └── summaries.jsonl
│
├── outputs/
│   ├── scores_*.jsonl
│   ├── scores_2.jsonl
│   └── outputs.txt
│
├── .env.example
├── .gitignore
├── README.md
├── report.md
└── requirements.txt
```

---

## Dataset

- 50 Japanese BBC news articles  
- 250 generated summaries (5 per article)

Each summary is evaluated against its source article.

⚠️ Note: Reference summaries are NOT treated as ground truth and are used only as a supporting signal (ROUGE).

---

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Setup API Key (NVIDIA DeepSeek API)

This project uses NVIDIA-hosted DeepSeek models via API.

#### Step 1: Create `.env` file

```bash
cp .env.example .env
```

#### Step 2: Add API key

Open `.env` and add:

```
NVIDIA_API_KEY=your_api_key_here
```

---

## How API Key is Used

The API key is loaded using `python-dotenv`:

```python
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.environ["NVIDIA_API_KEY"]
```

And passed to the client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)
```

---

## Important

- Never commit `.env` to GitHub  
- `.env` must be listed in `.gitignore`  
- Only `.env.example` should be shared  

---

## Running the Project

```bash
python exploration.py
```

---

## Evaluation Design

### LLM-based Metrics

- Faithfulness  
- Coverage  
- Relevance  
- Coherence  

---

### Heuristic Metric

Conciseness:

```
summary length / article length
```

For Japanese text:
- Character-level length is used

---

## Scoring

Each metric is scored from 1–5.

Final weighted score:

- Faithfulness: 0.3  
- Coverage: 0.3  
- Relevance: 0.15  
- Coherence: 0.15  
- Conciseness: 0.1  

---

### Additional Outputs

- ROUGE-L (diagnostic only)  
- Dominating metric (lowest score)

---

## Output Format

```
outputs/scores_<timestamp>.jsonl
```

Each record contains:
- summary_id  
- article_id  
- metric scores  
- overall score  
- dominating metric  
- ROUGE-L  

---

## Utilities

- retry_failed.py → re-run failed API calls  
- visualize_results.py → generate graphs for multiple iterations  
- sampleArticles.py → inspect random article-summary samples  

---

## Notes

- Multi-dimensional evaluation (not single-metric based)  
- Metrics are independent and non-overlapping  
- Conciseness measures only length efficiency  

---

## Limitations

- LLM evaluation may vary across runs  
- Coverage depends on implicit key fact extraction  
- Conciseness does not measure semantic density  

---

## Author

Anisha Jeni Ravi Sam