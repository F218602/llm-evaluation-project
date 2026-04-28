# 1. Exploration

## 1.1 Data Understanding

To understand the dataset, I first merged `articles.jsonl` and `summaries.jsonl` using `article_id`, enabling a unified view of each article alongside its five generated summaries.

I then sampled three articles and constructed structured views containing:

* Article title, text, and URL
* Reference summary
* All five generated summaries

Since the dataset is in Japanese, I translated both articles and summaries into English to enable deeper qualitative analysis.

---

## 1.2 Manual Inspection

For each sampled article, I performed a detailed qualitative comparison:

* Read the full article (using the source URL when necessary for context)
* Compared all five summaries side-by-side
* Evaluated each summary manually across:

  * factual correctness
  * completeness
  * relevance
  * clarity
  * length

This step was critical in understanding how summaries differ and where they fail.

---

## 1.3 Observed Failure Modes

The following recurring failure patterns emerged:

### 1. Hallucination / Incorrect Information

Some summaries introduced facts not present in the article.

**Example:**
A summary claimed suspects escaped to another location, while the article confirmed they were killed.

---

### 2. Incomplete Summaries

Some summaries were truncated or ended abruptly.

**Example:**
A summary ending mid-sentence after “The perpetrators fled the scene, but”.

---

### 3. Missing Key Information (Low Coverage)

Some summaries omitted critical elements such as:

* event outcomes
* number of casualties
* key entities

---

### 4. Poor Information Selection (Low Relevance)

Some summaries focused on secondary or less important details.

**Example:**
Describing tactical movements while missing the core event.

---

### 5. Overly Extractive / Verbose Summaries

Some outputs copied large portions of the article without meaningful compression.

---

### 6. Structural / Coherence Issues

Some summaries were:

* disjointed
* difficult to follow
* poorly structured

---

## 1.4 Key Insight

A key observation from this analysis:

> Summary quality is inherently multi-dimensional, and failure modes occur independently.

Examples:

* A summary can be **factually correct but incomplete**
* A summary can be **complete but hallucinated**
* A summary can be **concise but irrelevant**

This demonstrates that **single-metric evaluation (e.g., ROUGE) is insufficient**.

---

## 1.5 Dimension Discovery

Initially, ~10 candidate dimensions were identified:

* Faithfulness
* Coverage
* Relevance
* Coherence
* Conciseness
* Fluency
* Non-redundancy
* Completeness
* Consistency
* Style adherence

However, several of these overlapped or were subsets of broader concepts.

---

## 1.6 Final Metric Selection

The dimensions were consolidated into five core metrics:

| Metric       | Captures                            |
| ------------ | ----------------------------------- |
| Faithfulness | factual correctness, hallucinations |
| Coverage     | completeness of key information     |
| Relevance    | importance of selected content      |
| Coherence    | structure and readability           |
| Conciseness  | compression efficiency              |

These metrics collectively cover all observed failure modes while remaining interpretable and minimally overlapping.

---

## 1.7 Reference Summary Analysis

During exploration, I observed that reference summaries were not consistently high quality.

Some were:

* incomplete
* overly compressed
* missing key details

Therefore:

* Reference summaries were **not treated as ground truth**
* They were used only as a **supporting signal** (e.g., for ROUGE comparison)

---

## 1.8 Manual Scoring Sanity Check

To validate the usefulness of the selected dimensions, I manually scored summaries for one article:

| Dimension    | S1 | S2 | S3 | S4 | S5 |
| ------------ | -- | -- | -- | -- | -- |
| Faithfulness | 5  | 5  | 5  | 5  | 2  |
| Coverage     | 4  | 2  | 2  | 4  | 3  |
| Relevance    | 5  | 4  | 4  | 5  | 3  |
| Coherence    | 5  | 2  | 4  | 5  | 5  |
| Conciseness  | 5  | 4  | 3  | 5  | 5  |

This confirmed:

* Metrics distinguish summaries effectively
* Different summaries fail along different dimensions

---

## 1.9 Conclusion from Exploration

The exploration phase established that:

* Evaluation must be **multi-dimensional**
* Metrics should be **independent**
* **LLM-based evaluation is necessary** for semantic understanding
* Heuristic metrics alone are insufficient

These insights directly informed the evaluation design.

---

# 2. Evaluation Design

## 2.1 Objective

The evaluation aims to:

* Reflect real-world summary quality
* Capture multiple independent dimensions
* Produce interpretable and comparable scores
* Align with human judgment

---

## 2.2 Evaluation Dimensions

The system evaluates summaries across five dimensions:

| Metric       | Purpose                                  |
| ------------ | ---------------------------------------- |
| Faithfulness | Detect hallucinations and factual errors |
| Coverage     | Measure completeness                     |
| Relevance    | Assess importance of content             |
| Coherence    | Evaluate structure and readability       |
| Conciseness  | Measure compression efficiency           |

---

## 2.3 Mapping Metrics to Failure Modes

Each metric directly targets observed failure modes:

* Faithfulness → hallucinations
* Coverage → missing information
* Relevance → poor content selection
* Coherence → structural issues
* Conciseness → verbosity or over-compression

This ensures interpretability and diagnostic value.

---

## 2.4 Metric Implementation

### LLM-based Evaluation

Used for:

* Faithfulness
* Coverage
* Relevance
* Coherence

These require semantic understanding and contextual reasoning.

---

### Heuristic-based Evaluation

Used for:

* Conciseness

Based on summary-to-article length ratio.

---

## 2.5 Conciseness Design

Conciseness is evaluated using compression ratio:

* Optimal range (~3–8%) → highest score
* Slight deviations → moderate score
* Very long summaries → penalized
* Extremely short summaries → penalized

This balances brevity with information retention.

---

## 2.6 Prompt Design

A structured prompt ensures consistent LLM evaluation:

* Clear scoring rubric (1–5 scale)
* Independent evaluation per metric
* Strict evaluation instructions
* JSON-only output

This reduces variability and enforces standardization.

---

## 2.7 Metric Independence

Each metric is evaluated independently:

* Faithfulness → correctness only
* Coverage → completeness only
* Relevance → importance only
* Coherence → structure only

This prevents overlap and bias.

---

## 2.8 Scoring Strategy

A weighted score combines all metrics:

* Higher weight: Faithfulness, Coverage
* Medium weight: Relevance, Coherence
* Lower weight: Conciseness

This prioritizes correctness over style.


## 2.8.1 Scoring Implementation Details

The final score is computed as a weighted combination of all metrics:

* Faithfulness: 0.3
* Coverage: 0.3
* Relevance: 0.15
* Coherence: 0.15
* Conciseness: 0.1

This reflects the prioritization of factual correctness and completeness over stylistic aspects.

---

### Conciseness Calculation

Conciseness is calculated using the ratio of summary length to article length:

* 2% – 6% → Score 5 (optimal ultra-concise summaries)
* 6% – 10% → Score 4
* 10% – 20% → Score 3
* < 2% → Score 4 (very short but potentially high-density)
* > 20% → Score 2 (overly verbose)

This rewards dense summaries while penalizing excessive length.

---

### ROUGE as Supporting Signal

* ROUGE-L (F1 score) is computed between generated and reference summaries
* Used only as a diagnostic signal
* Not included in final scoring due to inconsistent reference quality

## 2.9 Bottleneck Metric

For each summary, the weakest dimension is identified.

Defined as the lowest-scoring metric (unweighted)
Represents the primary weakness of the summary

This:

Explains failures clearly
Improves interpretability
Helps debugging

---

## 2.10 Role of Reference Summary

Reference summaries are used only as a **secondary signal**, not ground truth, due to inconsistent quality.

---

## 2.11 Design Summary

The evaluation system:

* Combines LLM and heuristic methods
* Uses independent metrics
* Produces weighted scores
* Provides interpretable outputs

---

# 3. Validation

## 3.1 Human vs LLM Agreement

A subset of summaries was manually scored and compared with LLM outputs.

**Observation:**

* Strong alignment in ranking
* High-quality summaries scored consistently higher
* Poor summaries scored lower

**Conclusion:**
The evaluation aligns with human judgment and is meaningful.

---

## 3.2 Consistency Across Runs

The evaluation was run across multiple iterations.

**Findings:**

* Score differences were minimal (typically < 0.3)
* Correlation across runs was high
* Rankings remained stable

**Conclusion:**
The evaluation is reliable and reproducible.

---

## 3.3 Sensitivity to Failure Modes

Each metric responds appropriately to specific issues:

* Hallucination → low Faithfulness
* Missing info → low Coverage
* Irrelevance → low Relevance
* Poor structure → low Coherence
* Verbosity → low Conciseness

**Conclusion:**
Metrics are non-redundant and capture distinct dimensions.

---

## 3.4 Overall Validation Conclusion

The evaluation demonstrates:

* Alignment with human judgment
* Stability across runs
* Sensitivity to failure modes

This provides confidence in its reliability and usefulness.

---

# 4. Limitations

## 4.1 Dependence on LLM Judgment

* Non-deterministic outputs
* Potential misinterpretation
* Probabilistic scoring

---

## 4.2 Lack of Ground Truth

* Reference summaries are unreliable
* Validation relies on qualitative judgment

---

## 4.3 Conciseness Approximation

* Based only on length
* Does not measure information density

---

## 4.4 Coverage Metric Calibration Bias

The coverage metric is inherently difficult to calibrate due to the absence of a clearly defined ground truth for what constitutes a “complete” summary.

The current approach approximates completeness by prompting the LLM to internally identify 5–10 key facts from the article and measure their presence in the summary. While structured, this introduces a systematic bias:

* The LLM may identify more key facts than what is realistically expected in a high-quality summary
* Concise summaries that correctly capture the main event may still be penalized for omitting secondary details
* Coverage scores therefore tend to be **systematically lower across the dataset**

This reveals a mismatch between:

* **Article-level completeness** (broad fact coverage)
  vs
* **Summarization intent** (high-impact compression)

**Implication:**
The metric may under-score otherwise strong summaries, particularly those optimized for brevity and relevance.

**Potential Improvements:**

* Calibrate the prompt using high-quality reference summaries
* Learn coverage expectations via supervised or preference-based methods
* Adjust scoring thresholds to better reflect realistic summarization standards

---

## 4.5 Limited Linguistic Evaluation

Does not explicitly measure:

* fluency
* tone
* stylistic quality

---

## 4.6 Limited Validation Scope

* Some validation performed on subsets
* Full-scale validation could strengthen results

---

## 4.7 Prompt Sensitivity

* Evaluation depends on prompt design
* Small changes may affect scores

---

## 4.8 Limited Explainability

* Produces scores but not detailed reasoning
* Harder to debug individual cases

---

## 4.9 Future Improvements

* Add explanation generation
* Include fluency/style metrics
* Use multiple LLM runs with averaging
* Introduce pairwise ranking evaluation
* Expand human-labeled validation

---

## 4.10 Summary of Limitations

The system is effective but:

* Relies on LLM judgment
* Uses heuristic approximations
* Lacks full ground-truth validation

Despite these limitations, it provides a strong and practical multi-dimensional evaluation framework.

