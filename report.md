## 1. Exploration

### 1.1 Data Understanding

To understand the dataset, I first created a script to merge `articles.jsonl` and `summaries.jsonl` using `article_id`. This allowed me to view each article alongside its five corresponding summaries.

I sampled 3 articles and generated structured outputs containing:
- Article title, text, and URL
- Reference summary
- All 5 generated summaries

Since the dataset is in Japanese, I translated the articles and summaries into English to enable deeper qualitative analysis.

---

### 1.2 Manual Inspection

For each sampled article, I:

- Read the full article (via BBC URL for full context)
- Compared all 5 summaries side-by-side
- Evaluated each summary manually based on:
  - factual correctness
  - completeness
  - relevance
  - clarity
  - length

This process helped identify how summaries differ in quality and where they fail.

---

### 1.3 Observed Failure Modes

Through manual comparison, several recurring failure patterns emerged:

#### 1. Hallucination / Incorrect Information
Some summaries introduced facts not present in the article.

**Example:**
A summary incorrectly stated that the suspects escaped to another state, while the article confirmed they were killed.

---

#### 2. Incomplete Summaries
Some outputs were truncated or ended abruptly.

**Example:**
A summary ending mid-sentence after “the perpetrators fled…” without resolution.

---

#### 3. Missing Key Information (Low Coverage)
Some summaries captured only part of the story while missing critical elements such as:
- outcome of the event
- number of casualties
- key entities or context

---

#### 4. Poor Information Selection (Low Relevance)
Some summaries focused on secondary details instead of the main event.

**Example:**
Describing tactical details (SUV movement, police positioning) while missing the core event and outcome.

---

#### 5. Overly Extractive / Verbose Summaries
Some summaries copied large portions of the article without proper compression.

---

#### 6. Structural / Coherence Issues
Some summaries were:
- disjointed
- hard to follow
- poorly structured

---

### 1.4 Key Insight

A critical observation from this exploration was:

> Summary quality is multi-dimensional, and different failure modes occur independently.

For example:
- A summary can be **factually correct but incomplete**
- A summary can be **complete but include hallucinated details**
- A summary can be **concise but irrelevant**

This indicates that **no single metric (e.g., ROUGE) can capture overall quality**.

---

### 1.5 Dimension Discovery

Initially, ~10 potential evaluation dimensions were identified, including:

- Faithfulness  
- Coverage  
- Relevance  
- Coherence  
- Conciseness  
- Fluency  
- Non-redundancy  
- Completeness  
- Consistency  
- Style adherence  

However, many of these overlapped or were subsets of broader concepts.

---

### 1.6 Final Metric Selection

The dimensions were consolidated into 5 core evaluation criteria:

| Final Metric   | Captures |
|----------------|--------|
| Faithfulness   | hallucinations, factual errors |
| Coverage       | missing key information |
| Relevance      | importance of selected content |
| Coherence      | structure and readability |
| Conciseness    | length efficiency and compression |

These five dimensions together cover all major observed failure modes.

---

### 1.7 Reference Summary Analysis

During exploration, I also observed that:

> The provided reference summaries are not consistently high quality.

Some were:
- incomplete
- overly compressed
- missing important details

Therefore:
- Reference summaries were **not treated as ground truth**
- They were used only as a **supporting signal (e.g., for ROUGE comparison)**

---

### 1.8 Manual Scoring Validation

To validate the identified dimensions, I manually scored summaries for one article:

| Dimension    | S1 | S2 | S3 | S4 | S5 |
|--------------|----|----|----|----|----|
| Faithfulness | 5  | 5  | 5  | 5  | 2  |
| Coverage     | 3  | 2  | 2  | 3  | 3  |
| Relevance    | 5  | 4  | 4  | 5  | 3  |
| Coherence    | 5  | 2  | 4  | 5  | 5  |
| Conciseness  | 3  | 3  | 3  | 3  | 3  |

This confirmed that:
- The chosen dimensions effectively distinguish summary quality
- Different summaries fail in different ways across dimensions

---

### 1.9 Conclusion from Exploration

The exploration phase established that:

- Summary evaluation must be **multi-dimensional**
- Metrics must be **independent**
- LLM-based evaluation is necessary for:
  - semantic understanding
  - factual grounding
- Heuristic metrics alone (e.g., ROUGE) are insufficient

These findings directly informed the evaluation design described in the next section.

## 2. Evaluation Design

### 2.1 Design Objective

The objective of the evaluation system is to assess the quality of generated news summaries in a way that:

- Reflects real-world usefulness
- Captures multiple independent quality dimensions
- Produces interpretable and comparable scores
- Aligns with human judgment

From the exploration phase, it was clear that summary quality is inherently multi-dimensional, and no single metric is sufficient.

---

### 2.2 Selected Evaluation Dimensions

The evaluation is based on five key dimensions:

| Metric        | Purpose |
|--------------|--------|
| Faithfulness | Detect factual correctness and hallucinations |
| Coverage     | Measure completeness of key information |
| Relevance    | Ensure focus on important content |
| Coherence    | Evaluate clarity and structure |
| Conciseness  | Measure efficiency of compression |

These dimensions were selected because they collectively capture the major failure modes observed during exploration.

---

### 2.3 Mapping Dimensions to Failure Modes

Each metric corresponds to a specific failure pattern:

- Faithfulness → penalizes hallucinated or incorrect information  
- Coverage → penalizes missing critical details  
- Relevance → penalizes focus on less important content  
- Coherence → penalizes incomplete or poorly structured summaries  
- Conciseness → penalizes overly long or overly short summaries  

This mapping ensures that the evaluation remains interpretable and actionable.

---

### 2.4 Metric Implementation Strategy

The evaluation combines two approaches:

#### LLM-based Evaluation

The following dimensions are evaluated using a Large Language Model:

- Faithfulness  
- Coverage  
- Relevance  
- Coherence  

These dimensions require semantic understanding and contextual reasoning, which cannot be reliably captured using rule-based or lexical metrics.

---

#### Heuristic-based Evaluation

Conciseness is evaluated using a length-based heuristic.

The ratio between summary length and article length is used to determine how efficiently the content has been compressed.

---

### 2.5 Conciseness Design

Conciseness is defined as the ability to preserve key information while reducing length.

The scoring is based on the ratio of summary length to article length:

- A **balanced compression range** (approximately 3%–8% of the original length) is considered optimal and receives the highest score  
- Slightly longer summaries receive moderately high scores  
- Very long summaries are penalized for lack of compression  
- Extremely short summaries are also penalized due to the risk of missing important information  

This design ensures that conciseness reflects both brevity and information preservation.

---

### 2.6 Prompt Design

A structured prompt is used to guide the LLM evaluation.

Key characteristics:

- Clear scoring rubrics (1–5 scale) for each metric  
- Explicit instruction to evaluate each metric independently  
- Emphasis on consistency and strict evaluation  
- Output restricted to structured JSON format  

This ensures reliable and standardized scoring across summaries.

---

### 2.7 Metric Independence

A key design decision was to enforce strict separation between metrics:

- Faithfulness evaluates only factual correctness  
- Coverage evaluates only completeness  
- Relevance evaluates importance of content  
- Coherence evaluates readability and structure  

Each metric is assessed independently to avoid overlap and bias.

---

### 2.8 Final Scoring Strategy

A weighted scoring approach is used to compute the overall quality score.

The weights reflect the relative importance of each dimension:

- Faithfulness and Coverage are given the highest importance  
- Relevance and Coherence are moderately weighted  
- Conciseness has the lowest weight but still contributes to the final score  

This prioritization ensures that correctness and completeness are valued over stylistic aspects.

---

### 2.9 Bottleneck Metric (Weakest Dimension)

In addition to the overall score, the system identifies the weakest-performing dimension for each summary.

This represents the primary limitation of the summary and provides a clear explanation of where it fails.

---

### 2.10 Role of Reference Summary

The reference summary is used only as a supplementary signal.

It is not treated as ground truth because:

- Its quality is inconsistent across the dataset  
- Some references are incomplete or overly compressed  

This prevents bias and ensures a more robust evaluation.

---

### 2.11 Design Summary

The evaluation system:

- Uses LLMs for semantic evaluation  
- Uses heuristics for structural evaluation  
- Separates metrics to avoid bias  
- Applies weighted scoring for overall quality  
- Provides interpretable outputs through individual and bottleneck scores  

This design allows the system to evaluate summaries in a way that closely reflects real-world quality expectations.

## 3. Validation

### 3.1 Human vs LLM Agreement

To validate whether the evaluation reflects real quality, I manually scored summaries for a sample article across all dimensions.

The LLM-generated scores were then compared against manual judgments.

Result:
- The LLM ranking closely matched the human ranking of summaries
- High-quality summaries (S1, S4) consistently received higher scores
- Poor summaries (incomplete or hallucinated) received lower scores

This demonstrates that the evaluation aligns well with human intuition.

---

### 3.2 Consistency Across Runs

To test reliability, the evaluation was executed multiple times on the same dataset.

Results from consecutive runs were compared using:
- metric-wise score differences
- visualization plots

Observation:
- Score variance across runs was minimal (typically < 0.3)
- Relative ranking of summaries remained stable

This indicates that the evaluation is consistent and reproducible.

---

### 3.3 Metric Sensitivity to Failure Modes

The evaluation was further validated by checking whether each metric correctly penalizes its intended failure mode.

Examples:

- Hallucinated summaries → low Faithfulness
- Incomplete summaries → low Coherence and Coverage
- Irrelevant summaries → low Relevance
- Overly long summaries → lower Conciseness

This confirms that each metric behaves as expected and captures distinct quality aspects.

---

### 3.4 Overall Validation Conclusion

The evaluation system demonstrates:

- Alignment with human judgment
- Stability across multiple runs
- Sensitivity to different failure modes

Together, these provide confidence that the evaluation produces meaningful and reliable scores.

## 4. Limitations

While the proposed evaluation system captures multiple important aspects of summary quality, it has several limitations.

---

### 4.1 Dependence on LLM Judgement

The evaluation relies heavily on an LLM for scoring key dimensions such as faithfulness, coverage, relevance, and coherence.

As a result:
- Scores may vary slightly across runs
- The model may occasionally misinterpret context or nuance
- There is no absolute guarantee of correctness

Although consistency checks show low variance, the evaluation is still probabilistic rather than deterministic.

---

### 4.2 Limited Ground Truth for Validation

The dataset does not provide a reliable ground truth for summary quality.

- Reference summaries are inconsistent in quality
- Some are incomplete or overly compressed

As a result:
- Validation relies on human judgment and qualitative comparison rather than objective benchmarks

---

### 4.3 Conciseness as a Length-Based Heuristic

Conciseness is measured purely based on length ratio.

This introduces limitations:
- A short summary may still omit important information
- A longer summary may be justified if the article is complex
- The metric does not directly evaluate redundancy or information density

Thus, conciseness is only an approximate measure of summarization quality.

---

### 4.4 Limited Coverage of Linguistic Quality

The evaluation does not explicitly measure:

- Fluency (grammar and natural language quality)
- Stylistic appropriateness
- Tone or readability nuances

While coherence partially captures readability, finer aspects of language quality are not directly evaluated.

---

### 4.5 Evaluation Scope Limited to Sample Size

Due to API and time constraints:

- Full evaluation across all 250 summaries may be time-intensive
- Some validation experiments were performed on a subset of the data

Although the system generalizes conceptually, broader validation would strengthen confidence.

---

### 4.6 Potential Bias from Prompt Design

The behavior of the evaluation is influenced by the prompt used for the LLM.

- Slight changes in wording may affect scoring
- The model may exhibit implicit biases in interpreting importance or relevance

This introduces a dependency on prompt quality.

---

### 4.7 Lack of Fine-Grained Explanation

The system produces numerical scores but does not provide detailed explanations for each score.

- Difficult to fully understand *why* a summary received a specific score
- Limits debugging and interpretability at a deeper level

---

### 4.8 Future Improvements

If more time were available, the following improvements could be explored:

- Add explanation generation for each score
- Incorporate fluency and style metrics
- Use multiple LLM calls and average scores to improve stability
- Introduce pairwise ranking evaluation (comparing summaries directly)
- Expand validation with more human-labeled samples

---

### 4.9 Summary of Limitations

The evaluation system is effective at capturing core summary quality dimensions, but:

- It depends on LLM judgment
- It uses heuristic approximations for some metrics
- It lacks full ground-truth validation

Despite these limitations, it provides a strong and practical framework for evaluating summary quality in a multi-dimensional manner.