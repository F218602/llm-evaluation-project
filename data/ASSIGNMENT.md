# LLM Application Evaluation Design — Take-Home Assignment

**Estimated time**: 6–8 hours

## Context

You are an engineer responsible for evaluating the quality of
LLM-powered applications. The target application for this
assignment is a simple Japanese news article summarizer: given a
news article, it returns a summary of 3 sentences or fewer.

For this assignment, you do **not** need to build the summarizer.
We provide 250 summaries (5 per article, across 50 articles)
produced by a variety of methods and quality levels. Your job is to
**discover what quality means for this application, design an
evaluation that reflects that understanding, and validate it**.

## What this assignment measures

The primary subject of this assignment is your ability to **explore
and design an evaluation**, not your ability to write production
code. Specifically, we are looking for:

- How you interrogate the data to discover what failure modes exist
  and what quality dimensions matter.
- How you translate that understanding into an evaluation design.
- How you convince a reader that the evaluation's signals can be
  trusted.
- How you reason about what your evaluation does and does not
  measure.

Implementation quality is **not** a scoring axis. Your code needs
to run well enough to produce the results you report, but it does
not need to be polished, tested, or architected for production.
Submit whatever you actually used — including notebooks, scratch
files, and abandoned experiments if you find them useful as
evidence of your exploration.

## Data Provided

All data lives under `data/`. See `data/README.md` for the full
schema.

- **`data/articles.jsonl`** — 50 news articles from XL-Sum Japanese
  (`article_id`, `url`, `title`, `text`, `reference_summary`).
- **`data/summaries.jsonl`** — 250 summaries to evaluate
  (`summary_id`, `article_id`, `summary`). Each article has 5
  summaries produced by different methods. The methods and the
  quality of each summary are **not disclosed** to you.

The variety in the summaries is deliberate. Some are well-written
and faithful; others contain planted failure modes that your
evaluator should be able to detect. Discovering the distribution
empirically is part of the problem.

A note on the reference summaries in `articles.jsonl`: XL-Sum's
reference quality is known to be uneven. Treat references as one
signal among many, not as ground truth, and inspect them critically.

## Your Task

1. **Explore the data.** Read the articles and summaries. Form
   hypotheses about what failure modes are present, what quality
   dimensions matter, and how summaries differ from each other.
   The shape of the problem is something you discover, not
   something we hand you.
2. **Design an evaluation.** Based on what you found, define the
   quality dimensions you will measure, select or construct the
   metrics, and produce a score (or a set of scores) per summary.
   Your scores should make summaries comparable within and across
   articles.
3. **Validate the evaluation.** An evaluation is only useful to the
   extent its signals can be trusted. Demonstrate why a reader
   should believe your numbers. The choice of approach is yours to
   defend. This step is central, not optional.
4. **Write a report.** Describe what you discovered, the design
   decisions that followed, how you validated the evaluation, and
   what it does not measure.

## Deliverables

Submit your work as a zip file with the following structure:

~~~
submission/
├── README.md            # How to run what you submitted, and a
│                        # brief note on how you used AI tools.
├── report.md            # Your exploration, design, validation,
│                        # and limitations. PDF also acceptable.
│                        # 3–5 pages as a guideline.
├── scores.jsonl         # Your evaluator's output — one row per
│                        # summary_id with at least one score field.
│                        # Format is up to you; document it in README.
└── code/                # Whatever code you used to produce scores.jsonl
    └── ...              # and to run your validation. Notebooks,
                         # scratch files, and abandoned experiments
                         # are welcome here if they are part of how
                         # you actually worked.
~~~

Your submission should run against the provided `data/` directory
without modification. A `requirements.txt` or equivalent is
appreciated but not required if your README makes the environment
clear.

### What the report should cover

The report is the primary artifact. Suggested sections (adapt as
you see fit):

1. **Exploration.** What you saw when you looked at the data. What
   hypotheses you formed. What patterns or failure modes you
   identified. Which of your initial hypotheses survived contact
   with the data and which did not.
2. **Design.** The quality dimensions you chose to measure, the
   metrics you used to measure them, and why. Approaches you
   considered and rejected, and why you rejected them.
3. **Validation.** How you established that your evaluation's
   signals can be trusted. What you did, what you found, and what
   that tells a reader about the reliability of the scores.
4. **Limitations.** What your evaluation does not measure, what it
   measures poorly, and what you would do next if you had more
   time.

## Evaluation Criteria

Your work will be evaluated along the following axes, listed
roughly in order of weight.

- Depth of exploration
- Soundness of evaluation design
- Validation of the evaluation
- Self-awareness of limitations
- Reproducibility

Implementation quality is **not** scored.

## Constraints

- **You may use any language or library.**
- **Use of generative AI tools is permitted and expected.** In your
  README, briefly describe how you used them: which decisions were
  yours versus AI-assisted, and any prompts or interactions that
  materially shaped the result. We care about your judgment, not
  your typing speed.

This is an open-ended problem by design. We hope you find it
interesting to work on. Good luck.

---

Note: The data is in Japanese, but this assignment description is
in English. Both languages may be freely used in your report and
code comments.
