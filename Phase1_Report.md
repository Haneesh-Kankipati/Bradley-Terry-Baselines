---
project: 1.NLP
type: deliverable
phase: 1
updated: 2026-08-28
deadline: 2026-08-28T23:55:00+05:30
---

# CS F429 NLP — Phase 1 Report

## Problem Statement

Direct Preference Optimization (DPO) is the standard method for aligning
language model outputs to human preferences without training an explicit
reward model. DPO decides between two candidate responses using only the
policy model's own log-probability margin. This project asks whether a
frozen, pre-trained language model's *hidden representations* carry
preference-relevant information that a policy's log-probability margin does
not — and whether that signal can be combined with the policy margin at
decision time to produce a more accurate preference judgment than DPO alone,
or than a classical statistical baseline built on surface features (response
length, lexical overlap, prompt similarity).

Concretely, we compare four approaches to the same task — given a prompt and
two candidate responses, predict which one a human would prefer — chosen to
trace the actual historical progression of preference-optimization methods:

1. **Classical baseline**: a Bradley-Terry-style logistic regression over
   hand-crafted surface features (response length, lexical overlap with the
   prompt, TF-IDF similarity).
2. **Early-modern baseline**: PPO-based RLHF — a reward model trained on the
   preference pairs, then used to fine-tune the policy via reinforcement
   learning.
3. **Modern baseline**: LoRA-based DPO fine-tuning of a pre-trained instruct
   model, which replaced the PPO pipeline as the standard alignment method.
4. **Proposed method**: a decision-time reranker that fuses the DPO policy's
   margin with a preference direction estimated from the frozen base model's
   hidden representations.

## Motivation

Preference optimization (RLHF, DPO, and their variants) is the primary
mechanism by which today's deployed LLMs are aligned to human judgment, and
reranking/preference-scoring components are used throughout production LLM
pipelines (response selection, reward modeling, best-of-n sampling). Whether
a preference decision is actually driven by semantic understanding of the
response, or by superficial correlates like length, is a live and
practically important question — it bears directly on how much we should
trust automated preference judgments used in these pipelines. This project
gives an empirical, controlled answer for one setting: does information in a
frozen model's residual stream add real preference signal beyond both
surface-level heuristics and the policy's own training objective, once
known confounds (length, lexical overlap) are explicitly controlled for.

## Dataset

- **`trl-lib/ultrafeedback_binarized`** (UltraFeedback, binarized) — primary
  training and evaluation corpus. Public, pairwise chosen/rejected
  completions spanning instruction-following, reasoning, and general chat
  prompts, on the order of 60K binarized pairs. Used for LoRA DPO training
  (256–512 pairs per run), representation-direction fitting (training-only
  pairs), and held-out evaluation (up to 4,000 untouched pairs in pooled
  runs).
- **`Anthropic/hh-rlhf`** (HH-RLHF) — secondary corpus, ~160K pairwise
  helpfulness/harmlessness preference judgments from human annotators. Used
  exclusively for cross-corpus transfer evaluation (train direction on one
  corpus, test on the other) to check whether any representation-based
  signal generalizes beyond the corpus it was fit on, rather than reporting only how well a
  method regurgitates preferences on its home training corpus.

**Domain**: general-purpose instruction-following / chat preference judgments
(UltraFeedback) and helpfulness/harmlessness dialogue preference judgments
(HH-RLHF). Both are standard, widely-cited public preference datasets used in
the DPO and RLHF literature, which makes results comparable to published
baselines.

**Limitations**: preference labels are subjective human/AI-assisted
annotations and carry annotator bias; response length correlates strongly
with stated preference in both corpora (a documented confound we explicitly
control for via length-only baselines and length-residualization); the two
corpora differ in domain and register, so cross-corpus transfer results are
expected to be, and are treated as, a harder and more informative test than
within-corpus accuracy.

**Preprocessing**: extraction of prompt-disjoint pairwise (chosen, rejected)
examples; tokenization via the base model's tokenizer
(`Qwen/Qwen2.5-1.5B-Instruct`); hidden-state extraction at layer 21 (~75%
network depth) with mean-pooled and last-token pooling variants; optional
length-residualization of extracted features to isolate signal beyond
response length.

## Plan

**Phase 2 (due Oct 15, 2026) — one working baseline + demo:**
- Implement the classical baseline as a standalone module: Bradley-Terry
  logistic regression over length, lexical-overlap, and TF-IDF-similarity
  features (built from existing nuisance-control code already validated
  in earlier exploration).
- Run LoRA DPO fine-tuning of `Qwen2.5-1.5B-Instruct` on UltraFeedback as
  the modern baseline.
- Stretch, run as a parallel workstream: train a Bradley-Terry reward model
  on the same preference pairs, then PPO-fine-tune the policy against it, as
  the early-modern baseline. Heavier lift than DPO (reward model + RL loop,
  more hyperparameter-sensitive) — scoped as a separate track so it doesn't
  block the core classical-vs-DPO comparison if it slips.
- Evaluate all implemented baselines on held-out, prompt-disjoint pairs;
  report accuracy, preliminary error breakdown.
- CLI/GUI demo showing training of the DPO baseline and a live
  multi-baseline prediction comparison on sample pairs.

**Phase 3 (due Nov 21, 2026) — novel contribution + full system:**
- Finalize the frozen-representation reranker: fuse the DPO policy margin
  with a preference direction fit on frozen base-model representations
  (training-pairs only), fixed before evaluation.
- Run a pre-registered confirmatory experiment: layer, estimator, pooling,
  residualization, and fusion coefficient fixed in advance from
  training-only validation; evaluated across both corpora and multiple
  seeds against the classical baseline, the PPO baseline (if completed),
  the DPO baseline, and length/lexical controls, with paired bootstrap
  confidence intervals.
- Quantitative error analysis (accuracy broken down by length bucket and by
  corpus) and qualitative error analysis (inspection of cases where the
  representation-based signal and the policy margin disagree).
- Final report and live demo of the complete three-way comparison.
