# Experiments Catalog & Protocols ? BankingLLM-Optimizer

This document details the nine standardized experimental protocols executed across the BankingLLM-Optimizer research program.

---

## Experiment 1: Zero-Shot Baseline
- **Objective**: Establish the raw baseline capability of the LLM to classify 77 fine-grained intents without demonstrations.
- **Dataset Partition**: 500 validation samples (`eval_val`).
- **Input Parameters**:
  - Model: `openai/gpt-oss-20b`
  - Temperature: `0.0`
  - Demonstrations: `None`
  - Prompt: `prompts/zero_shot.txt`
- **Output Artifacts**: `results/tables/zero_shot_results.csv`

---

## Experiment 2: One-Shot Canonical Demonstration
- **Objective**: Assess whether a single exemplar (`card_arrival`) improves structural JSON conformity and intent alignment.
- **Dataset Partition**: 500 validation samples (`eval_val`).
- **Input Parameters**:
  - Demonstrations: 1 fixed exemplar from training pool.
  - Prompt: `prompts/one_shot.txt`
- **Output Artifacts**: `results/tables/one_shot_results.csv`

---

## Experiment 3: Static Few-Shot In-Context Learning
- **Objective**: Measure performance gains with five static, domain-spanning exemplars.
- **Dataset Partition**: 500 validation samples (`eval_val`).
- **Input Parameters**:
  - Demonstrations: 5 fixed diverse exemplars (cards, PIN, transfers, deposits).
  - Prompt: `prompts/few_shot.txt`
- **Output Artifacts**: `results/tables/few_shot_results.csv`

---

## Experiment 4: Retrieval-Augmented Dynamic Few-Shot
- **Objective**: Test whether query-adaptive demonstration retrieval via FAISS outperforms static few-shot demonstrations.
- **Dataset Partition**: 500 validation samples (`eval_val`).
- **Input Parameters**:
  - Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`
  - Search Metric: Inner Product (Cosine)
  - Demonstrations: Top-$K$ dynamic exemplars from 9,002 dev pool.
  - Prompt: `prompts/dynamic_few_shot.txt`

---

## Experiment 5: Demonstration Scale (K-Shot Ablation)
- **Objective**: Identify the optimal trade-off between exemplar count, Macro-F1 accuracy, latency, and token overhead.
- **Dataset Partition**: 500 validation samples (`eval_val`).
- **Evaluated Settings**: $K \in \{1, 3, 5, 8, 10\}$.
- **Output Artifacts**:
  - `results/tables/dynamic_few_shot_results.csv`
  - `results/figures/k_vs_performance.png`

---

## Experiment 6: Diversity-Aware Retrieval Comparison
- **Objective**: Evaluate whether enforcing intent diversity among retrieved candidates reduces false analogy errors.
- **Dataset Partition**: 500 validation samples (`eval_val`).
- **Evaluated Settings**:
  - Strategy A: Pure Cosine Nearest Neighbors ($K=5$).
  - Strategy B: Top-20 Candidate Pool with Intent De-duplication ($K=5$).
- **Output Artifacts**: Diversity vs Similarity comparative summary in Notebook 03.

---

## Experiment 7: Prompt Component Ablation Study
- **Objective**: Dissect the marginal impact of individual prompt engineering elements.
- **Evaluated Variants**:
  - **Prompt A**: Role + Task description.
  - **Prompt B**: Role + Task + 77 Intent List.
  - **Prompt C**: Role + Task + Intent List + Demonstrations.
  - **Prompt D**: Role + Task + Intent List + Demonstrations + Disambiguation Rules.
  - **Prompt E**: Role + Task + Intent List + Demonstrations + Rules + JSON Schema.
  - **Prompt F**: Final Optimized Dynamic Prompt.
- **Output Artifacts**:
  - `results/tables/prompt_ablation_results.csv`
  - `prompts/optimized.txt`

---

## Experiment 8: Confidence-Based Routing & Triage
- **Objective**: Measure accuracy gains and escalation workloads across confidence acceptance thresholds.
- **Dataset Partition**: Official test evaluation subset.
- **Thresholds Evaluated**: $\tau \in \{0.60, 0.70, 0.80, 0.90\}$.
- **Measured Metrics**: Coverage rate, review rate, accepted prediction accuracy, overall accuracy.
- **Output Artifacts**: `results/tables/confidence_routing_analysis.csv`

---

## Experiment 9: Cross-Model Parameter Scaling
- **Objective**: Compare intent discrimination, latency, and cost between `openai/gpt-oss-20b` and `openai/gpt-oss-120b`.
- **Dataset Partition**: 500 official test samples.
- **Controlled Setup**: Identical queries, prompts, retrieved exemplars, temperature ($0.0$).
- **Output Artifacts**:
  - `results/tables/model_comparison.csv`
  - `results/figures/model_comparison.png`
