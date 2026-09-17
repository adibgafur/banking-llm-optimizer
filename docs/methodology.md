# Experimental Methodology ? BankingLLM-Optimizer

## 1. Central Research Questions

1. **Primary Research Question**:
   > How does prompt strategy affect LLM-based fine-grained banking intent classification, and can retrieval-based dynamic few-shot prompting improve performance without fine-tuning the underlying model?

2. **Secondary Research Questions**:
   - **RQ1 (Demonstration Scale)**: How does Zero-Shot compare with One-Shot, Static Few-Shot, and Dynamic Few-Shot ($K \in \{1, 3, 5, 8, 10\}$)?
   - **RQ2 (Retrieval Quality)**: Does diversity-aware candidate selection outperform naive similarity-only retrieval in intent classification?
   - **RQ3 (Operational Trade-offs)**: What are the Pareto trade-offs between Macro-F1 accuracy, P95 inference latency, prompt token overhead, and monetary cost?
   - **RQ4 (Model Scaling)**: How does parameter scale (`openai/gpt-oss-20b` vs `openai/gpt-oss-120b`) affect fine-grained intent discrimination under identical in-context demonstrations?
   - **RQ5 (Confidence Calibration)**: Can self-reported model confidence be effectively utilized for automated triage routing and escalation?

---

## 2. Benchmark Dataset & Strict Isolation Protocol

We evaluate our systems on **BANKING77** (Casanueva et al., ACL 2020), a domain-specific conversational intent detection benchmark comprising **13,083 customer service queries** annotated across **77 fine-grained banking intents**.

### Data Partitioning Strategy

| Partition | Samples | Proportion | Purpose & Constraints |
|:---|:---:|:---:|:---|
| **Development Exemplar Pool** | 9,002 | 68.8% | Source pool for static exemplars and FAISS dynamic retrieval index. |
| **Validation Benchmark** | 1,001 | 7.6% | Hyperparameter tuning, prompt component ablations, and $K$-shot scaling. |
| **Official Test Benchmark** | 3,080 | 23.6% | **Completely untouched**. Reserved solely for final evaluation reporting. |

### Data Leakage Prevention Assertions
To guarantee scientific validity:
1. **Index Isolation**: FAISS indices and static prompt exemplars are populated exclusively from the 9,002 training pool examples.
2. **Zero Overlap**: Formal set intersection checks confirm:
   $$\text{Set}(\text{IDs}_{\text{train\_pool}}) \cap \text{Set}(\text{IDs}_{\text{val}}) = \emptyset$$
   $$\text{Set}(\text{IDs}_{\text{train\_pool}}) \cap \text{Set}(\text{IDs}_{\text{test}}) = \emptyset$$
3. **No Test Tuning**: All prompt optimizations, rules, and parameter selections ($K$, thresholds) are tuned strictly on the validation set before running test evaluation.

---

## 3. Retrieval Formulation

### Dense Vector Representations
Each text query $x$ is mapped to a dense embedding $e_x \in \mathbb{R}^{384}$ using `sentence-transformers/all-MiniLM-L6-v2`:
$$\hat{e}_x = \frac{e_x}{\|e_x\|_2}$$

### Similarity-Only Retrieval
Given a customer query $q$, the system retrieves the top-$K$ exemplars $\{d_1, \dots, d_K\}$ maximizing cosine similarity via inner product:
$$d^* = \arg\max_{d \in \mathcal{D}_{\text{pool}}} (\hat{e}_q \cdot \hat{e}_d)$$

### Diversity-Aware Selection
To mitigate exemplar redundancy (e.g., retrieving five identical paraphrases of `card_arrival`), we implement a deterministic diversity strategy:
1. Retrieve candidate set $\mathcal{C} = \text{Top-}M(q, \mathcal{D}_{\text{pool}})$ where $M = 20$.
2. Seed selected set $\mathcal{S} = \{c_1\}$ with the highest similarity candidate.
3. For remaining slots $2 \dots K$, select candidates that maximize intent diversity across the candidate set:
   $$c_j \in \mathcal{C} \quad \text{s.t.} \quad \text{Intent}(c_j) \notin \{\text{Intent}(s) \mid s \in \mathcal{S}\}$$
4. If candidate classes are saturated, fill remaining slots by descending similarity.

---

## 4. Evaluation Framework & Metrics

### Multi-Class Classification Metrics
Due to the fine-grained nature of 77 classes, we measure:
- **Overall Accuracy**:
  $$\text{Acc} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(y_i = \hat{y}_i)$$
- **Macro-Averaged F1**: Unweighted mean of F1 across all 77 intents, penalizing failure on rare or difficult classes:
  $$\text{Macro-F1} = \frac{1}{77} \sum_{c=1}^{77} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$
- **Weighted F1**: Class-frequency weighted harmonic mean.
- **Top Confusion Pairs**: Ranked list of class pairs $(c_i, c_j)$ sorted by frequency of misattribution.

### Operational Profiling
- **Inference Latency**: Quantified at the query level; reports Mean, Median (P50), and 95th percentile (P95) latency in milliseconds.
- **Token Accounting**: Captures API input tokens $T_{\text{in}}$ and output tokens $T_{\text{out}}$.
- **Financial Cost Model**:
  $$\text{Cost} = \left(\frac{\sum T_{\text{in}}}{10^6} \times P_{\text{in}}\right) + \left(\frac{\sum T_{\text{out}}}{10^6} \times P_{\text{out}}\right)$$

---

## 5. Confidence Routing Architecture

Let $c \in [0.0, 1.0]$ represent the self-reported confidence score emitted by the LLM and $\tau \in [0.60, 0.90]$ represent the operational threshold:

$$\text{Decision}(q) = \begin{cases} \text{Accept } \hat{y} & \text{if } c \ge \tau \text{ and } \hat{y} \in \mathcal{I}_{77} \\ \text{Escalate to Human Review} & \text{otherwise} \end{cases}$$

We evaluate:
- **Coverage**: Proportion of queries handled automatically ($\frac{|\mathcal{Q}_{\text{accepted}}|}{|\mathcal{Q}_{\text{total}}|}$).
- **Accepted Accuracy**: Precision on automated subset ($\frac{|\mathcal{Q}_{\text{correct}} \cap \mathcal{Q}_{\text{accepted}}|}{|\mathcal{Q}_{\text{accepted}}|}$).
- **Review Rate**: Escalation load directed to human agents ($\frac{|\mathcal{Q}_{\text{review}}|}{|\mathcal{Q}_{\text{total}}|}$).

---

## 6. Threat to Validity & Limitations

1. **Uncalibrated Confidence**: LLMs frequently exhibit overconfidence on out-of-distribution queries. Self-reported scores should be viewed as heuristics rather than calibrated posterior probabilities $P(y \mid x)$.
2. **Monolingual Focus**: BANKING77 is an English-only corpus; cross-lingual transfer requires separate multilingual validation.
3. **Retrieval Distractors**: In dynamic few-shot retrieval, irrelevant nearest neighbors can introduce distractor bias, leading the model to hallucinate false analogies.
