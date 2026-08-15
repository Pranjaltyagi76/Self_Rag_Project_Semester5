# Self-RAG: Adaptive Retrieval and Self-Critique
### Semester 5 Project Report

---

## Abstract

Retrieval-Augmented Generation (RAG) reduces factual errors in language models, but standard RAG retrieves a fixed number of passages for *every* query and never checks whether what it retrieved was useful. This project reimplements **Self-RAG** (Asai et al., ICLR 2024) at inference time: the system decides on demand whether retrieval is needed, filters retrieved passages for relevance, and critiques its own answer for grounding and usefulness before returning it. We compare it against a no-retrieval baseline and a conventional always-retrieve RAG pipeline on the same questions.

## 1. Introduction

Large language models answer from parametric knowledge and can state false things confidently. RAG mitigates this by attaching retrieved text to the prompt. Two weaknesses remain:

1. **Indiscriminate retrieval.** Retrieving for a question like "what is 15 plus 27?" adds noise and cost with no benefit.
2. **Unchecked evidence.** Passages are used whether or not they are relevant, and the model is not required to ground its answer in them.

Self-RAG addresses both by making retrieval and self-assessment explicit decisions.

## 2. Background: the original paper

Self-RAG trains a single LM to emit four kinds of **reflection tokens** interleaved with its output:

| Token | Decision | Values |
|---|---|---|
| `Retrieve` | Fetch passages now? | yes / no / continue |
| `IsRel` | Is this passage relevant? | relevant / irrelevant |
| `IsSup` | Is the statement supported? | fully / partially / no |
| `IsUse` | Is the answer useful? | 1–5 |

Because these are ordinary tokens, the model is controllable at inference: thresholds can be tuned per task without retraining. The authors report that Self-RAG (7B/13B) outperforms ChatGPT and retrieval-augmented Llama2-chat on open-domain QA, reasoning, and fact verification, with notable gains in citation accuracy for long-form generation.

See `PAPER_SUMMARY.md` for a fuller summary.

## 3. Method: our reimplementation

Training reflection tokens requires GPT-4-distilled labels and substantial GPU time — out of scope for a semester project on CPU-only hardware. We therefore reproduce the *behaviour* rather than the *training*: each reflection token becomes a short, constrained prompt to a small instruction-tuned LLM, and the pipeline branches on the parsed result.

```
query
  |
  +-- needs_retrieval? --no--> answer from parametric knowledge
        | yes
        v
  retrieve top-k (ChromaDB)
        v
  is_relevant filter --none relevant--> fall back to parametric answer
        | some relevant
        v
  generate grounded answer with [1][2] citations
        v
  is_supported + usefulness --weak--> regenerate once with strict grounding
        v
  answer + reflection metadata
```

**Design choice.** One model plays both generator and critic, matching the paper's "single arbitrary LM" framing. Every reflection prompt asks for a single word or digit, which makes parsing deterministic and cheap.

## 4. Implementation

| Component | Choice | Rationale |
|---|---|---|
| LLM | Groq `llama-3.1-8b-instant` | Free tier, no GPU needed; Ollama fallback for offline use |
| Embeddings | all-MiniLM-L6-v2 | ~80 MB, runs on CPU in milliseconds |
| Vector store | ChromaDB | Pure Python, persists locally, no server |
| Corpus | 20 curated passages | Small enough to inspect by hand; swap in Wikipedia to scale |

All three systems return an identical metadata schema (`retrieved`, `num_retrieved`, `passages_used`, `support`, `usefulness`, `regenerated`, `passages`), so the evaluation harness and the demo treat them uniformly.

## 5. Experimental setup

**Systems compared:** No-RAG, Vanilla RAG (always retrieve top-k), Self-RAG.

**Dataset:** the bundled `sample` set — 20 questions, of which 15 require external facts and 5 (arithmetic, grammar, basic reasoning) do not. The no-retrieval-needed questions are what make the adaptive behaviour measurable; without them every system would retrieve 100% of the time and the paper's central claim would be invisible.

**Metrics:**

- **Accuracy** — a gold answer appears in the generated text (the paper's PopQA metric; free-form answers make strict matching unfair).
- **Exact match** — the whole normalised answer equals a gold answer.
- **Retrieval rate** — the share of questions the system retrieved for.
- **Support rate** — share of judged answers the critic called fully or partially supported.
- **Retrieve-decision agreement** — precision/recall/F1 of the `Retrieve` decision against the human `needs_retrieval` labels. Reported only for Self-RAG, since a fixed policy has nothing to agree about.

## 6. Results

> **These numbers are not yet filled in.** Generate them by running the
> evaluation, then paste the produced table and figures here:
>
> ```
> python eval/run_eval.py
> python eval/plots.py
> ```
>
> `eval/plots.py` writes `docs/RESULTS.md` plus the three figures below.
> Copy that table into this section and delete this note.

### 6.1 Answer quality

| System | Accuracy % | Exact match % |
|---|---|---|
| No-RAG | _to be filled_ | _to be filled_ |
| Vanilla RAG | _to be filled_ | _to be filled_ |
| Self-RAG | _to be filled_ | _to be filled_ |

![Answer quality](fig_accuracy.png)

### 6.2 Retrieval behaviour

The headline comparison: Vanilla RAG retrieves for 100% of questions by construction, whereas Self-RAG should retrieve only when its `Retrieve` decision says external facts are needed.

| System | Retrieval rate % | Avg passages used |
|---|---|---|
| No-RAG | 0.0 | 0.0 |
| Vanilla RAG | 100.0 | _to be filled_ |
| Self-RAG | _to be filled_ | _to be filled_ |

![Retrieval rate](fig_retrieval_rate.png)

### 6.3 Grounding quality

| System | Supported % | Regenerated % |
|---|---|---|
| Vanilla RAG | _to be filled_ | 0.0 |
| Self-RAG | _to be filled_ | _to be filled_ |

![Grounding quality](fig_support.png)

### 6.4 Retrieve-decision agreement

Self-RAG's `Retrieve` decision vs. the human labels: _to be filled_ (precision / recall / F1).

## 7. Discussion

_Fill in after running the evaluation._ Points worth addressing:

- Did Self-RAG match or beat Vanilla RAG's accuracy **while retrieving less often**? That trade-off is the paper's core claim.
- On which questions did the `Retrieve` decision disagree with the human label, and why?
- Did relevance filtering ever discard a passage that was actually needed?
- How often did the strict-grounding regeneration fire, and did it help?

## 8. Limitations

- **Self-assessment is not verification.** The critic is the same model that wrote the answer; a confidently wrong support judgment still passes.
- **Prompted, not trained.** We approximate reflection tokens with prompts, so we do not reproduce the paper's learned token distributions or its tree-structured decoding over multiple passages.
- **Small corpus.** 20 passages make retrieval easy and inflate scores relative to a realistic open-domain setting.
- **Small eval set.** 20 questions give noisy percentages; each question moves a metric by 5 points.
- **Cost of critique.** Self-RAG issues several extra LLM calls per question, so it is slower than vanilla RAG.

## 9. Conclusion

_Fill in after running the evaluation._ Summarise whether adaptive retrieval plus self-critique improved answer quality, reduced unnecessary retrieval, or both, and what the measured trade-offs were.

## References

Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2024). *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.* ICLR 2024. https://arxiv.org/abs/2310.11511

Mallen, A., Asai, A., Zhong, V., Das, R., Khashabi, D., & Hajishirzi, H. (2023). *When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories.* ACL 2023. (PopQA)
