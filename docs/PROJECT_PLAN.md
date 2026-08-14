# Project Plan — Self-RAG (Semester 5)

## Goal

Reimplement Self-RAG at inference time (adaptive retrieval + self-critique) and show it
improves over vanilla RAG and a no-retrieval baseline. CPU-only, free tools, team of mixed
Python skill.

## Systems compared

1. **No-RAG** — LLM answers from parametric knowledge only.
2. **Vanilla RAG** — always retrieve top-k passages, then answer.
3. **Self-RAG** — decide *whether* to retrieve, filter irrelevant passages, verify support
   and usefulness, regenerate once if weak.

## Pipeline (Self-RAG)

```
Query -> [needs_retrieval?] --no--> answer from parametric knowledge
             | yes
             v
      retrieve top-k (ChromaDB)
             v
      [is_relevant] filter passages
             v
      generate grounded answer + citations
             v
      [is_supported] + [usefulness] -> accept / regenerate once
```

## Tech stack

| Layer            | Choice                                   |
|------------------|------------------------------------------|
| LLM (gen+critic) | Groq API `llama-3.1-8b-instant` (Ollama fallback) |
| Embeddings       | sentence-transformers all-MiniLM-L6-v2   |
| Vector store     | ChromaDB                                 |
| Eval set         | PopQA subset (~500 Q)                     |
| Demo             | Streamlit                                |

## Team roles

| Role | Owner | Files | Deliverable |
|------|-------|-------|-------------|
| A — Retrieval     | TBD | `data/build_index.py`, `src/retriever.py` | `retrieve(query) -> passages` |
| B — LLM / Critic  | TBD | `src/llm.py`, `src/prompts.py`, `src/reflect.py` | 4 reflection functions |
| C — Orchestration | TBD | `src/self_rag.py`, `src/vanilla_rag.py` | both pipelines |
| D — Eval + Demo   | TBD | `eval/run_eval.py`, `eval/plots.py`, `app.py` | metrics + UI |

## Phase roadmap & commits

| Phase | Focus            | Commits |
|-------|------------------|---------|
| 0 | Setup                | 4  |
| 1 | Retrieval            | 5  |
| 2 | LLM + critic         | 6  |
| 3 | Vanilla RAG          | 3  |
| 4 | Self-RAG             | 5  |
| 5 | Evaluation           | 5  |
| 6 | Demo                 | 3  |
| 7 | Docs / release       | 4  |
|   | **Total**            | **35** |

## Evaluation metrics

- Answer accuracy (exact-match / substring) across the three systems.
- **Retrieval rate** — Self-RAG should retrieve only when needed (< 100%).
- Support / citation accuracy.

## Timeline (~8 weeks)

1. Wk 1: read paper, setup, build index
2. Wk 2–3: retriever + reflection functions + vanilla RAG
3. Wk 4–5: Self-RAG orchestrator + integration
4. Wk 6: evaluation + charts
5. Wk 7: Streamlit demo
6. Wk 8: report + slides + buffer
