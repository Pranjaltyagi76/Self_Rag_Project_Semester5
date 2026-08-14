# Self-RAG Project — Semester 5

An inference-time reimplementation of **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection** (Asai et al., ICLR 2024).

Instead of training reflection tokens into a model, this project reproduces the paper's core ideas — **adaptive retrieval** (decide *when* to retrieve) and **self-critique** (grade the passages and the model's own answer) — as an explicit pipeline driven by a small instruction-tuned LLM. Runs on **CPU only** using free tools.

## What we build

Three systems compared on the same questions:

1. **No-RAG** — the LLM alone (parametric knowledge only)
2. **Vanilla RAG** — always retrieve a fixed top-k
3. **Self-RAG** — retrieve only when needed, filter irrelevant passages, and verify the answer is supported

## The four reflection signals

| Paper token | Our step | Purpose |
|-------------|----------|---------|
| `Retrieve` | `needs_retrieval` | Skip retrieval when the model already knows |
| `IsRel`    | `is_relevant`     | Drop irrelevant retrieved passages |
| `IsSup`    | `is_supported`    | Check the answer is grounded in the passages |
| `IsUse`    | `usefulness`      | Score overall answer usefulness (regenerate if weak) |

## Tech stack (all free, CPU-friendly)

- **LLM:** Groq API (`llama-3.1-8b-instant`), with an offline Ollama fallback
- **Embeddings:** `sentence-transformers` all-MiniLM-L6-v2
- **Vector store:** ChromaDB
- **Eval:** PopQA subset
- **Demo:** Streamlit

## Project structure

```
data/        corpus + index building
src/         llm, retriever, reflection, pipelines
eval/        evaluation harness + plots
docs/        project plan + paper summary
app.py       Streamlit demo
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then add your free Groq API key
```

## Team

Semester 5 college project. See `docs/PROJECT_PLAN.md` for the full plan and role split.
