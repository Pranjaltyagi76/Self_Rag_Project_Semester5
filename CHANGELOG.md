# Changelog

## v1.0 — Final submission

First complete version of the Self-RAG semester project: an inference-time
reimplementation of Self-RAG (Asai et al., ICLR 2024) that runs on CPU with free
tools.

### Retrieval
- 20-passage sample corpus with a JSONL loader
- all-MiniLM-L6-v2 embeddings, loaded lazily for CPU use
- Persistent ChromaDB index and a top-k dense retriever

### LLM and critic
- Groq (`llama-3.1-8b-instant`) wrapper with an offline Ollama fallback
- Constrained prompt templates for every reflection step
- All four reflection functions: `needs_retrieval` (Retrieve), `is_relevant`
  (IsRel), `is_supported` (IsSup), `usefulness` (IsUse)

### Pipelines
- **No-RAG** — parametric knowledge only
- **Vanilla RAG** — always retrieve top-k
- **Self-RAG** — adaptive retrieval, relevance filtering, support/usefulness
  critique with one strict-grounding regeneration
- One shared metadata schema across all three systems

### Evaluation
- Bundled sample eval set (15 questions needing retrieval, 5 not) and a PopQA loader
- Accuracy and exact-match metrics with SQuAD-style normalisation
- Harness running all three systems, resilient to individual API failures
- Retrieval rate, support rate, regeneration rate, and retrieve-decision agreement
- Comparison charts and a generated results table

### Demo
- Streamlit app showing the answer, each reflection decision, and cited sources
- Sidebar controls, setup status detection, and example questions
- Graceful setup messages instead of tracebacks when dependencies or keys are missing

### Tests
- Retrieval sanity checks
- Vanilla RAG end-to-end wiring tests
- Reflection parsing tests, including a regression for multi-digit usefulness replies

All pipeline tests run without an API key or a built index by injecting mocks.
