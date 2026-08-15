# Self-RAG Project — Semester 5

An inference-time reimplementation of **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection** (Asai et al., ICLR 2024).

The original paper trains a 7B/13B model to emit special *reflection tokens*. That needs GPT-4-distilled training data and serious GPU time. This project reproduces the paper's **behaviour** instead: the same four reflection decisions, implemented as explicit pipeline steps driven by a small instruction-tuned LLM. It runs on **CPU only**, using free tools.

## What it does

Three systems answer the same questions so they can be compared:

| System | Behaviour |
|---|---|
| **No-RAG** | Answers from the model's own (parametric) knowledge. Never retrieves. |
| **Vanilla RAG** | Always retrieves a fixed top-k, then answers. No filtering, no self-check. |
| **Self-RAG** | Decides *whether* to retrieve, drops irrelevant passages, then verifies its own answer. |

## The four reflection signals

| Paper token | Our function | Question it answers |
|---|---|---|
| `Retrieve` | `needs_retrieval` | Does this question need external facts at all? |
| `IsRel` | `is_relevant` | Is each retrieved passage actually relevant? |
| `IsSup` | `is_supported` | Is the answer grounded in those passages? |
| `IsUse` | `usefulness` | Is the answer useful (rated 1–5)? |

## Pipeline

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

## Tech stack (all free, CPU-friendly)

| Layer | Choice |
|---|---|
| LLM (generator **and** critic) | Groq API `llama-3.1-8b-instant` (Ollama fallback for offline) |
| Embeddings | `sentence-transformers` all-MiniLM-L6-v2 (~80 MB) |
| Vector store | ChromaDB (persistent, local) |
| Eval sets | bundled sample set, or PopQA via HuggingFace |
| Demo | Streamlit |

## Setup

```bash
pip install -r requirements.txt
```

Copy the environment template and add a **free** Groq API key from https://console.groq.com/keys:

```bash
cp .env.example .env
```

Then build the vector index (downloads the embedding model on first run):

```bash
python data/build_index.py
```

### Offline alternative

No API key? Install [Ollama](https://ollama.com), pull a small model, and set `LLM_BACKEND=ollama` in `.env`:

```bash
ollama pull phi3:mini
```

## Usage

Run the interactive demo:

```bash
python -m streamlit run app.py
```

Run the evaluation across all three systems, then summarise and plot it:

```bash
python eval/run_eval.py
```

```bash
python eval/stats.py
```

```bash
python eval/plots.py
```

`run_eval.py` accepts `--dataset {sample,popqa}`, `--n`, `--k`, and `--systems`. For example, Self-RAG only on 100 PopQA questions:

```bash
python eval/run_eval.py --dataset popqa --n 100 --systems self_rag
```

> **Note on PopQA:** it asks about long-tail entities that the 20-passage sample corpus does not cover, so scores will be low until you index a larger corpus (e.g. a Wikipedia dump). The bundled `sample` set is aligned with the shipped corpus and is the one to use for a working demo.

## Running the tests

```bash
python -m pytest tests/
```

The pipeline tests inject mock retrievers and generators, so they pass **without** an API key or a built index.

## Project structure

```
data/
  corpus/sample_passages.jsonl   20 passages the retriever searches
  build_index.py                 embeds the corpus into ChromaDB
src/
  config.py        shared constants
  embeddings.py    sentence-transformers wrapper
  retriever.py     top-k dense retrieval
  llm.py           Groq / Ollama chat wrapper
  prompts.py       all prompt templates
  reflect.py       the four reflection functions
  generation.py    cited answer generation
  no_rag.py        baseline: parametric only
  vanilla_rag.py   baseline: always retrieve
  self_rag.py      the Self-RAG orchestrator
eval/
  eval_data.py     sample / PopQA loaders
  metrics.py       accuracy and exact match
  run_eval.py      runs all three systems
  stats.py         retrieval rate, support rate, decision agreement
  plots.py         comparison charts + results table
tests/             sanity and wiring tests
app.py             Streamlit demo
docs/              project plan, paper summary, report
```

Every system returns `(answer, info)` with the same metadata keys — `retrieved`, `num_retrieved`, `passages_used`, `support`, `usefulness`, `regenerated`, `passages` — so the harness and the demo read them all the same way.

## Metrics reported

- **Accuracy** — a gold answer appears in the generated text (the metric the paper uses for PopQA).
- **Exact match** — the whole normalised answer equals a gold answer.
- **Retrieval rate** — how often the system retrieved. Vanilla RAG is 100% by construction; Self-RAG should be lower, which is the paper's adaptive-retrieval claim.
- **Support rate** — how often the critic judged the answer grounded (`IsSup`).
- **Regeneration rate** — how often the strict-grounding retry fired.

## Troubleshooting

| Problem | Fix |
|---|---|
| `GROQ_API_KEY is not set` | Copy `.env.example` to `.env` and paste your key. |
| `Chroma collection ... not found` | Run `python data/build_index.py`. |
| `No module named 'chromadb'` | Run `pip install -r requirements.txt`. |
| Rate-limit errors from Groq | Free tier is limited; re-run with a smaller `--n`. |

## Documentation

- `docs/PROJECT_PLAN.md` — plan, team roles, phase roadmap
- `docs/PAPER_SUMMARY.md` — summary of the original paper
- `docs/REPORT.md` — the write-up, including the results section

## Reference

Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2024). *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.* ICLR 2024. https://arxiv.org/abs/2310.11511
