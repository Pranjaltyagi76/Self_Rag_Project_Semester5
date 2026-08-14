# Paper Summary — Self-RAG

**Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection**
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi — ICLR 2024.

## Problem

Standard Retrieval-Augmented Generation (RAG) has two weaknesses:

1. **Always retrieves** a fixed number of passages, even when the query needs no external
   knowledge — this injects noise and wastes context.
2. **Never checks** what it retrieved — irrelevant passages get used anyway, and the model
   isn't required to ground its answer in them.

## Core idea: reflection tokens

Self-RAG trains a single LM to emit special **reflection tokens** interleaved with its output,
turning generation into a sequence of decisions:

| Token      | Question it answers                        | Values |
|------------|--------------------------------------------|--------|
| `Retrieve` | Do I need to fetch passages now?           | yes / no / continue |
| `IsRel`    | Is this retrieved passage relevant?        | relevant / irrelevant |
| `IsSup`    | Is my statement supported by the passage?  | fully / partially / no support |
| `IsUse`    | Is the overall answer useful?              | 1–5 |

Because these are just tokens, the model becomes **controllable at inference** — you can tune
thresholds per task (favor grounding for factual QA, loosen it for creative tasks).

## Results

Self-RAG (7B, 13B) reportedly beats ChatGPT and retrieval-augmented Llama2-chat on
open-domain QA, reasoning, and fact verification, with notable gains in **factuality and
citation accuracy** for long-form generation.

## Limitations

- Self-assessment is a learned heuristic, not ground-truth verification.
- Original training depends on GPT-4-distilled critique labels (costly).
- Tree-structured decoding adds inference overhead.

## How our project relates

We reproduce the **behavior** (adaptive retrieval + the four critiques) at inference time using
a small instruction-tuned LLM and prompting, instead of training reflection tokens. This keeps
the project CPU-only and free while faithfully demonstrating the paper's central contribution.
