"""Streamlit demo for the Self-RAG project.

Run from the repo root:

    streamlit run app.py

Ask a question and watch the pipeline's reflection decisions: whether it chose to
retrieve, which passages survived relevance filtering, and how it judged its own
answer. Heavy imports (chromadb, groq, sentence-transformers) happen lazily
inside the query handler, so the page still loads and explains what is missing
if setup is incomplete.
"""
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

SYSTEM_LABELS = {
    "self_rag": "Self-RAG (adaptive + self-critique)",
    "vanilla_rag": "Vanilla RAG (always retrieve)",
    "no_rag": "No-RAG (parametric only)",
}

# Two need lookup, one does not -- so the adaptive Retrieve decision is visible.
EXAMPLE_QUESTIONS = [
    "Who wrote the play Hamlet?",
    "Which engineer designed the Eiffel Tower?",
    "What is 15 plus 27?",
]

REFLECTION_TOKENS = [
    ("Retrieve", "Does this question need external facts?"),
    ("IsRel", "Is each retrieved passage actually relevant?"),
    ("IsSup", "Is the answer supported by those passages?"),
    ("IsUse", "Is the answer useful (rated 1-5)?"),
]


def get_system(name):
    """Import the selected pipeline lazily and return its answer function."""
    if name == "self_rag":
        from self_rag import self_rag_answer

        return self_rag_answer
    if name == "vanilla_rag":
        from vanilla_rag import vanilla_rag_answer

        return vanilla_rag_answer
    from no_rag import no_rag_answer

    return no_rag_answer


def run_query(system_name, question, k):
    """Run one question through a system, returning (answer, info, error)."""
    try:
        system_fn = get_system(system_name)
        answer, info = system_fn(question, k=k)
        return answer, info, None
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"


def render_sidebar():
    """Settings, setup status, and a short explainer. Returns (system, k)."""
    with st.sidebar:
        st.header("Settings")
        system_name = st.selectbox(
            "System",
            list(SYSTEM_LABELS),
            format_func=lambda n: SYSTEM_LABELS[n],
            help="Compare the adaptive pipeline against the two baselines.",
        )
        k = st.slider(
            "Passages to retrieve (k)", 1, 10, 5,
            help="Ignored by No-RAG, which never retrieves.",
        )

        st.divider()
        st.subheader("Setup")
        backend = os.getenv("LLM_BACKEND", "groq")
        key_set = bool(os.getenv("GROQ_API_KEY")) and (
            os.getenv("GROQ_API_KEY") != "your_groq_api_key_here"
        )
        index_built = (ROOT / "chroma_db").exists()

        st.caption(f"LLM backend: `{backend}`")
        if backend == "groq":
            st.caption(("API key found" if key_set else "GROQ_API_KEY missing in .env"))
        st.caption(
            "Index built" if index_built else "Index missing - run `python data/build_index.py`"
        )

        st.divider()
        st.subheader("Reflection tokens")
        for token, meaning in REFLECTION_TOKENS:
            st.caption(f"**{token}** - {meaning}")

    return system_name, k


def render_reflection(info):
    """Show the four reflection decisions the pipeline made, in paper order."""
    st.subheader("Reflection steps")

    retrieved = info.get("retrieved")
    num_retrieved = info.get("num_retrieved", 0)
    used = info.get("passages_used", 0)

    # Step 1 - Retrieve: was external knowledge needed at all?
    if retrieved:
        st.markdown("**1. Retrieve** -> `yes` - the question needs external facts.")
    else:
        st.markdown(
            "**1. Retrieve** -> `no` - answered from the model's own knowledge, "
            "no passages fetched."
        )

    # Step 2 - IsRel: how many retrieved passages survived relevance filtering?
    if retrieved and num_retrieved:
        dropped = num_retrieved - used
        st.markdown(
            f"**2. IsRel** -> kept **{used}** of **{num_retrieved}** retrieved "
            f"passages ({dropped} judged irrelevant and dropped)."
        )
    elif retrieved:
        st.markdown("**2. IsRel** -> no passages were retrieved to filter.")

    # Steps 3 and 4 - grounding and usefulness of the generated answer.
    support = info.get("support")
    usefulness = info.get("usefulness")
    if support is not None or usefulness is not None:
        col_a, col_b = st.columns(2)
        col_a.metric("IsSup (grounding)", str(support))
        col_b.metric("IsUse (usefulness)", f"{usefulness}/5")

    if info.get("regenerated"):
        st.warning(
            "The first answer was unsupported or unhelpful, so it was regenerated "
            "once with strict grounding."
        )
    elif support is not None:
        st.success("The first answer passed both critiques - no regeneration needed.")


def render_sources(info):
    """List the passages actually used, numbered to match the answer's citations."""
    passages = info.get("passages") or []
    if not passages:
        return
    st.subheader("Cited sources")
    for i, p in enumerate(passages, start=1):
        title = p.get("title") or "Untitled"
        score = p.get("score")
        label = f"[{i}] {title}" + (f"  -  similarity {score:.3f}" if score else "")
        with st.expander(label):
            st.write(p.get("text", ""))


def main():
    st.set_page_config(
        page_title="Self-RAG Demo",
        page_icon="*",
        layout="centered",
        # Keep the controls and setup status visible when demoing.
        initial_sidebar_state="expanded",
    )
    system_name, k = render_sidebar()

    st.title("Self-RAG Demo")
    st.caption(
        "Self-Reflective Retrieval-Augmented Generation - Asai et al., ICLR 2024"
    )

    # Example questions fill the input box via session state.
    st.write("Try an example:")
    for col, example in zip(st.columns(len(EXAMPLE_QUESTIONS)), EXAMPLE_QUESTIONS):
        if col.button(example, use_container_width=True):
            st.session_state["question"] = example

    question = st.text_input(
        "Question", key="question", placeholder="Ask anything about the indexed corpus..."
    )
    asked = st.button("Ask", type="primary")

    if asked and question.strip():
        with st.spinner(f"Running {SYSTEM_LABELS[system_name]}..."):
            answer, info, error = run_query(system_name, question.strip(), k)

        if error:
            st.error(error)
            st.info(
                "Setup checklist: `pip install -r requirements.txt`, build the index "
                "with `python data/build_index.py`, and set GROQ_API_KEY in `.env`."
            )
        else:
            st.subheader("Answer")
            st.write(answer)
            render_reflection(info)
            render_sources(info)
    elif asked:
        st.warning("Enter a question first.")


if __name__ == "__main__":
    main()
