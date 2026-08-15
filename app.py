"""Streamlit demo for the Self-RAG project.

Run from the repo root:

    streamlit run app.py

Ask a question and see the answer produced by the selected system. Heavy imports
(chromadb, groq, sentence-transformers) happen lazily inside the query handler,
so the page still loads and explains what is missing if setup is incomplete.
"""
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


def main():
    st.set_page_config(page_title="Self-RAG Demo", page_icon="*", layout="centered")
    st.title("Self-RAG Demo")
    st.caption(
        "Self-Reflective Retrieval-Augmented Generation - Asai et al., ICLR 2024"
    )

    system_name = st.selectbox(
        "System",
        list(SYSTEM_LABELS),
        format_func=lambda n: SYSTEM_LABELS[n],
    )
    k = st.slider("Passages to retrieve (k)", 1, 10, 5)
    question = st.text_input("Question", placeholder="Who wrote the play Hamlet?")

    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Thinking..."):
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
            st.session_state["last_info"] = info


if __name__ == "__main__":
    main()
