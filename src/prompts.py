"""Prompt templates for the four Self-RAG reflection steps.

Each reflection step maps to one of the paper's reflection tokens:

    Retrieve  -> RETRIEVE_*   (do we need to look anything up?)
    IsRel     -> ISREL_*      (is a retrieved passage relevant?)
    IsSup     -> ISSUP_*      (is the answer supported by the passages?)
    IsUse     -> ISUSE_*      (how useful is the answer?)

Templates are formatted with str.format(); keep the {placeholders} intact.
Prompts ask for a single constrained token/word so the output is easy to parse.
"""

# --- Retrieve: decide whether external knowledge is needed -------------------
RETRIEVE_SYSTEM = (
    "You decide whether a question needs external factual lookup. "
    "Answer with a single word only."
)
RETRIEVE_PROMPT = """Does answering the following question require looking up external \
factual knowledge (specific facts, names, dates, events), as opposed to general \
reasoning, opinion, or common sense?

Answer with exactly one word: YES or NO.

Question: {query}
Answer:"""

# --- IsRel: judge passage relevance -----------------------------------------
ISREL_SYSTEM = (
    "You judge whether a passage is relevant to a question. "
    "Answer with a single word only."
)
ISREL_PROMPT = """Question: {query}

Passage: {passage}

Is this passage relevant and useful for answering the question?
Answer with exactly one word: RELEVANT or IRRELEVANT.
Answer:"""

# --- IsSup: judge whether the answer is grounded in the passages ------------
ISSUP_SYSTEM = (
    "You judge whether an answer is supported by the given passages. "
    "Answer with a single word only."
)
ISSUP_PROMPT = """Question: {query}

Passages:
{passages}

Answer: {answer}

How well is the answer supported by the passages above?
Reply with exactly one word:
FULLY (every claim is supported), PARTIALLY (some claims supported), or NO (not supported).
Response:"""

# --- IsUse: rate overall usefulness -----------------------------------------
ISUSE_SYSTEM = (
    "You rate how useful an answer is to a question on a 1-5 scale. "
    "Answer with a single digit only."
)
ISUSE_PROMPT = """Question: {query}

Answer: {answer}

Rate how useful this answer is for the question, from 1 to 5
(5 = directly and completely answers it, 1 = not useful at all).
Respond with a single digit from 1 to 5.
Rating:"""

# --- Answer generation with cited sources -----------------------------------
GEN_SYSTEM = (
    "You are a careful assistant that answers using the provided numbered "
    "sources and cites them inline as [n]."
)
GEN_PROMPT = """Answer the question using the numbered sources below. Cite every \
source you rely on inline, like [1] or [2]. Keep the answer concise (1-3 sentences).
{strictness}
Sources:
{sources}

Question: {query}
Answer:"""

# Inserted into GEN_PROMPT when strict grounding is requested (e.g. on regeneration).
GEN_STRICTNESS = (
    "Use ONLY the sources. If they do not contain the answer, reply exactly: "
    "I don't know.\n"
)
