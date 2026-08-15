"""Answer-quality metrics for the three systems.

Two metrics, both computed against a list of acceptable gold answers:

  accuracy      1 if any gold answer appears anywhere in the generated text.
                This is the metric the Self-RAG paper reports for PopQA: models
                answer in free-form prose, so requiring an exact string match
                would unfairly punish "The play was written by Shakespeare."

  exact_match   1 if the *entire* normalised prediction equals a normalised gold
                answer. Stricter, and reported alongside accuracy for contrast.

Normalisation follows the standard SQuAD recipe: lowercase, strip punctuation,
strip the articles a/an/the, and collapse whitespace.
"""
import re
import string


def normalize_answer(text):
    """Lowercase, remove punctuation and articles, collapse whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def accuracy_score(prediction, gold_answers):
    """1 if any gold answer occurs as a substring of the prediction."""
    pred = normalize_answer(prediction)
    if not pred:
        return 0
    for gold in gold_answers:
        gold_norm = normalize_answer(gold)
        if gold_norm and gold_norm in pred:
            return 1
    return 0


def exact_match_score(prediction, gold_answers):
    """1 if the whole normalised prediction equals a normalised gold answer."""
    pred = normalize_answer(prediction)
    return int(any(pred == normalize_answer(gold) for gold in gold_answers))


def score_prediction(prediction, gold_answers):
    """Return both metrics for a single prediction."""
    return {
        "accuracy": accuracy_score(prediction, gold_answers),
        "exact_match": exact_match_score(prediction, gold_answers),
    }


def aggregate(scores):
    """Average a list of per-example score dicts into overall percentages."""
    if not scores:
        return {"accuracy": 0.0, "exact_match": 0.0, "n": 0}
    n = len(scores)
    return {
        "accuracy": 100.0 * sum(s["accuracy"] for s in scores) / n,
        "exact_match": 100.0 * sum(s["exact_match"] for s in scores) / n,
        "n": n,
    }


if __name__ == "__main__":
    gold = ["William Shakespeare", "Shakespeare"]
    for pred in ["Shakespeare", "The play was written by William Shakespeare [1].", "Dickens"]:
        print(f"{score_prediction(pred, gold)}  <-  {pred!r}")
