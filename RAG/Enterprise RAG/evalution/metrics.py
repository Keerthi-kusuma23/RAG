# evaluation/metrics.py
from typing import List

def exact_match(predictions: List[str], references: List[str]) -> float:
    """
    Calculate exact match score for QA predictions.
    """
    correct = sum([pred.strip() == ref.strip() for pred, ref in zip(predictions, references)])
    return correct / len(predictions) if predictions else 0.0

def f1_score(predictions: List[str], references: List[str]) -> float:
    """
    Calculate F1 score for QA predictions.
    """
    def f1(pred, ref):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        common = set(pred_tokens) & set(ref_tokens)
        if not common:
            return 0.0
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        return 2 * (precision * recall) / (precision + recall)

    scores = [f1(p, r) for p, r in zip(predictions, references)]
    return sum(scores) / len(scores) if scores else 0.0
