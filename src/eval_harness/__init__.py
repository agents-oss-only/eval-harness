"""eval-harness: Lightweight framework for evaluating LLM outputs against golden datasets."""

from .dataset import Dataset, Sample
from .evaluators import ExactMatchEvaluator, ContainsEvaluator, Evaluator, RegexEvaluator
from .runner import run_eval, EvalResult, EvalReport

__all__ = [
    "Dataset",
    "Sample",
    "Evaluator",
    "ExactMatchEvaluator",
    "ContainsEvaluator",
    "RegexEvaluator",
    "run_eval",
    "EvalResult",
    "EvalReport",
]
