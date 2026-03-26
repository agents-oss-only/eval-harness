"""Evaluation runner — applies an evaluator over a dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .dataset import Dataset, Sample
from .evaluators import Evaluator


@dataclass
class EvalResult:
    """Result for a single sample evaluation.

    Attributes:
        sample: The original sample.
        output: The LLM-generated output string.
        score: Score returned by the evaluator (0.0–1.0).
        passed: ``True`` if *score* >= *pass_threshold*.
    """

    sample: Sample
    output: str
    score: float
    passed: bool


@dataclass
class EvalReport:
    """Aggregated results for a complete evaluation run.

    Attributes:
        results: Per-sample :class:`EvalResult` objects.
        pass_threshold: The threshold used to determine pass/fail.
    """

    results: List[EvalResult] = field(default_factory=list)
    pass_threshold: float = 1.0

    @property
    def total(self) -> int:
        """Total number of samples evaluated."""
        return len(self.results)

    @property
    def passed(self) -> int:
        """Number of samples that passed."""
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        """Number of samples that failed."""
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        """Fraction of samples that passed (0.0–1.0). Returns 0.0 for empty reports."""
        if self.total == 0:
            return 0.0
        return self.passed / self.total

    @property
    def mean_score(self) -> float:
        """Mean evaluator score across all samples. Returns 0.0 for empty reports."""
        if self.total == 0:
            return 0.0
        return sum(r.score for r in self.results) / self.total

    def __repr__(self) -> str:
        return (
            f"EvalReport(total={self.total}, passed={self.passed}, "
            f"pass_rate={self.pass_rate:.1%}, mean_score={self.mean_score:.3f})"
        )


ModelFn = Callable[[str], str]


def run_eval(
    dataset: Dataset,
    model_fn: ModelFn,
    evaluator: Evaluator,
    *,
    pass_threshold: float = 1.0,
) -> EvalReport:
    """Run an evaluation of *model_fn* over *dataset* using *evaluator*.

    For each sample in *dataset*, calls ``model_fn(sample.input)`` to get the
    model's output, then calls ``evaluator(output, sample.expected)`` to score
    it.

    Args:
        dataset: The :class:`Dataset` of golden samples.
        model_fn: A callable ``(input: str) -> str`` that produces LLM output.
            This can wrap any LLM API — Anthropic, OpenAI, a local model, etc.
        evaluator: An :class:`Evaluator` instance (or any callable with the
            same ``(output, expected) -> float`` signature).
        pass_threshold: Minimum score for a sample to be counted as passing.
            Defaults to ``1.0`` (exact pass only).

    Returns:
        An :class:`EvalReport` with per-sample results and aggregate metrics.

    Example::

        from eval_harness import Dataset, ExactMatchEvaluator, run_eval

        dataset = Dataset.from_jsonl("qa.jsonl")
        report = run_eval(dataset, my_llm, ExactMatchEvaluator())
        print(report)  # EvalReport(total=100, passed=87, pass_rate=87.0%, ...)
    """
    report = EvalReport(pass_threshold=pass_threshold)

    for sample in dataset:
        output = model_fn(sample.input)
        score = evaluator(output, sample.expected)
        passed = score >= pass_threshold
        report.results.append(
            EvalResult(sample=sample, output=output, score=score, passed=passed)
        )

    return report
