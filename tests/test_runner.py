"""Tests for the evaluation runner."""

import pytest
from eval_harness import (
    Dataset,
    ExactMatchEvaluator,
    ContainsEvaluator,
    run_eval,
    EvalReport,
    EvalResult,
)


def _make_dataset(pairs):
    return Dataset.from_list([{"input": inp, "expected": exp} for inp, exp in pairs])


class TestRunEval:
    def test_all_pass(self):
        ds = _make_dataset([("q1", "a1"), ("q2", "a2")])
        model = lambda prompt: {"q1": "a1", "q2": "a2"}[prompt]
        report = run_eval(ds, model, ExactMatchEvaluator())
        assert report.total == 2
        assert report.passed == 2
        assert report.failed == 0
        assert report.pass_rate == 1.0
        assert report.mean_score == 1.0

    def test_all_fail(self):
        ds = _make_dataset([("q1", "a1"), ("q2", "a2")])
        model = lambda prompt: "wrong"
        report = run_eval(ds, model, ExactMatchEvaluator())
        assert report.passed == 0
        assert report.failed == 2
        assert report.pass_rate == 0.0
        assert report.mean_score == 0.0

    def test_partial_pass(self):
        ds = _make_dataset([("q1", "a1"), ("q2", "a2"), ("q3", "a3")])
        answers = {"q1": "a1", "q2": "wrong", "q3": "a3"}
        model = lambda prompt: answers[prompt]
        report = run_eval(ds, model, ExactMatchEvaluator())
        assert report.passed == 2
        assert report.failed == 1
        assert abs(report.pass_rate - 2 / 3) < 1e-9

    def test_contains_evaluator(self):
        ds = _make_dataset([("q1", "Paris"), ("q2", "Berlin")])
        answers = {"q1": "The answer is Paris, France.", "q2": "It's not Berlin."}
        model = lambda prompt: answers[prompt]
        report = run_eval(ds, model, ContainsEvaluator())
        assert report.passed == 2

    def test_pass_threshold(self):
        ds = _make_dataset([("q1", "a1")])
        # Score will be 0.0 since output doesn't match
        model = lambda prompt: "wrong"
        report = run_eval(ds, model, ExactMatchEvaluator(), pass_threshold=0.5)
        assert report.passed == 0

    def test_pass_threshold_partial(self):
        # Use a custom evaluator that gives 0.5 score
        class HalfEvaluator(ExactMatchEvaluator):
            def score(self, output, expected):
                return 0.5

        ds = _make_dataset([("q", "a")])
        report = run_eval(ds, lambda p: p, HalfEvaluator(), pass_threshold=0.5)
        assert report.passed == 1  # 0.5 >= 0.5

        report2 = run_eval(ds, lambda p: p, HalfEvaluator(), pass_threshold=0.6)
        assert report2.passed == 0  # 0.5 < 0.6

    def test_empty_dataset(self):
        ds = _make_dataset([])
        report = run_eval(ds, lambda p: p, ExactMatchEvaluator())
        assert report.total == 0
        assert report.pass_rate == 0.0
        assert report.mean_score == 0.0

    def test_result_fields(self):
        ds = _make_dataset([("hello", "world")])
        model = lambda prompt: "world"
        report = run_eval(ds, model, ExactMatchEvaluator())
        result = report.results[0]
        assert isinstance(result, EvalResult)
        assert result.output == "world"
        assert result.score == 1.0
        assert result.passed is True
        assert result.sample.input == "hello"
        assert result.sample.expected == "world"

    def test_report_repr(self):
        ds = _make_dataset([("q", "a")])
        model = lambda prompt: "a"
        report = run_eval(ds, model, ExactMatchEvaluator())
        r = repr(report)
        assert "total=1" in r
        assert "passed=1" in r
        assert "100.0%" in r
