"""Tests for built-in evaluators."""

import pytest
from eval_harness import ExactMatchEvaluator, ContainsEvaluator


class TestExactMatchEvaluator:
    def test_exact_match_scores_1(self):
        ev = ExactMatchEvaluator()
        assert ev("Paris", "Paris") == 1.0

    def test_mismatch_scores_0(self):
        ev = ExactMatchEvaluator()
        assert ev("Lyon", "Paris") == 0.0

    def test_strip_enabled_by_default(self):
        ev = ExactMatchEvaluator()
        assert ev("  Paris  ", "Paris") == 1.0

    def test_strip_disabled(self):
        ev = ExactMatchEvaluator(strip=False)
        assert ev("  Paris  ", "Paris") == 0.0

    def test_case_sensitive_by_default(self):
        ev = ExactMatchEvaluator()
        assert ev("paris", "Paris") == 0.0

    def test_case_insensitive(self):
        ev = ExactMatchEvaluator(case_sensitive=False)
        assert ev("paris", "Paris") == 1.0

    def test_callable_interface(self):
        ev = ExactMatchEvaluator()
        assert ev("a", "a") == 1.0
        assert ev("a", "b") == 0.0


class TestContainsEvaluator:
    def test_contains_scores_1(self):
        ev = ContainsEvaluator()
        assert ev("The capital of France is Paris.", "Paris") == 1.0

    def test_not_contains_scores_0(self):
        ev = ContainsEvaluator()
        assert ev("The capital of France is Lyon.", "Paris") == 0.0

    def test_strip_enabled_by_default(self):
        ev = ContainsEvaluator()
        assert ev("Result: 42", "  42  ") == 1.0

    def test_strip_disabled(self):
        ev = ContainsEvaluator(strip=False)
        assert ev("Result: 42", "  42  ") == 0.0

    def test_case_sensitive_by_default(self):
        ev = ContainsEvaluator()
        assert ev("The answer is paris.", "Paris") == 0.0

    def test_case_insensitive(self):
        ev = ContainsEvaluator(case_sensitive=False)
        assert ev("The answer is paris.", "Paris") == 1.0

    def test_exact_output_also_passes(self):
        ev = ContainsEvaluator()
        assert ev("Paris", "Paris") == 1.0

    def test_callable_interface(self):
        ev = ContainsEvaluator()
        assert ev("hello world", "world") == 1.0
