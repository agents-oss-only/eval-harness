"""Tests for built-in evaluators."""

import re

import pytest
from eval_harness import ExactMatchEvaluator, ContainsEvaluator, RegexEvaluator


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


class TestRegexEvaluator:
    def test_fixed_pattern_match(self):
        ev = RegexEvaluator(r"\d{4}-\d{2}-\d{2}")
        assert ev("2024-01-15", "") == 1.0

    def test_fixed_pattern_no_match(self):
        ev = RegexEvaluator(r"\d{4}-\d{2}-\d{2}")
        assert ev("January 2024", "") == 0.0

    def test_expected_used_as_pattern_when_no_fixed(self):
        ev = RegexEvaluator()
        assert ev("The answer is 42.", r"\b42\b") == 1.0
        assert ev("The answer is 43.", r"\b42\b") == 0.0

    def test_full_match_true(self):
        ev = RegexEvaluator(r"\d+", full_match=True)
        assert ev("12345", "") == 1.0
        assert ev("12345abc", "") == 0.0

    def test_full_match_false_default(self):
        ev = RegexEvaluator(r"\d+")
        assert ev("abc 123 def", "") == 1.0

    def test_flags_ignorecase(self):
        ev = RegexEvaluator(r"yes|no", flags=re.IGNORECASE)
        assert ev("YES", "") == 1.0
        assert ev("No", "") == 1.0
        assert ev("maybe", "") == 0.0

    def test_flags_applied_with_expected_pattern(self):
        ev = RegexEvaluator(flags=re.IGNORECASE)
        assert ev("Paris", r"paris") == 1.0

    def test_callable_interface(self):
        ev = RegexEvaluator(r"\d+")
        assert ev("abc123", "") == 1.0
        assert ev("abcdef", "") == 0.0
