"""Built-in evaluators for comparing LLM outputs against golden answers."""

from __future__ import annotations

import abc


class Evaluator(abc.ABC):
    """Abstract base class for output evaluators.

    Subclass this to implement custom comparison logic. The ``__call__``
    interface lets evaluators be used as plain callables.

    Example::

        class StartsWithEvaluator(Evaluator):
            def score(self, output: str, expected: str) -> float:
                return 1.0 if output.startswith(expected) else 0.0
    """

    @abc.abstractmethod
    def score(self, output: str, expected: str) -> float:
        """Return a score in [0.0, 1.0] comparing *output* to *expected*.

        Args:
            output: The actual LLM output string.
            expected: The golden/expected answer string.

        Returns:
            A float in ``[0.0, 1.0]`` where ``1.0`` is a perfect match.
        """

    def __call__(self, output: str, expected: str) -> float:
        return self.score(output, expected)


class ExactMatchEvaluator(Evaluator):
    """Score 1.0 if *output* exactly equals *expected*, else 0.0.

    Args:
        strip: If ``True`` (default), strip leading/trailing whitespace
            from both strings before comparing.
        case_sensitive: If ``True`` (default), comparison is case-sensitive.

    Example::

        ev = ExactMatchEvaluator()
        ev("Paris", "Paris")   # → 1.0
        ev("paris", "Paris")   # → 0.0
        ev = ExactMatchEvaluator(case_sensitive=False)
        ev("paris", "Paris")   # → 1.0
    """

    def __init__(self, *, strip: bool = True, case_sensitive: bool = True) -> None:
        self.strip = strip
        self.case_sensitive = case_sensitive

    def score(self, output: str, expected: str) -> float:
        a, b = output, expected
        if self.strip:
            a, b = a.strip(), b.strip()
        if not self.case_sensitive:
            a, b = a.lower(), b.lower()
        return 1.0 if a == b else 0.0


class ContainsEvaluator(Evaluator):
    """Score 1.0 if *expected* appears anywhere inside *output*, else 0.0.

    Useful for checking that a key phrase or answer token is present without
    requiring an exact full-string match.

    Args:
        case_sensitive: If ``True`` (default), comparison is case-sensitive.
        strip: If ``True`` (default), strip whitespace from *expected* before
            checking containment.

    Example::

        ev = ContainsEvaluator()
        ev("The capital of France is Paris.", "Paris")  # → 1.0
        ev("The capital of France is Lyon.", "Paris")   # → 0.0
    """

    def __init__(self, *, strip: bool = True, case_sensitive: bool = True) -> None:
        self.strip = strip
        self.case_sensitive = case_sensitive

    def score(self, output: str, expected: str) -> float:
        a, b = output, expected
        if self.strip:
            b = b.strip()
        if not self.case_sensitive:
            a, b = a.lower(), b.lower()
        return 1.0 if b in a else 0.0
