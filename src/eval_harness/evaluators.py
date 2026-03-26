"""Built-in evaluators for comparing LLM outputs against golden answers."""

from __future__ import annotations

import abc
import re
from typing import Optional, Union


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


class RegexEvaluator(Evaluator):
    """Score 1.0 if *output* matches a regex pattern, else 0.0.

    If *pattern* is provided at construction time, it is used for every
    evaluation call and *expected* is ignored.  If *pattern* is ``None``
    (the default), *expected* is compiled and used as the pattern on each
    call — which lets golden datasets store per-sample regex strings.

    Args:
        pattern: Fixed regex pattern to match against *output*.  When
            ``None``, *expected* is used as the pattern instead.
        flags: ``re`` module flags (e.g. ``re.IGNORECASE``).  Defaults to
            ``0`` (no flags).
        full_match: If ``True``, requires the entire *output* to match
            (``re.fullmatch``).  If ``False`` (default), a match anywhere
            in *output* is sufficient (``re.search``).

    Example::

        # Fixed pattern — check output looks like an ISO date
        ev = RegexEvaluator(r"^\\d{4}-\\d{2}-\\d{2}$", full_match=True)
        ev("2024-01-15", "")   # → 1.0
        ev("January 2024", "")  # → 0.0

        # Per-sample patterns stored in the golden dataset
        ev = RegexEvaluator()
        ev("The answer is 42.", r"\\b42\\b")   # → 1.0
        ev("The answer is 43.", r"\\b42\\b")   # → 0.0

        # Case-insensitive search
        ev = RegexEvaluator(r"yes|no", flags=re.IGNORECASE)
        ev("YES", "")  # → 1.0
    """

    def __init__(
        self,
        pattern: Optional[str] = None,
        *,
        flags: Union[int, re.RegexFlag] = 0,
        full_match: bool = False,
    ) -> None:
        self._fixed_pattern = re.compile(pattern, flags) if pattern is not None else None
        self._flags = flags
        self._full_match = full_match

    def score(self, output: str, expected: str) -> float:
        compiled = self._fixed_pattern
        if compiled is None:
            compiled = re.compile(expected, self._flags)
        match_fn = compiled.fullmatch if self._full_match else compiled.search
        return 1.0 if match_fn(output) is not None else 0.0
