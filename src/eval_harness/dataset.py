"""Dataset loading for golden evaluation sets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class Sample:
    """A single evaluation sample.

    Attributes:
        input: The prompt or input string to pass to the LLM.
        expected: The expected/golden output string.
        id: Optional identifier for the sample.
        metadata: Optional extra fields from the source file.
    """

    input: str
    expected: str
    id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Dataset:
    """A collection of evaluation samples loaded from a JSONL file.

    Each line in the file must be a JSON object with at least ``input`` and
    ``expected`` fields. An optional ``id`` field is used as the sample
    identifier. All other fields are stored in ``metadata``.

    Args:
        path: Path to a ``.jsonl`` file.

    Example::

        dataset = Dataset.from_jsonl("benchmarks/qa.jsonl")
        for sample in dataset:
            output = my_llm(sample.input)
            ...
    """

    def __init__(self, samples: List[Sample]) -> None:
        self._samples = samples

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "Dataset":
        """Load a dataset from a JSONL file.

        Args:
            path: Path to the JSONL file. Each line must contain ``input``
                and ``expected`` keys.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If a line is missing required keys or is not valid JSON.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        samples: List[Sample] = []
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {lineno}: {exc}") from exc

                if "input" not in obj:
                    raise ValueError(f"Line {lineno} missing required key 'input'")
                if "expected" not in obj:
                    raise ValueError(f"Line {lineno} missing required key 'expected'")

                sample_id = obj.pop("id", None)
                known_keys = {"input", "expected"}
                metadata = {k: v for k, v in obj.items() if k not in known_keys}

                samples.append(
                    Sample(
                        input=obj["input"],
                        expected=obj["expected"],
                        id=str(sample_id) if sample_id is not None else None,
                        metadata=metadata,
                    )
                )

        return cls(samples)

    @classmethod
    def from_list(cls, items: List[Dict[str, Any]]) -> "Dataset":
        """Create a dataset from a list of dicts (useful for testing).

        Args:
            items: List of dicts, each with ``input`` and ``expected`` keys.
        """
        samples = []
        for i, obj in enumerate(items):
            if "input" not in obj or "expected" not in obj:
                raise ValueError(f"Item {i} missing 'input' or 'expected' key")
            known_keys = {"input", "expected", "id"}
            metadata = {k: v for k, v in obj.items() if k not in known_keys}
            samples.append(
                Sample(
                    input=obj["input"],
                    expected=obj["expected"],
                    id=obj.get("id"),
                    metadata=metadata,
                )
            )
        return cls(samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._samples)

    def __getitem__(self, index: int) -> Sample:
        return self._samples[index]
