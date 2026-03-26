"""Tests for the Dataset class."""

import json
import pytest
from pathlib import Path

from eval_harness import Dataset, Sample


class TestDatasetFromList:
    def test_basic_load(self):
        ds = Dataset.from_list([
            {"input": "What is 2+2?", "expected": "4"},
            {"input": "Capital of France?", "expected": "Paris"},
        ])
        assert len(ds) == 2
        assert ds[0].input == "What is 2+2?"
        assert ds[0].expected == "4"
        assert ds[1].input == "Capital of France?"

    def test_optional_id(self):
        ds = Dataset.from_list([{"id": "q1", "input": "Hi", "expected": "Hello"}])
        assert ds[0].id == "q1"

    def test_no_id_is_none(self):
        ds = Dataset.from_list([{"input": "Hi", "expected": "Hello"}])
        assert ds[0].id is None

    def test_extra_fields_go_to_metadata(self):
        ds = Dataset.from_list([
            {"input": "Hi", "expected": "Hello", "category": "greeting", "weight": 2}
        ])
        assert ds[0].metadata == {"category": "greeting", "weight": 2}

    def test_missing_input_raises(self):
        with pytest.raises(ValueError, match="missing"):
            Dataset.from_list([{"expected": "4"}])

    def test_missing_expected_raises(self):
        with pytest.raises(ValueError, match="missing"):
            Dataset.from_list([{"input": "2+2"}])

    def test_iteration(self):
        items = [{"input": f"q{i}", "expected": str(i)} for i in range(5)]
        ds = Dataset.from_list(items)
        inputs = [s.input for s in ds]
        assert inputs == ["q0", "q1", "q2", "q3", "q4"]


class TestDatasetFromJsonl:
    def test_basic_load(self, tmp_path: Path):
        f = tmp_path / "data.jsonl"
        f.write_text(
            json.dumps({"input": "2+2", "expected": "4"}) + "\n" +
            json.dumps({"input": "3+3", "expected": "6"}) + "\n",
            encoding="utf-8",
        )
        ds = Dataset.from_jsonl(f)
        assert len(ds) == 2
        assert ds[0].input == "2+2"
        assert ds[1].expected == "6"

    def test_blank_lines_ignored(self, tmp_path: Path):
        f = tmp_path / "data.jsonl"
        f.write_text(
            json.dumps({"input": "a", "expected": "b"}) + "\n\n",
            encoding="utf-8",
        )
        ds = Dataset.from_jsonl(f)
        assert len(ds) == 1

    def test_file_not_found_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            Dataset.from_jsonl(tmp_path / "nope.jsonl")

    def test_bad_json_raises(self, tmp_path: Path):
        f = tmp_path / "bad.jsonl"
        f.write_text("not json\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            Dataset.from_jsonl(f)

    def test_missing_input_raises(self, tmp_path: Path):
        f = tmp_path / "bad.jsonl"
        f.write_text(json.dumps({"expected": "4"}) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing required key 'input'"):
            Dataset.from_jsonl(f)

    def test_missing_expected_raises(self, tmp_path: Path):
        f = tmp_path / "bad.jsonl"
        f.write_text(json.dumps({"input": "2+2"}) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing required key 'expected'"):
            Dataset.from_jsonl(f)

    def test_id_field_loaded(self, tmp_path: Path):
        f = tmp_path / "data.jsonl"
        f.write_text(
            json.dumps({"id": "sample-1", "input": "a", "expected": "b"}) + "\n",
            encoding="utf-8",
        )
        ds = Dataset.from_jsonl(f)
        assert ds[0].id == "sample-1"

    def test_numeric_id_coerced_to_str(self, tmp_path: Path):
        f = tmp_path / "data.jsonl"
        f.write_text(
            json.dumps({"id": 42, "input": "a", "expected": "b"}) + "\n",
            encoding="utf-8",
        )
        ds = Dataset.from_jsonl(f)
        assert ds[0].id == "42"
