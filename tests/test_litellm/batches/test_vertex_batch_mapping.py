import json
from types import SimpleNamespace

import pytest

from litellm.batches.batch_utils import _get_batch_output_file_content_as_dictionary


class DummyBatch(SimpleNamespace):
    pass


@pytest.mark.asyncio
async def test_should_map_vertex_output_by_key_field(monkeypatch):
    """Ensure Vertex output lines with key/custom_id are mapped directly."""
    input_lines = [
        {"custom_id": "id-1", "body": {"model": "m", "messages": []}},
        {"custom_id": "id-2", "body": {"model": "m", "messages": []}},
    ]
    output_lines = [
        {"key": "id-2", "response": {"status_code": 200}},
        {"key": "id-1", "response": {"status_code": 200}},
    ]

    content_map = {
        "gs://bucket/input.jsonl": "\n".join(
            json.dumps(line) for line in input_lines
        ).encode("utf-8"),
        "gs://bucket/output.jsonl": "\n".join(
            json.dumps(line) for line in output_lines
        ).encode("utf-8"),
    }

    async def fake_afile_content(*, file_id, **_kwargs):
        return SimpleNamespace(content=content_map[file_id])

    monkeypatch.setattr("litellm.files.main.afile_content", fake_afile_content)

    batch = DummyBatch(
        input_file_id="gs://bucket/input.jsonl",
        output_file_id="gs://bucket/output.jsonl",
    )

    result = await _get_batch_output_file_content_as_dictionary(
        batch=batch,
        custom_llm_provider="vertex_ai",
    )

    assert [line.get("custom_id") for line in result] == ["id-2", "id-1"]


@pytest.mark.asyncio
async def test_should_map_vertex_output_by_input_order(monkeypatch):
    """Should error when output lacks key/custom_id fields."""
    input_lines = [
        {"custom_id": "id-1", "body": {"model": "m", "messages": []}},
        {"custom_id": "id-2", "body": {"model": "m", "messages": []}},
    ]
    output_lines = [
        {"response": {"status_code": 200}},
        {"response": {"status_code": 200}},
    ]

    content_map = {
        "gs://bucket/input.jsonl": "\n".join(
            json.dumps(line) for line in input_lines
        ).encode("utf-8"),
        "gs://bucket/output.jsonl": "\n".join(
            json.dumps(line) for line in output_lines
        ).encode("utf-8"),
    }

    async def fake_afile_content(*, file_id, **_kwargs):
        return SimpleNamespace(content=content_map[file_id])

    monkeypatch.setattr("litellm.files.main.afile_content", fake_afile_content)

    batch = DummyBatch(
        input_file_id="gs://bucket/input.jsonl",
        output_file_id="gs://bucket/output.jsonl",
    )

    with pytest.raises(
        ValueError,
        match="Vertex AI batch output is missing custom_id/key",
    ):
        await _get_batch_output_file_content_as_dictionary(
            batch=batch,
            custom_llm_provider="vertex_ai",
        )


@pytest.mark.asyncio
async def test_should_raise_on_vertex_output_count_mismatch(monkeypatch):
    """Guard against mismatched input/output line counts for Vertex."""
    input_lines = [
        {"custom_id": "id-1", "body": {"model": "m", "messages": []}},
        {"custom_id": "id-2", "body": {"model": "m", "messages": []}},
    ]
    output_lines = [
        {"response": {"status_code": 200}},
    ]

    content_map = {
        "gs://bucket/input.jsonl": "\n".join(
            json.dumps(line) for line in input_lines
        ).encode("utf-8"),
        "gs://bucket/output.jsonl": "\n".join(
            json.dumps(line) for line in output_lines
        ).encode("utf-8"),
    }

    async def fake_afile_content(*, file_id, **_kwargs):
        return SimpleNamespace(content=content_map[file_id])

    monkeypatch.setattr("litellm.files.main.afile_content", fake_afile_content)

    batch = DummyBatch(
        input_file_id="gs://bucket/input.jsonl",
        output_file_id="gs://bucket/output.jsonl",
    )

    with pytest.raises(ValueError, match="output line count does not match input"):
        await _get_batch_output_file_content_as_dictionary(
            batch=batch,
            custom_llm_provider="vertex_ai",
        )
