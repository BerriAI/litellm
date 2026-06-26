"""Unit coverage for the batch-files helpers (no proxy needed, always runs).

These pin the pure logic the e2e guard relies on: decoding the proxy's unified
file id, and building a well-formed OpenAI batch JSONL. If decoding silently
accepted a bogus id, the e2e read-back assertion would be meaningless - so the
reject cases matter as much as the happy path.
"""

from __future__ import annotations

import base64
import json

import pytest

from batch_client import batch_jsonl, parse_unified_file_id


def _encode(decoded: str) -> str:
    return base64.urlsafe_b64encode(decoded.encode("utf-8")).decode("utf-8").rstrip("=")


def test_parse_unified_file_id_extracts_target_and_uri() -> None:
    uri = "https://generativelanguage.googleapis.com/v1beta/files/abc123"
    file_id = _encode(
        "litellm_proxy:application/jsonl;unified_id,xyz;"
        f"target_model_names,gemini-2.5-flash;llm_output_file_id,{uri}"
    )
    parsed = parse_unified_file_id(file_id)
    assert parsed.target_models == ("gemini-2.5-flash",)
    assert parsed.provider_file_uri == uri


def test_parse_unified_file_id_handles_multiple_targets() -> None:
    file_id = _encode(
        "litellm_proxy:application/jsonl;"
        "target_model_names,model-a,model-b;"
        "llm_output_file_id,https://example.com/files/f1"
    )
    assert parse_unified_file_id(file_id).target_models == ("model-a", "model-b")


def test_parse_unified_file_id_rejects_non_managed_id() -> None:
    with pytest.raises(ValueError):
        parse_unified_file_id(_encode("openai-file-abc123"))


def test_parse_unified_file_id_rejects_id_without_provider_uri() -> None:
    with pytest.raises(ValueError):
        parse_unified_file_id(
            _encode(
                "litellm_proxy:application/jsonl;target_model_names,gemini-2.5-flash"
            )
        )


def test_batch_jsonl_emits_one_request_per_line_with_unique_ids() -> None:
    raw = batch_jsonl("gemini-2.5-flash", lines=5)
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == 5
    assert [r["custom_id"] for r in rows] == [f"req-{i}" for i in range(5)]
    assert {r["body"]["model"] for r in rows} == {"gemini-2.5-flash"}
    assert all(r["method"] == "POST" for r in rows)


def test_batch_jsonl_padding_grows_the_file() -> None:
    small = batch_jsonl("m", lines=10)
    padded = batch_jsonl("m", lines=10, pad_bytes=1000)
    assert len(padded) - len(small) >= 10 * 1000
    assert len(batch_jsonl("m", lines=10)) == len(small)
