"""
Streaming JSONL transform regression tests for LIT-3382.

The original (pre-fix) pipeline materialised the OpenAI batch JSONL payload at
least 5 times concurrently (decoded str, splitlines list, list-of-dicts,
list-of-vertex-dicts, joined output string). On ~1 GiB uploads the pod OOM'd
and uvicorn workers restarted mid-upload.

The fix replaces the list-based pipeline with a streaming generator
(``_iter_openai_jsonl_entries`` + ``_stream_openai_jsonl_to_vertex_jsonl_string``)
and peeks only the first entry inside ``VertexAIFilesConfig.get_object_name``.

These tests assert:
1. Output parity with the legacy list-based path on a representative payload.
2. Every supported ``FileTypes`` input form (str, bytes, tuple, file-like,
   PathLike) is honored by the streaming helpers.
3. Edge cases (empty input, single entry, CRLF line endings, blank lines,
   trailing newlines) match the legacy semantics.
4. Peak traced memory under a sizeable synthetic payload is bounded well
   below the old ~5x amplification (we assert <= 3.5x).
"""
import gc
import io
import json
import os
import sys
import tempfile
import tracemalloc
from pathlib import Path

import pytest

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")),
)

from litellm.llms.vertex_ai.files.transformation import (  # noqa: E402
    VertexAIFilesConfig,
    VertexAIJsonlFilesTransformation,
    _iter_openai_file_lines,
    _iter_openai_jsonl_entries,
    _stream_openai_jsonl_to_vertex_jsonl_string,
)
from litellm.types.llms.openai import CreateFileRequest  # noqa: E402


def _entry(i: int) -> dict:
    return {
        "custom_id": f"req-{i}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gemini-1.5-flash",
            "messages": [{"role": "user", "content": f"hello {i}"}],
            "max_tokens": 32,
        },
    }


def _jsonl_bytes(n: int, line_sep: str = "\n") -> bytes:
    return line_sep.join(json.dumps(_entry(i)) for i in range(n)).encode("utf-8")


class TestIterOpenAIFileLines:
    def test_yields_lines_from_bytes(self):
        # Mirrors str.splitlines() semantics: trailing "\n" does not yield an
        # extra empty trailing line.
        assert list(_iter_openai_file_lines(b"a\nb\nc\n")) == ["a", "b", "c"]

    def test_yields_lines_from_str(self):
        assert list(_iter_openai_file_lines("a\nb\nc")) == ["a", "b", "c"]

    def test_strips_crlf(self):
        assert list(_iter_openai_file_lines(b"a\r\nb\r\nc")) == ["a", "b", "c"]

    def test_unwraps_tuple_form(self):
        payload = ("batch.jsonl", b"a\nb", "application/jsonl")
        assert list(_iter_openai_file_lines(payload)) == ["a", "b"]

    def test_reads_file_like_bytes(self):
        bio = io.BytesIO(b"a\nb\nc\n")
        assert list(_iter_openai_file_lines(bio)) == ["a", "b", "c"]

    def test_reads_file_like_text(self):
        sio = io.StringIO("a\nb\nc\n")
        assert list(_iter_openai_file_lines(sio)) == ["a", "b", "c"]

    def test_reads_pathlike(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as fh:
            fh.write("a\nb\nc\n")
            path = fh.name
        try:
            assert list(_iter_openai_file_lines(Path(path))) == ["a", "b", "c"]
        finally:
            os.unlink(path)

    def test_empty_input(self):
        assert list(_iter_openai_file_lines(b"")) == []
        assert list(_iter_openai_file_lines("")) == []

    def test_no_trailing_newline(self):
        assert list(_iter_openai_file_lines(b"only")) == ["only"]


class TestIterOpenAIJsonlEntries:
    def test_yields_parsed_entries(self):
        payload = _jsonl_bytes(3)
        assert list(_iter_openai_jsonl_entries(payload)) == [
            _entry(0),
            _entry(1),
            _entry(2),
        ]

    def test_skips_blank_and_whitespace_lines(self):
        payload = (
            json.dumps(_entry(0)) + "\n\n   \n" + json.dumps(_entry(1)) + "\n"
        ).encode("utf-8")
        assert list(_iter_openai_jsonl_entries(payload)) == [_entry(0), _entry(1)]

    def test_handles_crlf(self):
        payload = (
            json.dumps(_entry(0)) + "\r\n" + json.dumps(_entry(1)) + "\r\n"
        ).encode("utf-8")
        assert list(_iter_openai_jsonl_entries(payload)) == [_entry(0), _entry(1)]

    def test_empty_payload(self):
        assert list(_iter_openai_jsonl_entries(b"")) == []

    def test_lazy_parsing_does_not_consume_beyond_first(self):
        """``next()`` must not parse any line beyond the first."""
        payload = b"\n".join(
            [json.dumps(_entry(0)).encode(), b"NOT-JSON", b"NOT-JSON"]
        )
        first = next(_iter_openai_jsonl_entries(payload))
        assert first == _entry(0)


class TestStreamOutputParity:
    def test_matches_legacy_pipeline(self):
        payload = _jsonl_bytes(20)
        xform = VertexAIJsonlFilesTransformation()
        new_str, _first = _stream_openai_jsonl_to_vertex_jsonl_string(
            payload, xform._map_openai_to_vertex_params
        )
        legacy_list = [
            json.loads(line)
            for line in payload.decode().splitlines()
            if line.strip()
        ]
        legacy_vertex = (
            xform._transform_openai_jsonl_content_to_vertex_ai_jsonl_content(
                legacy_list
            )
        )
        legacy_str = "\n".join(json.dumps(item) for item in legacy_vertex)
        assert new_str == legacy_str

    def test_returns_first_entry(self):
        payload = _jsonl_bytes(3)
        xform = VertexAIJsonlFilesTransformation()
        _out, first = _stream_openai_jsonl_to_vertex_jsonl_string(
            payload, xform._map_openai_to_vertex_params
        )
        assert first == _entry(0)

    def test_empty_payload_yields_empty_string_and_none(self):
        xform = VertexAIJsonlFilesTransformation()
        out, first = _stream_openai_jsonl_to_vertex_jsonl_string(
            b"", xform._map_openai_to_vertex_params
        )
        assert out == ""
        assert first is None


class TestPublicApiParity:
    def test_jsonl_files_transformation_output(self):
        payload = _jsonl_bytes(5)
        xform = VertexAIJsonlFilesTransformation()
        out_str, object_name = (
            xform.transform_openai_file_content_to_vertex_ai_file_content(
                openai_file_content=payload
            )
        )
        assert out_str.count("\n") == 4
        parsed = [json.loads(line) for line in out_str.split("\n")]
        for i, item in enumerate(parsed):
            assert "request" in item
            assert item["request"]["labels"]["litellm_custom_id"] == f"req-{i}"
        assert "publishers/google/models" in object_name
        assert "gemini" in object_name.lower()

    def test_files_config_create_file_request_output(self):
        payload = _jsonl_bytes(5)
        cfg = VertexAIFilesConfig()
        req = CreateFileRequest(
            file=("batch.jsonl", payload, "application/jsonl"),
            purpose="batch",
        )
        out = cfg.transform_create_file_request(
            model="",
            create_file_data=req,
            optional_params={},
            litellm_params={},
        )
        assert isinstance(out, str)
        assert out.count("\n") == 4
        parsed = [json.loads(line) for line in out.split("\n")]
        assert all("request" in item for item in parsed)

    def test_get_object_name_peeks_only_first_entry(self):
        """``get_object_name(batch)`` must not parse every line."""
        payload = _jsonl_bytes(3) + b"\nNOT-JSON-LINE\n"
        cfg = VertexAIFilesConfig()
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            extract_file_data,
        )
        extracted = extract_file_data(
            ("batch.jsonl", payload, "application/jsonl")
        )
        name = cfg.get_object_name(extracted, "batch")
        assert "publishers/google/models" in name

    def test_get_object_name_handles_empty_jsonl(self):
        cfg = VertexAIFilesConfig()
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            extract_file_data,
        )
        extracted = extract_file_data(("batch.jsonl", b"", "application/jsonl"))
        name = cfg.get_object_name(extracted, "batch")
        assert "/uploads/" in name


class TestPeakMemory:
    """Assert peak traced memory through the streaming pipeline stays well
    under the legacy ~5x amplification."""

    def _build_payload(self, num_entries: int) -> bytes:
        rows = []
        for i in range(num_entries):
            rows.append(
                json.dumps(
                    {
                        "custom_id": f"req-{i}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": "gemini-1.5-flash",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "Summarize: "
                                    + ("Lorem ipsum dolor sit amet. " * 30),
                                }
                            ],
                            "max_tokens": 32,
                        },
                    }
                )
            )
        return ("\n".join(rows)).encode("utf-8")

    def test_streaming_keeps_peak_below_legacy_amplification(self):
        payload = self._build_payload(8_000)

        gc.collect()
        tracemalloc.start()
        xform = VertexAIJsonlFilesTransformation()
        out, _name = (
            xform.transform_openai_file_content_to_vertex_ai_file_content(
                openai_file_content=payload
            )
        )
        _cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        amplification = peak / len(payload)
        assert amplification < 3.5, (
            f"Streaming transform peak memory amplification ({amplification:.2f}x) "
            f"exceeded the safety ceiling (3.5x). Input={len(payload)} bytes, "
            f"peak={peak} bytes, output={len(out)} bytes."
        )

    def test_get_object_name_peak_is_bounded(self):
        payload = self._build_payload(8_000)

        cfg = VertexAIFilesConfig()
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            extract_file_data,
        )
        extracted = extract_file_data(
            ("batch.jsonl", payload, "application/jsonl")
        )

        gc.collect()
        tracemalloc.start()
        _name = cfg.get_object_name(extracted, "batch")
        _cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        amplification = peak / len(payload)
        assert amplification < 1.5, (
            f"get_object_name peak memory amplification ({amplification:.2f}x) "
            f"is too high; expected near 1x since only the first entry is parsed."
        )
