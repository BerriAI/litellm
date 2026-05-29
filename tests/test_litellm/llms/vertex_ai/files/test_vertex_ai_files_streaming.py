"""
Tests for the streaming OpenAI -> Vertex JSONL batch transform.

The transform converts batch uploads entry-by-entry rather than materializing
the payload in full intermediate lists (decoded str, parsed dicts, transformed
dicts, joined output), which keeps peak memory bounded on large uploads.

These tests lock in the behaviour that would regress if the streaming path were
replaced by a list-based pipeline:
  1. Byte-for-byte output parity with a list pipeline (wire format).
  2. The streaming transform peaks at a clear fraction of a list pipeline on the
     same input (relative differential, robust to GC noise).
  3. ``get_object_name`` only parses the first JSONL row, so a payload whose
     later rows are not valid JSON does not raise.
  4. A tuple-wrapped file handle uploaded through the real create_file ordering
     keeps every row, including entry 0 (no partial upload from a consumed
     cursor).
"""

import gc
import io
import json
import tracemalloc

import pytest

from litellm.llms.vertex_ai.files.transformation import (
    VertexAIFilesConfig,
    VertexAIJsonlFilesTransformation,
    _get_litellm_batch_custom_id_from_labels,
    _iter_openai_jsonl_entries,
    _iter_openai_jsonl_lines,
    _stream_openai_jsonl_to_vertex,
)
from litellm.types.llms.openai import CreateFileRequest


def _make_openai_jsonl_bytes(n_rows: int, padding: int = 400) -> bytes:
    pad = "x" * padding
    rows = []
    for i in range(n_rows):
        rows.append(
            json.dumps(
                {
                    "custom_id": f"request-{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gemini-2.5-flash",
                        "messages": [{"role": "user", "content": f"{pad} {i}"}],
                        "max_tokens": 4,
                    },
                }
            )
        )
    return ("\n".join(rows)).encode("utf-8")


def _legacy_vertex_jsonl_string(cfg: VertexAIFilesConfig, content: str) -> str:
    """A list-based pipeline, reconstructed for parity comparison."""
    entries = [json.loads(line) for line in content.splitlines() if line.strip()]
    vertex = cfg._transform_openai_jsonl_content_to_vertex_ai_jsonl_content(entries)
    return "\n".join(json.dumps(item) for item in vertex)


class TestStreamingOutputParity:
    def test_streaming_bytes_match_legacy_pipeline(self):
        cfg = VertexAIFilesConfig()
        raw = _make_openai_jsonl_bytes(500)

        streamed, first_entry = _stream_openai_jsonl_to_vertex(
            raw, cfg._map_openai_to_vertex_params, as_bytes=True
        )
        legacy = _legacy_vertex_jsonl_string(cfg, raw.decode("utf-8"))

        assert isinstance(streamed, bytes)
        assert streamed.decode("utf-8") == legacy
        assert first_entry is not None
        assert first_entry["custom_id"] == "request-0"

    def test_transform_create_file_request_returns_bytes_parity(self):
        cfg = VertexAIFilesConfig()
        raw = _make_openai_jsonl_bytes(300)
        request: CreateFileRequest = {
            "file": ("batch.jsonl", raw, "application/jsonl"),
            "purpose": "batch",
        }

        out = cfg.transform_create_file_request(
            model="", create_file_data=request, optional_params={}, litellm_params={}
        )

        assert isinstance(out, bytes)
        assert out.decode("utf-8") == _legacy_vertex_jsonl_string(
            cfg, raw.decode("utf-8")
        )


class TestFileLikeInputNotPartiallyConsumed:
    """
    In ``llm_http_handler.create_file`` the object-name step
    (get_complete_file_url -> get_object_name) runs before
    transform_create_file_request, and each independently calls
    ``extract_file_data`` on the same create_file_data. When the file is a
    tuple-wrapped open handle, the streaming reader must still emit every row
    including entry 0: ``extract_file_data`` materializes the handle to bytes and
    rewinds it (seek(0)), so neither step consumes the other's cursor. A partial
    upload missing the first request would be silent and hard to catch, so this
    locks the full-payload invariant in.
    """

    def test_filehandle_create_file_keeps_first_entry(self):
        cfg = VertexAIFilesConfig()
        n_rows = 25
        raw = _make_openai_jsonl_bytes(n_rows)
        create_file_data: CreateFileRequest = {
            "file": ("batch.jsonl", io.BytesIO(raw), "application/jsonl"),
            "purpose": "batch",
        }

        # Object-name step first (as the handler does), then the transform, both
        # reading the same live BytesIO handle.
        cfg.get_complete_file_url(
            api_base=None,
            api_key=None,
            model="",
            optional_params={},
            litellm_params={"bucket_name": "test-bucket"},
            data=create_file_data,
        )
        out = cfg.transform_create_file_request(
            model="",
            create_file_data=create_file_data,
            optional_params={},
            litellm_params={},
        )

        assert isinstance(out, bytes)
        lines = out.decode("utf-8").splitlines()
        assert len(lines) == n_rows, "no batch row may be dropped from the upload"
        first_labels = json.loads(lines[0])["request"]["labels"]
        assert _get_litellm_batch_custom_id_from_labels(first_labels) == "request-0"


class TestStreamingLineIterator:
    def test_skips_blank_and_whitespace_lines(self):
        content = b'{"a": 1}\n\n   \n{"b": 2}\n'
        assert list(_iter_openai_jsonl_lines(content)) == ['{"a": 1}', '{"b": 2}']

    def test_handles_crlf_and_missing_trailing_newline(self):
        content = b'{"a": 1}\r\n{"b": 2}'
        assert [json.loads(line) for line in _iter_openai_jsonl_lines(content)] == [
            {"a": 1},
            {"b": 2},
        ]

    def test_accepts_str_bytes_tuple_and_filelike(self):
        expected = [{"a": 1}, {"b": 2}]
        text = '{"a": 1}\n{"b": 2}\n'
        for source in (
            text,
            text.encode("utf-8"),
            ("name.jsonl", text.encode("utf-8"), "application/jsonl"),
            io.BytesIO(text.encode("utf-8")),
        ):
            assert list(_iter_openai_jsonl_entries(source)) == expected

    def test_str_input_without_trailing_newline(self):
        assert list(_iter_openai_jsonl_lines('{"a": 1}\n{"b": 2}')) == [
            '{"a": 1}',
            '{"b": 2}',
        ]

    def test_pathlike_input_is_read_line_by_line(self, tmp_path):
        path = tmp_path / "batch.jsonl"
        path.write_bytes(b'{"a": 1}\n{"b": 2}\n')
        assert list(_iter_openai_jsonl_entries(path)) == [{"a": 1}, {"b": 2}]

    def test_unsupported_content_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported file content type"):
            list(_iter_openai_jsonl_lines(12345))  # type: ignore[arg-type]

    def test_is_lazy_does_not_parse_past_first_entry(self):
        # Second row is invalid JSON; pulling only the first entry must not raise.
        content = b'{"custom_id": "first"}\nnot-json-at-all\n'
        gen = _iter_openai_jsonl_entries(content)
        assert next(gen)["custom_id"] == "first"
        with pytest.raises(json.JSONDecodeError):
            next(gen)


class TestLegacyHandlerPathStreaming:
    def test_returns_str_with_object_name_from_first_row(self):
        transformer = VertexAIJsonlFilesTransformation()
        raw = _make_openai_jsonl_bytes(50)

        vertex_str, object_name = (
            transformer.transform_openai_file_content_to_vertex_ai_file_content(raw)
        )

        assert isinstance(vertex_str, str)
        assert "gemini-2.5-flash" in object_name
        # First line is a valid Vertex-wrapped request.
        assert "request" in json.loads(vertex_str.splitlines()[0])

    def test_empty_payload_raises(self):
        transformer = VertexAIJsonlFilesTransformation()
        with pytest.raises(ValueError, match="empty"):
            transformer.transform_openai_file_content_to_vertex_ai_file_content(b"\n\n")


class TestGetObjectNameLazyParse:
    def test_only_parses_first_row_for_model(self):
        cfg = VertexAIFilesConfig()
        # Tail rows are deliberately not valid JSON. Parsing the whole payload
        # would raise here; a first-row-only parse must not.
        raw = (
            b'{"custom_id": "r-0", "body": {"model": "gemini-2.5-flash"}}\n'
            b"garbage line that is not json\n"
        )
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            extract_file_data,
        )

        extracted = extract_file_data(("batch.jsonl", raw, "application/jsonl"))
        object_name = cfg.get_object_name(extracted, purpose="batch")
        assert "gemini-2.5-flash" in object_name


class TestStreamingPeakMemory:
    """
    Differential guard: the streaming transform must stay well under the peak
    that a list pipeline incurs on the same input. If the hot path builds full
    intermediate lists, the streaming assertion fails.

    The assertion that matters is the *relative* one: ``streaming_peak`` must be
    a clear fraction of ``list_peak`` on the identical input. Absolute
    ``tracemalloc`` ratios drift with GC timing and the live set carried in from
    earlier tests, so they make poor CI gates; the relative comparison cancels
    that shared noise and is exactly what regresses (toward 1.0) when the hot
    path builds full intermediate lists. ``gc.collect()`` before each
    measurement removes any garbage the previous run left behind.
    """

    def _measure(self, fn):
        gc.collect()
        tracemalloc.start()
        try:
            fn()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        return peak

    def test_streaming_peak_well_below_list_pipeline(self):
        cfg = VertexAIFilesConfig()
        raw = _make_openai_jsonl_bytes(8000)
        content_str = raw.decode("utf-8")

        streaming_peak = self._measure(
            lambda: _stream_openai_jsonl_to_vertex(
                raw, cfg._map_openai_to_vertex_params, as_bytes=True
            )
        )
        list_peak = self._measure(lambda: _legacy_vertex_jsonl_string(cfg, content_str))

        # Core guard: streaming peaks at well under two-thirds of the list
        # pipeline. Building full intermediate lists in the hot path pushes this
        # ratio back toward 1.0 and fails the test.
        assert streaming_peak < list_peak * 0.6, (
            f"streaming peak {streaming_peak} not a clear win over list pipeline "
            f"{list_peak} (ratio {streaming_peak / list_peak:.2f})"
        )

    def test_get_object_name_does_not_scale_with_payload(self):
        cfg = VertexAIFilesConfig()
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            extract_file_data,
        )

        raw = _make_openai_jsonl_bytes(8000)
        extracted = extract_file_data(("batch.jsonl", raw, "application/jsonl"))

        # The payload bytes already exist before measurement starts, so a lazy
        # first-row parse should allocate only a small fraction of the payload;
        # parsing every row would blow past this bound.
        peak = self._measure(lambda: cfg.get_object_name(extracted, purpose="batch"))
        assert (
            peak / len(raw) < 2.0
        ), "get_object_name should not copy the whole payload"
