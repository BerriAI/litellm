"""
Regression tests for LIT-3382: large Gemini batch uploads OOM-restart the proxy.

The OpenAI -> Vertex JSONL transform used to materialize the payload in several
full intermediate lists (decoded str, splitlines list, list of parsed dicts,
list of transformed dicts, joined output). On a ~1 GB upload that tripped the
OOM-killer on the customer's workers. The hot paths now stream entry-by-entry.

These tests lock in three things that would regress if the streaming path were
reverted to the list-based pipeline:
  1. Byte-for-byte output parity with the legacy list pipeline (wire format).
  2. A peak-memory ceiling on the streaming transform, paired with a measurement
     of the legacy list pipeline that exceeds it on the same input.
  3. ``get_object_name`` only parses the first JSONL row (it no longer crashes on
     a payload whose later rows are not valid JSON).
"""

import io
import json
import tracemalloc

import pytest

from litellm.llms.vertex_ai.files.transformation import (
    VertexAIFilesConfig,
    VertexAIJsonlFilesTransformation,
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
    """The pre-fix list-based pipeline, reconstructed for parity comparison."""
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
        # Tail rows are deliberately not valid JSON. The legacy implementation
        # parsed the whole payload and would crash here.
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
    that the legacy list pipeline incurs on the same input. If the hot path is
    reverted to building full intermediate lists, the streaming assertion fails.
    """

    def _measure(self, fn):
        tracemalloc.start()
        try:
            fn()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        return peak

    def test_streaming_peak_well_below_legacy(self):
        cfg = VertexAIFilesConfig()
        raw = _make_openai_jsonl_bytes(8000)
        input_bytes = len(raw)
        content_str = raw.decode("utf-8")

        streaming_peak = self._measure(
            lambda: _stream_openai_jsonl_to_vertex(
                raw, cfg._map_openai_to_vertex_params, as_bytes=True
            )
        )
        legacy_peak = self._measure(
            lambda: _legacy_vertex_jsonl_string(cfg, content_str)
        )

        streaming_amp = streaming_peak / input_bytes
        legacy_amp = legacy_peak / input_bytes

        assert streaming_amp < 5.0, f"streaming peak {streaming_amp:.2f}x too high"
        assert legacy_amp > 6.0, f"legacy peak {legacy_amp:.2f}x unexpectedly low"
        # The streaming path must be a clear, large improvement, not a wash.
        assert streaming_peak < legacy_peak / 2

    def test_get_object_name_does_not_scale_with_payload(self):
        cfg = VertexAIFilesConfig()
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            extract_file_data,
        )

        raw = _make_openai_jsonl_bytes(8000)
        extracted = extract_file_data(("batch.jsonl", raw, "application/jsonl"))

        peak = self._measure(lambda: cfg.get_object_name(extracted, purpose="batch"))
        assert (
            peak / len(raw) < 2.0
        ), "get_object_name should not copy the whole payload"
