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
import time
import tracemalloc

import httpx
import pytest

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.base_llm.files.transformation import BaseFileUploadStream
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.vertex_ai.files.transformation import (
    VertexAIFilesConfig,
    _OpenAIToVertexBatchUploadStream,
    _get_litellm_batch_custom_id_from_labels,
    _iter_openai_jsonl_entries,
    _iter_openai_jsonl_lines,
    _openai_batch_jsonl_entry_to_vertex_wrapped_request,
)
from litellm.types.llms.openai import CreateFileRequest


def _resumable_stream(transformed) -> BaseFileUploadStream:
    """Pull the streaming body out of a resumable-upload transform result."""
    return transformed["resumable_chunked_upload"]["body_stream"]


def _join_upload_body(transformed) -> bytes:
    """Materialize a transform result's upload body for byte-level assertions."""
    if isinstance(transformed, dict) and "resumable_chunked_upload" in transformed:
        return b"".join(_resumable_stream(transformed).iter_bytes())
    if isinstance(transformed, BaseFileUploadStream):
        return b"".join(transformed.iter_bytes())
    if isinstance(transformed, str):
        return transformed.encode("utf-8")
    return transformed


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


def _reference_vertex_jsonl_string(cfg: VertexAIFilesConfig, content: str) -> str:
    """Row-by-row reference output built eagerly from the live single-entry
    transform, so the streaming path can be checked against it for parity."""
    entries = [json.loads(line) for line in content.splitlines() if line.strip()]
    return "\n".join(
        json.dumps(
            _openai_batch_jsonl_entry_to_vertex_wrapped_request(
                entry, cfg._map_openai_to_vertex_params
            )
        )
        for entry in entries
    )


class TestStreamingOutputParity:
    def test_transform_create_file_request_returns_resumable_stream_parity(self):
        cfg = VertexAIFilesConfig()
        raw = _make_openai_jsonl_bytes(300)
        request: CreateFileRequest = {
            "file": ("batch.jsonl", raw, "application/jsonl"),
            "purpose": "batch",
        }

        out = cfg.transform_create_file_request(
            model="", create_file_data=request, optional_params={}, litellm_params={}
        )

        # A batch upload must be a resumable-upload config carrying a streaming
        # body, so the handler can chunk it; a buffered bytes/str return would
        # defeat the OOM fix.
        assert isinstance(out, dict) and "resumable_chunked_upload" in out
        assert isinstance(_resumable_stream(out), BaseFileUploadStream)
        assert _join_upload_body(out).decode("utf-8") == _reference_vertex_jsonl_string(
            cfg, raw.decode("utf-8")
        )


class TestFileLikeInputNotPartiallyConsumed:
    """
    In ``llm_http_handler.create_file`` the object-name step
    (get_complete_file_url -> get_object_name) runs before
    transform_create_file_request, and both read the same create_file_data
    source. When the file is a tuple-wrapped open handle, the streaming reader
    must still emit every row including entry 0: ``_iter_openai_jsonl_lines``
    rewinds a seekable source (seek(0)) before each pass, so the object-name
    step's partial read of the cursor does not consume the upload. A partial
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
            litellm_params={"gcs_bucket_name": "test-bucket"},
            data=create_file_data,
        )
        out = cfg.transform_create_file_request(
            model="",
            create_file_data=create_file_data,
            optional_params={},
            litellm_params={},
        )

        lines = _join_upload_body(out).decode("utf-8").splitlines()
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

    def test_non_seekable_handle_raises_instead_of_dropping_first_row(self):
        # The handle is read twice (object-name probe, then body). A non-seekable
        # handle can't rewind, so it must fail loudly rather than silently resume
        # mid-stream and omit the opening batch request.
        class _NonSeekable:
            def __init__(self, raw: bytes):
                self._buf = io.BytesIO(raw)

            def read(self, *args):
                return self._buf.read(*args)

            def __iter__(self):
                return iter(self._buf)

            def seek(self, *args):
                raise io.UnsupportedOperation("not seekable")

        handle = _NonSeekable(
            b'{"custom_id": "request-0"}\n{"custom_id": "request-1"}\n'
        )
        with pytest.raises(ValueError, match="seekable"):
            list(_iter_openai_jsonl_lines(handle))

    def test_is_lazy_does_not_parse_past_first_entry(self):
        # Second row is invalid JSON; pulling only the first entry must not raise.
        content = b'{"custom_id": "first"}\nnot-json-at-all\n'
        gen = _iter_openai_jsonl_entries(content)
        assert next(gen)["custom_id"] == "first"
        with pytest.raises(json.JSONDecodeError):
            next(gen)


class TestGetObjectNameLazyParse:
    def test_only_parses_first_row_for_model(self):
        cfg = VertexAIFilesConfig()
        # Tail rows are deliberately not valid JSON. Parsing the whole payload
        # would raise here; a first-row-only parse must not.
        raw = (
            b'{"custom_id": "r-0", "body": {"model": "gemini-2.5-flash"}}\n'
            b"garbage line that is not json\n"
        )
        object_name = cfg.get_object_name(
            ("batch.jsonl", raw, "application/jsonl"), purpose="batch"
        )
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

        def drain_stream():
            # Consume the upload body one row at a time, as the chunked uploader
            # does, without accumulating it.
            for _ in _OpenAIToVertexBatchUploadStream(
                raw, cfg._map_openai_to_vertex_params
            ).iter_bytes():
                pass

        streaming_peak = self._measure(drain_stream)
        list_peak = self._measure(
            lambda: _reference_vertex_jsonl_string(cfg, content_str)
        )

        # Core guard: the lazily consumed streaming body peaks well under a list
        # pipeline that materializes every transformed row. Building full
        # intermediate lists in the hot path pushes this ratio back toward 1.0.
        assert streaming_peak < list_peak * 0.6, (
            f"streaming peak {streaming_peak} not a clear win over list pipeline "
            f"{list_peak} (ratio {streaming_peak / list_peak:.2f})"
        )

    def test_get_object_name_does_not_scale_with_payload(self):
        cfg = VertexAIFilesConfig()
        raw = _make_openai_jsonl_bytes(8000)
        file_data = ("batch.jsonl", raw, "application/jsonl")

        # The payload bytes already exist before measurement starts, so a lazy
        # first-row parse should allocate only a small fraction of the payload;
        # parsing every row would blow past this bound.
        peak = self._measure(lambda: cfg.get_object_name(file_data, purpose="batch"))
        assert (
            peak / len(raw) < 2.0
        ), "get_object_name should not copy the whole payload"


class TestPathSourcedStreaming:
    """
    The proxy spools large batch uploads to a temp file and passes a pathlib.Path
    as the file content instead of pre-reading bytes, so the transform streams
    from disk. These lock in that a Path source yields identical output, keeps
    every row, stays memory-bounded, and is re-iterable (multi-model uploads).
    """

    def _write_jsonl(self, tmp_path, n_rows, padding=400):
        raw = _make_openai_jsonl_bytes(n_rows, padding=padding)
        path = tmp_path / "batch.jsonl"
        path.write_bytes(raw)
        return path, raw

    def _batch_request(self, path) -> CreateFileRequest:
        return {"file": ("batch.jsonl", path, "application/jsonl"), "purpose": "batch"}

    def test_transform_from_path_matches_legacy_and_keeps_all_rows(self, tmp_path):
        cfg = VertexAIFilesConfig()
        n_rows = 200
        path, raw = self._write_jsonl(tmp_path, n_rows)
        data = self._batch_request(path)

        url = cfg.get_complete_file_url(
            api_base=None,
            api_key=None,
            model="",
            optional_params={},
            litellm_params={"gcs_bucket_name": "test-bucket"},
            data=data,
        )
        assert "uploadType=resumable" in url

        out = cfg.transform_create_file_request(
            model="", create_file_data=data, optional_params={}, litellm_params={}
        )
        assert isinstance(out, dict) and "resumable_chunked_upload" in out
        body = _join_upload_body(out).decode("utf-8")
        assert body == _reference_vertex_jsonl_string(cfg, raw.decode("utf-8"))
        lines = body.splitlines()
        assert len(lines) == n_rows, "no batch row may be dropped from a Path source"
        first_labels = json.loads(lines[0])["request"]["labels"]
        assert _get_litellm_batch_custom_id_from_labels(first_labels) == "request-0"

    def test_path_source_peak_stays_below_payload(self, tmp_path):
        cfg = VertexAIFilesConfig()
        path, raw = self._write_jsonl(tmp_path, 8000)
        data = self._batch_request(path)

        def run():
            cfg.get_complete_file_url(
                api_base=None,
                api_key=None,
                model="",
                optional_params={},
                litellm_params={"gcs_bucket_name": "test-bucket"},
                data=data,
            )
            out = cfg.transform_create_file_request(
                model="", create_file_data=data, optional_params={}, litellm_params={}
            )
            for _ in _resumable_stream(out).iter_bytes():
                pass  # drain without accumulating

        gc.collect()
        tracemalloc.start()
        try:
            run()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        # Streaming from disk must not materialize the payload. Reading the whole
        # file into bytes (the pre-fix path) would push peak past the file size.
        assert peak < len(raw) * 0.3, (
            f"peak {peak} not bounded vs payload {len(raw)} "
            f"(ratio {peak / len(raw):.2f})"
        )

    def test_path_source_stream_is_reiterable(self, tmp_path):
        cfg = VertexAIFilesConfig()
        path, _ = self._write_jsonl(tmp_path, 50)
        data = self._batch_request(path)

        out = cfg.transform_create_file_request(
            model="", create_file_data=data, optional_params={}, litellm_params={}
        )
        stream = _resumable_stream(out)
        first = b"".join(stream.iter_bytes())
        second = b"".join(stream.iter_bytes())
        assert first == second and len(first) > 0


_GCS_OBJECT_JSON = {
    "id": "test-bucket/litellm-vertex-files/x/123",
    "name": "litellm-vertex-files/x",
    "size": "0",
    "timeCreated": "2026-01-01T00:00:00.000000Z",
    "purpose": "batch",
}


class _FixedBytesStream(BaseFileUploadStream):
    """Streaming body of exact, controllable bytes for protocol-edge tests."""

    def __init__(self, data: bytes, piece: int = 64):
        self._data = data
        self._piece = piece

    def iter_bytes(self):
        for i in range(0, len(self._data), self._piece):
            yield self._data[i : i + self._piece]


def _logging_obj() -> Logging:
    return Logging(
        model="",
        messages=[],
        stream=False,
        call_type="acreate_file",
        start_time=time.time(),
        litellm_call_id="test",
        function_id="",
    )


def _gcs_resumable_mock(session_url: str, final_status: int = 200):
    """A fake GCS resumable endpoint: POST opens a session (URI in Location),
    each PUT appends and returns 308 until the final chunk returns 200/201."""
    state = {"received": bytearray(), "ranges": [], "methods": [], "urls": []}

    async def handler(request: httpx.Request) -> httpx.Response:
        state["methods"].append(request.method)
        state["urls"].append(str(request.url))
        if request.method == "POST":
            return httpx.Response(200, headers={"location": session_url})
        body = await request.aread()
        content_range = request.headers["content-range"]
        state["ranges"].append(content_range)
        state["received"].extend(body)
        if content_range.rsplit("/", 1)[-1] == "*":
            return httpx.Response(
                308, headers={"range": f"bytes=0-{len(state['received']) - 1}"}
            )
        return httpx.Response(final_status, json=_GCS_OBJECT_JSON)

    return handler, state


def _async_handler_with(mock) -> AsyncHTTPHandler:
    handler = AsyncHTTPHandler()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(mock))
    return handler


class TestResumableUploadUrl:
    def test_batch_jsonl_uses_resumable_upload_type(self):
        cfg = VertexAIFilesConfig()
        request: CreateFileRequest = {
            "file": ("batch.jsonl", _make_openai_jsonl_bytes(3), "application/jsonl"),
            "purpose": "batch",
        }
        url = cfg.get_complete_file_url(
            api_base=None,
            api_key=None,
            model="",
            optional_params={},
            litellm_params={"gcs_bucket_name": "test-bucket"},
            data=request,
        )
        assert "uploadType=resumable" in url
        assert "uploadType=media" not in url

    def test_batch_text_plain_uses_resumable_upload_type(self):
        # Clients often label a .jsonl batch upload as text/plain; it must still
        # take the streaming/resumable path, not the buffered media path.
        cfg = VertexAIFilesConfig()
        request: CreateFileRequest = {
            "file": ("batch.jsonl", _make_openai_jsonl_bytes(3), "text/plain"),
            "purpose": "batch",
        }
        url = cfg.get_complete_file_url(
            api_base=None,
            api_key=None,
            model="",
            optional_params={},
            litellm_params={"gcs_bucket_name": "test-bucket"},
            data=request,
        )
        assert "uploadType=resumable" in url
        assert "uploadType=media" not in url

    def test_binary_upload_stays_simple_media(self):
        cfg = VertexAIFilesConfig()
        request: CreateFileRequest = {
            "file": ("doc.pdf", b"%PDF-1.4 binary", "application/pdf"),
            "purpose": "user_data",
        }
        url = cfg.get_complete_file_url(
            api_base=None,
            api_key=None,
            model="",
            optional_params={},
            litellm_params={"gcs_bucket_name": "test-bucket"},
            data=request,
        )
        assert "uploadType=media" in url
        assert "uploadType=resumable" not in url


class TestResumableStreamBody:
    def test_stream_matches_legacy_pipeline(self):
        cfg = VertexAIFilesConfig()
        raw = _make_openai_jsonl_bytes(120)
        stream = _OpenAIToVertexBatchUploadStream(raw, cfg._map_openai_to_vertex_params)
        assert b"".join(stream.iter_bytes()).decode(
            "utf-8"
        ) == _reference_vertex_jsonl_string(cfg, raw.decode("utf-8"))

    def test_stream_is_reiterable_for_retries(self):
        # A one-shot generator would make a transport retry upload an empty body;
        # iter_bytes() must yield the full payload every call.
        cfg = VertexAIFilesConfig()
        raw = _make_openai_jsonl_bytes(40)
        stream = _OpenAIToVertexBatchUploadStream(raw, cfg._map_openai_to_vertex_params)
        first = b"".join(stream.iter_bytes())
        second = b"".join(stream.iter_bytes())
        assert first == second and len(first) > 0

    def test_stream_is_reiterable_for_seekable_file_like_input(self):
        # A seekable handle (BytesIO, temp file) must be rewound between calls;
        # otherwise the first iter_bytes() exhausts it and a retry would upload
        # an empty body silently.
        cfg = VertexAIFilesConfig()
        raw = _make_openai_jsonl_bytes(40)
        stream = _OpenAIToVertexBatchUploadStream(
            io.BytesIO(raw), cfg._map_openai_to_vertex_params
        )
        first = b"".join(stream.iter_bytes())
        second = b"".join(stream.iter_bytes())
        assert first == second and len(first) > 0


class TestResumableChunking:
    def test_intermediate_chunks_are_exactly_chunk_size(self):
        pieces = list(BaseLLMHTTPHandler._iter_resumable_chunks(iter([b"x" * 10]), 4))
        assert pieces == [b"xxxx", b"xxxx", b"xx"]

    def test_exact_multiple_yields_no_trailing_empty(self):
        # An exactly chunk-aligned stream yields only full chunks; the upload
        # finalizes on the last data chunk instead of an extra empty request.
        pieces = list(BaseLLMHTTPHandler._iter_resumable_chunks(iter([b"x" * 8]), 4))
        assert pieces == [b"xxxx", b"xxxx"]

    def test_empty_stream_yields_nothing(self):
        # A 0-byte stream yields no chunks; the caller finalizes with one empty
        # request (bytes */0).
        assert list(BaseLLMHTTPHandler._iter_resumable_chunks(iter([]), 4)) == []

    def test_default_chunk_size_is_256kib_multiple(self):
        assert BaseLLMHTTPHandler._RESUMABLE_CHUNK_SIZE % (256 * 1024) == 0

    def test_content_range_intermediate_uses_star_total(self):
        assert (
            BaseLLMHTTPHandler._resumable_content_range(0, 4096, is_final=False)
            == "bytes 0-4095/*"
        )

    def test_content_range_final_uses_real_total(self):
        assert (
            BaseLLMHTTPHandler._resumable_content_range(8192, 100, is_final=True)
            == "bytes 8192-8291/8292"
        )

    def test_content_range_empty_finalize(self):
        assert (
            BaseLLMHTTPHandler._resumable_content_range(8192, 0, is_final=True)
            == "bytes */8192"
        )


@pytest.mark.asyncio
class TestResumableUploadProtocol:
    """End-to-end against a faked GCS resumable endpoint. These are the tests
    that fail if the handler buffers the whole body, drops bytes, mislabels a
    Content-Range, follows the 308 instead of continuing, or skips finalize."""

    async def _run(self, raw: bytes, chunk_size: int, final_status: int = 200):
        cfg = VertexAIFilesConfig()
        request: CreateFileRequest = {
            "file": ("batch.jsonl", raw, "application/jsonl"),
            "purpose": "batch",
        }
        api_base = cfg.get_complete_file_url(
            api_base=None,
            api_key=None,
            model="",
            optional_params={},
            litellm_params={"gcs_bucket_name": "test-bucket"},
            data=request,
        )
        transformed = cfg.transform_create_file_request(
            model="", create_file_data=request, optional_params={}, litellm_params={}
        )
        transformed["resumable_chunked_upload"]["chunk_size"] = chunk_size
        expected = _join_upload_body(transformed)

        session_url = "https://storage.googleapis.com/upload/sess?upload_id=SID"
        mock, state = _gcs_resumable_mock(session_url, final_status=final_status)
        response = await BaseLLMHTTPHandler().async_create_file(
            transformed_request=transformed,
            litellm_params={},
            provider_config=cfg,
            headers={"Authorization": "Bearer x"},
            api_base=api_base,
            logging_obj=_logging_obj(),
            client=_async_handler_with(mock),
            timeout=None,
        )
        return expected, state, response, session_url, api_base

    async def test_streams_in_chunks_and_reassembles(self):
        raw = _make_openai_jsonl_bytes(300)
        chunk_size = 4096
        expected, state, response, session_url, api_base = await self._run(
            raw, chunk_size
        )

        # One session-open POST, then a sequence of chunk PUTs.
        assert state["methods"][0] == "POST"
        assert set(state["methods"][1:]) == {"PUT"}
        assert state["methods"].count("PUT") >= 2, "payload must span multiple chunks"

        # POST opens a resumable session; every chunk goes to the session URI.
        assert "uploadType=resumable" in state["urls"][0]
        assert all(u == session_url for u in state["urls"][1:])

        # Every non-final chunk is exactly chunk_size with an unknown-total range;
        # the final chunk carries the real total.
        intermediate = state["ranges"][:-1]
        for index, content_range in enumerate(intermediate):
            assert (
                content_range
                == f"bytes {index * chunk_size}-{(index + 1) * chunk_size - 1}/*"
            )
        total = len(expected)
        last_offset = len(intermediate) * chunk_size
        if last_offset == total:  # payload landed on a chunk boundary
            assert state["ranges"][-1] == f"bytes */{total}"
        else:
            assert state["ranges"][-1] == f"bytes {last_offset}-{total - 1}/{total}"

        # The bytes GCS received are exactly the transformed batch payload.
        assert bytes(state["received"]) == expected
        assert response.object == "file"

    async def test_exact_multiple_finalizes_on_last_data_chunk(self):
        # A body that is an exact multiple of the chunk size finalizes on its
        # last data chunk (bytes (TOTAL-chunk)-(TOTAL-1)/TOTAL), with no extra
        # empty finalize request.
        chunk_size = 256
        total = chunk_size * 3
        stream = _FixedBytesStream(b"a" * total)
        config = {"body_stream": stream, "chunk_size": chunk_size}
        session_url = "https://storage.googleapis.com/upload/sess?upload_id=SID"
        mock, state = _gcs_resumable_mock(session_url)

        response = await BaseLLMHTTPHandler()._aresumable_chunked_upload(
            client=_async_handler_with(mock),
            initiate_url="https://storage.googleapis.com/upload?uploadType=resumable",
            base_headers={"Authorization": "Bearer x"},
            config=config,
            timeout=None,
        )

        assert state["ranges"][-1] == f"bytes {total - chunk_size}-{total - 1}/{total}"
        assert "*" not in state["ranges"][-1]
        assert bytes(state["received"]) == b"a" * total
        assert response.status_code == 200

    async def test_failed_chunk_raises(self):
        raw = _make_openai_jsonl_bytes(80)
        with pytest.raises(Exception):
            await self._run(raw, chunk_size=4096, final_status=403)
