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
import tempfile
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


def _upload_stream(transformed) -> BaseFileUploadStream:
    """Pull the streaming body out of the upload transform result."""
    return transformed["streaming_media_upload"]["body_stream"]


def _join_upload_body(transformed) -> bytes:
    """Materialize a transform result's upload body for byte-level assertions."""
    if isinstance(transformed, dict) and "streaming_media_upload" in transformed:
        return b"".join(_upload_stream(transformed).iter_bytes())
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
        json.dumps(_openai_batch_jsonl_entry_to_vertex_wrapped_request(entry, cfg._map_openai_to_vertex_params))
        for entry in entries
    )


class TestStreamingOutputParity:
    def test_transform_create_file_request_returns_streaming_body_parity(self):
        cfg = VertexAIFilesConfig()
        raw = _make_openai_jsonl_bytes(300)
        request: CreateFileRequest = {
            "file": ("batch.jsonl", raw, "application/jsonl"),
            "purpose": "batch",
        }

        out = cfg.transform_create_file_request(
            model="", create_file_data=request, optional_params={}, litellm_params={}
        )

        # A batch upload must be a streaming-media config carrying a streaming
        # body, so the handler can stream it to GCS; a buffered bytes/str return
        # would defeat the OOM fix.
        assert isinstance(out, dict) and "streaming_media_upload" in out
        assert isinstance(_upload_stream(out), BaseFileUploadStream)
        assert _join_upload_body(out).decode("utf-8") == _reference_vertex_jsonl_string(cfg, raw.decode("utf-8"))


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

        handle = _NonSeekable(b'{"custom_id": "request-0"}\n{"custom_id": "request-1"}\n')
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
        raw = b'{"custom_id": "r-0", "body": {"model": "gemini-2.5-flash"}}\ngarbage line that is not json\n'
        object_name = cfg.get_object_name(("batch.jsonl", raw, "application/jsonl"), purpose="batch")
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
            for _ in _OpenAIToVertexBatchUploadStream(raw, cfg._map_openai_to_vertex_params).iter_bytes():
                pass

        streaming_peak = self._measure(drain_stream)
        list_peak = self._measure(lambda: _reference_vertex_jsonl_string(cfg, content_str))

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
        assert peak / len(raw) < 2.0, "get_object_name should not copy the whole payload"


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
        assert "uploadType=media" in url

        out = cfg.transform_create_file_request(model="", create_file_data=data, optional_params={}, litellm_params={})
        assert isinstance(out, dict) and "streaming_media_upload" in out
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
            for _ in _upload_stream(out).iter_bytes():
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
        assert peak < len(raw) * 0.3, f"peak {peak} not bounded vs payload {len(raw)} (ratio {peak / len(raw):.2f})"

    def test_path_source_stream_is_reiterable(self, tmp_path):
        cfg = VertexAIFilesConfig()
        path, _ = self._write_jsonl(tmp_path, 50)
        data = self._batch_request(path)

        out = cfg.transform_create_file_request(model="", create_file_data=data, optional_params={}, litellm_params={})
        stream = _upload_stream(out)
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


def _gcs_media_mock(status: int = 200):
    """A fake GCS simple-media endpoint: one request carries the whole object;
    capture the body and headers and return the object resource."""
    state = {"received": bytearray(), "methods": [], "urls": [], "headers": [], "timeouts": []}

    async def handler(request: httpx.Request) -> httpx.Response:
        state["methods"].append(request.method)
        state["urls"].append(str(request.url))
        state["headers"].append(dict(request.headers))
        # httpx records the resolved per-request timeout here, so the test can
        # assert the caller's timeout was forwarded rather than the client default.
        state["timeouts"].append(request.extensions.get("timeout"))
        state["received"].extend(await request.aread())
        return httpx.Response(status, json=_GCS_OBJECT_JSON)

    return handler, state


def _async_handler_with(mock) -> AsyncHTTPHandler:
    handler = AsyncHTTPHandler()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(mock))
    return handler


class TestUploadUrl:
    def test_batch_jsonl_uses_media_upload_type(self):
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
        # A single media upload is one continuous transfer (no per-chunk
        # round-trips), which is what keeps large uploads under client/LB timeouts.
        assert "uploadType=media" in url
        assert "uploadType=resumable" not in url

    def test_binary_upload_uses_media_upload_type(self):
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


class TestUploadStreamBody:
    def test_stream_matches_legacy_pipeline(self):
        cfg = VertexAIFilesConfig()
        raw = _make_openai_jsonl_bytes(120)
        stream = _OpenAIToVertexBatchUploadStream(raw, cfg._map_openai_to_vertex_params)
        assert b"".join(stream.iter_bytes()).decode("utf-8") == _reference_vertex_jsonl_string(cfg, raw.decode("utf-8"))

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
        stream = _OpenAIToVertexBatchUploadStream(io.BytesIO(raw), cfg._map_openai_to_vertex_params)
        first = b"".join(stream.iter_bytes())
        second = b"".join(stream.iter_bytes())
        assert first == second and len(first) > 0


@pytest.mark.asyncio
class TestStreamingMediaUpload:
    """End-to-end against a faked GCS media endpoint. These fail if the handler
    buffers the payload in memory, drops bytes, omits Content-Length (which would
    flip httpx to chunked transfer-encoding), or makes more than one request."""

    async def _run(self, raw: bytes, status: int = 200, timeout=None):
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
        expected = _join_upload_body(transformed)
        mock, state = _gcs_media_mock(status=status)
        response = await BaseLLMHTTPHandler().async_create_file(
            transformed_request=transformed,
            litellm_params={},
            provider_config=cfg,
            headers={"Authorization": "Bearer x"},
            api_base=api_base,
            logging_obj=_logging_obj(),
            client=_async_handler_with(mock),
            timeout=timeout,
        )
        return expected, state, response

    async def test_single_request_carries_whole_payload(self):
        raw = _make_openai_jsonl_bytes(300)
        expected, state, response = await self._run(raw)

        # Exactly one request (the single media upload), and it lands on the
        # media endpoint, not a resumable session.
        assert state["methods"] == ["POST"]
        assert "uploadType=media" in state["urls"][0]

        # The body is streamed with chunked transfer-encoding and no
        # Content-Length, which is what proves it is neither buffered in memory
        # nor staged to a temp file (the disk-exhaustion guard) before sending.
        headers = state["headers"][0]
        assert headers.get("transfer-encoding") == "chunked"
        assert "content-length" not in headers
        # httpx reassembles the chunked body; GCS receives exactly the transform.
        assert bytes(state["received"]) == expected
        assert response.object == "file"

    async def test_failed_upload_raises(self):
        raw = _make_openai_jsonl_bytes(80)
        with pytest.raises(Exception):
            await self._run(raw, status=403)

    async def test_request_timeout_is_forwarded(self):
        # The caller's per-request timeout must reach the GCS upload; every other
        # upload branch forwards it. httpx records the resolved timeout in
        # request.extensions["timeout"]; a dropped timeout would show the client
        # default instead of the value passed here.
        raw = _make_openai_jsonl_bytes(20)
        _, state, _ = await self._run(raw, timeout=httpx.Timeout(137.0))
        forwarded = state["timeouts"][0]
        assert forwarded is not None
        assert forwarded.get("read") == 137.0 and forwarded.get("write") == 137.0

    async def test_upload_does_not_stage_to_disk(self, monkeypatch):
        # Disk-exhaustion guard: the transformed body must stream to GCS, never be
        # written to a temp file first. If any tempfile is created during the
        # upload, an attacker could fill the proxy's temp volume with large
        # concurrent uploads.
        created = []
        real_tempfile = tempfile.TemporaryFile
        monkeypatch.setattr(tempfile, "TemporaryFile", lambda *a, **k: (created.append(1), real_tempfile(*a, **k))[1])
        await self._run(_make_openai_jsonl_bytes(50))
        assert created == []
