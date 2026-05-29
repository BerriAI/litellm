"""Tests for streaming (temp-file) Vertex batch file uploads.

Covers the no-copy JSONL line iteration, the temp-file transform, the
``transform_create_file_request`` contract returning a ``StreamingFileUploadBody``,
and the HTTP handler streaming that temp file and cleaning it up.
"""

import json
import os
import tempfile
from types import SimpleNamespace

import httpx
import pytest

from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.vertex_ai.files import transformation as t
from litellm.llms.vertex_ai.files.transformation import VertexAIFilesConfig
from litellm.types.files import StreamingFileUploadBody


def _map(_body):
    return {}


def _batch_rows(n):
    rows = [
        {
            "custom_id": f"r-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gemini-2.5-flash",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4,
            },
        }
        for i in range(n)
    ]
    return ("\n".join(json.dumps(r) for r in rows) + "\n").encode("utf-8")


def test_iter_nonempty_lines_bytes_and_str_skip_blanks():
    raw = b'{"a":1}\n\n  \n{"b":2}\n'
    lines = list(t._iter_nonempty_jsonl_lines_from_content(raw))
    assert lines == ['{"a":1}', '{"b":2}']
    assert list(t._iter_nonempty_jsonl_lines_from_content(raw.decode())) == lines


def test_first_entry_from_content_skips_blanks():
    raw = b'\n  \n{"custom_id":"x","body":{"model":"m"}}\n{"custom_id":"y"}\n'
    first = t._first_jsonl_entry_from_content(raw)
    assert first["custom_id"] == "x"


def test_stream_to_tempfile_roundtrip_and_size():
    content = _batch_rows(4)
    path, size, first = t._stream_openai_jsonl_to_vertex_tempfile(content, _map)
    try:
        assert first["custom_id"] == "r-0"
        with open(path, "rb") as f:
            on_disk = f.read()
        assert len(on_disk) == size
        out_lines = on_disk.decode("utf-8").split("\n")
        assert len(out_lines) == 4
        for ln in out_lines:
            d = json.loads(ln)
            assert "request" in d
            assert d["request"]["labels"]["litellm_custom_id"].startswith("r-")
    finally:
        t._safe_remove_file(path)
    assert not os.path.exists(path)


def test_transform_create_file_request_returns_streaming_body():
    cfg = VertexAIFilesConfig()
    create_file_data = {
        "file": ("batch.jsonl", _batch_rows(1), "application/jsonl"),
        "purpose": "batch",
    }
    result = cfg.transform_create_file_request(
        model="",
        create_file_data=create_file_data,
        optional_params={},
        litellm_params={},
    )
    assert isinstance(result, StreamingFileUploadBody)
    try:
        assert result.size > 0
        assert os.path.exists(result.path)
        assert result.content_type == "application/jsonl"
    finally:
        t._safe_remove_file(result.path)


def test_empty_batch_raises_and_cleans_up():
    cfg = VertexAIFilesConfig()
    create_file_data = {
        "file": ("batch.jsonl", b"\n   \n", "application/jsonl"),
        "purpose": "batch",
    }
    with pytest.raises(ValueError):
        cfg.transform_create_file_request(
            model="",
            create_file_data=create_file_data,
            optional_params={},
            litellm_params={},
        )


class _FakeRawClient:
    """Mimics httpx.AsyncClient.build_request/send, draining the streamed body."""

    def __init__(self):
        self.received = b""

    def build_request(self, method, url, headers=None, content=None, timeout=None):
        return SimpleNamespace(
            method=method, url=url, headers=headers, _content=content
        )

    async def send(self, request):
        async for chunk in request._content:
            self.received += chunk
        return httpx.Response(
            status_code=200,
            json={
                "id": "bucket/obj/123",
                "name": "obj",
                "size": str(len(self.received)),
                "timeCreated": "2026-05-29T00:00:00Z",
            },
            request=httpx.Request("POST", request.url),
        )


@pytest.mark.asyncio
async def test_async_streaming_upload_consumes_body_and_cleans_up():
    handler = BaseLLMHTTPHandler()
    raw = _FakeRawClient()
    async_client = SimpleNamespace(client=raw)

    fd, path = tempfile.mkstemp(suffix=".jsonl")
    payload = b'{"request": {"contents": []}}\n{"request": {"contents": []}}'
    with os.fdopen(fd, "wb") as f:
        f.write(payload)
    body = StreamingFileUploadBody(
        path=path, size=len(payload), content_type="application/jsonl"
    )

    resp = await handler._stream_file_upload_async(
        streaming_body=body,
        async_httpx_client=async_client,
        api_base="https://storage.googleapis.com/upload",
        headers={"Authorization": "Bearer x"},
        provider_config=VertexAIFilesConfig(),
        timeout=60.0,
    )

    assert resp.status_code == 200
    assert raw.received == payload  # full body streamed through unchanged
    assert not os.path.exists(path)  # temp file cleaned up
