"""
Regression tests for https://github.com/BerriAI/litellm/issues/36493

/v1/images/edits on the openai path silently dropped unknown provider params
(e.g. seed) and the extra_body escape hatch, unlike /v1/images/generations.
"""

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

PNG_BYTES = b"\x89PNG\r\n\x1a\nfakepng"


def _capture_image_edit_request(captured):
    def respond(request):
        captured["content_type"] = request.headers.get("content-type")
        captured["body"] = request.content
        return httpx.Response(200, json={"created": 1712697600, "data": [{"b64_json": "aW1n"}]})

    return respond


def _multipart_text_fields(content_type: str, body: bytes) -> dict:
    boundary = content_type.split("boundary=")[1].encode()
    return {
        part.split(b'name="')[1].split(b'"')[0].decode(): part.partition(b"\r\n\r\n")[2].rstrip(b"\r\n-").decode()
        for part in body.split(b"--" + boundary)
        if b'name="' in part and b"filename=" not in part
    }


def test_image_edit_forwards_provider_params_and_extra_body():
    captured = {}
    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_capture_image_edit_request(captured))))

    response = litellm.image_edit(
        model="openai/gpt-image-1",
        image=PNG_BYTES,
        prompt="add a hat",
        api_key="sk-test",
        api_base="https://edit.example/v1",
        client=client,
        seed=42,
        extra_body={"quality_level": "high"},
    )

    assert captured["content_type"].startswith("multipart/form-data")
    fields = _multipart_text_fields(captured["content_type"], captured["body"])
    assert fields["seed"] == "42"
    assert fields["quality_level"] == "high"
    assert "extra_body" not in fields
    assert fields["model"] == "gpt-image-1"
    assert fields["prompt"] == "add a hat"
    assert b'name="image[]"' in captured["body"]
    assert response.data


def test_image_edit_extra_body_takes_precedence_over_kwargs():
    captured = {}
    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_capture_image_edit_request(captured))))

    litellm.image_edit(
        model="openai/gpt-image-1",
        image=PNG_BYTES,
        prompt="add a hat",
        api_key="sk-test",
        api_base="https://edit.example/v1",
        client=client,
        seed=42,
        extra_body={"seed": 7},
    )

    assert _multipart_text_fields(captured["content_type"], captured["body"])["seed"] == "7"


@pytest.mark.asyncio
async def test_aimage_edit_forwards_extra_body():
    """aimage_edit used to drop extra_headers/extra_query/extra_body when
    building its partial, so they never reached image_edit."""
    captured = {}
    client = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(_capture_image_edit_request(captured)))

    response = await litellm.aimage_edit(
        model="openai/gpt-image-1",
        image=PNG_BYTES,
        prompt="add a hat",
        api_key="sk-test",
        api_base="https://edit.example/v1",
        client=client,
        seed=42,
        extra_body={"quality_level": "high"},
    )

    fields = _multipart_text_fields(captured["content_type"], captured["body"])
    assert fields["seed"] == "42"
    assert fields["quality_level"] == "high"
    assert "extra_body" not in fields
    assert response.data
