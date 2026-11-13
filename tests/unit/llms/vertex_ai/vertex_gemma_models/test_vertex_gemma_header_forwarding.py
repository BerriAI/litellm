"""Unit tests verifying caller headers forwarding for Vertex AI Gemma models."""

import sys
from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.llms.vertex_ai.vertex_gemma_models.main import VertexAIGemmaModels

USER_API_BASE = "https://32277599999999999.us-central1-10582012152.prediction.vertexai.goog/v1/projects/PROJECT_ID/locations/us-central1/endpoints/ENDPOINT_ID:predict"

GEMMA_RESPONSE = {
    "deployedModelId": "1207280419999999999",
    "model": "projects/993702345710/locations/us-central1/models/gemma-3-12b-it-1222199011122",
    "modelDisplayName": "gemma-3-12b-it-1222199011122",
    "modelVersionId": "1",
    "predictions": {
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Hello from Gemma!",
                    "role": "assistant",
                },
            }
        ],
        "created": 1739818800,
        "id": "chatcmpl-test-gemma",
        "model": "gemma-3-12b-it-1222199011122",
        "object": "chat.completion",
        "usage": {
            "completion_tokens": 5,
            "prompt_tokens": 10,
            "total_tokens": 15,
        },
    },
}


@pytest.fixture
def gemma_chat_route(respx_mock, monkeypatch):
    """The Gemma route with vertexai import, access token minting and network mocked."""
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    vertexai_stub = MagicMock()
    monkeypatch.setitem(sys.modules, "vertexai", vertexai_stub)
    monkeypatch.setitem(sys.modules, "vertexai.preview", vertexai_stub.preview)
    monkeypatch.setattr(
        VertexAIGemmaModels,
        "_ensure_access_token",
        lambda self, **kwargs: ("fake-gemma-token", "PROJECT_ID"),
    )
    return respx_mock.post(USER_API_BASE)


def _gemma_kwargs(**overrides):
    return {
        "model": "vertex_ai/gemma/gemma-3-12b-it-1222199011122",
        "messages": [{"role": "user", "content": "hello"}],
        "api_base": USER_API_BASE,
        "vertex_project": "PROJECT_ID",
        "vertex_location": "us-central1",
        **overrides,
    }


@pytest.mark.parametrize("header_kwarg", ["extra_headers", "headers"])
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
@pytest.mark.respx(assert_all_called=True)
async def test_gemma_sends_the_callers_headers_with_the_vertex_token(gemma_chat_route, header_kwarg, sync_mode):
    gemma_chat_route.mock(return_value=httpx.Response(200, json=GEMMA_RESPONSE))
    kwargs = _gemma_kwargs(**{header_kwarg: {"X-Custom-Tenant": "tenant-xyz"}})

    response = litellm.completion(**kwargs) if sync_mode else await litellm.acompletion(**kwargs)

    assert response.choices[0].message.content == "Hello from Gemma!"
    sent_headers = gemma_chat_route.calls[0].request.headers
    assert sent_headers["x-custom-tenant"] == "tenant-xyz"
    assert sent_headers["authorization"] == "Bearer fake-gemma-token"
    assert sent_headers["content-type"] == "application/json"


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
@pytest.mark.respx(assert_all_called=True)
async def test_gemma_streaming_sends_the_callers_headers(gemma_chat_route, sync_mode):
    gemma_chat_route.mock(return_value=httpx.Response(200, json=GEMMA_RESPONSE))
    kwargs = _gemma_kwargs(stream=True, extra_headers={"X-Custom-Tenant": "tenant-xyz"})

    if sync_mode:
        text = "".join(chunk.choices[0].delta.content or "" for chunk in litellm.completion(**kwargs))
    else:
        stream = await litellm.acompletion(**kwargs)
        text = "".join([chunk.choices[0].delta.content or "" async for chunk in stream])

    assert text == "Hello from Gemma!"
    sent_headers = gemma_chat_route.calls[0].request.headers
    assert sent_headers["x-custom-tenant"] == "tenant-xyz"
    assert sent_headers["authorization"] == "Bearer fake-gemma-token"
    assert sent_headers["content-type"] == "application/json"


@pytest.mark.asyncio
@pytest.mark.respx(assert_all_called=True)
async def test_gemma_keeps_the_vertex_token_over_a_caller_authorization_header(
    gemma_chat_route,
):
    gemma_chat_route.mock(return_value=httpx.Response(200, json=GEMMA_RESPONSE))

    await litellm.acompletion(**_gemma_kwargs(extra_headers={"Authorization": "Bearer from-caller"}))

    sent_headers = gemma_chat_route.calls[0].request.headers
    assert sent_headers["authorization"] == "Bearer fake-gemma-token"
