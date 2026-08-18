"""Vertex Model Garden: OpenAPI base URL for publisher/model ids vs per-endpoint path."""

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.llms.vertex_ai.vertex_model_garden.main import (
    _vertex_model_garden_model_id_in_json_body,
    create_vertex_url,
)


@pytest.mark.parametrize(
    "model,expect_openapi_base",
    [
        ("xai/grok-4.1-fast-reasoning", True),
        ("openai/foo/bar", True),
        ("5464397967697903616", False),
        ("gpt-oss-20b-maas", False),
    ],
)
def test_create_vertex_url_openapi_vs_deployed_endpoint(
    model: str, expect_openapi_base: bool
) -> None:
    url = create_vertex_url(
        vertex_location="us-central1",
        vertex_project="my-project",
        stream=False,
        model=model,
    )
    if expect_openapi_base:
        assert "/v1/projects/my-project/locations/us-central1/endpoints/openapi" in url
    else:
        assert (
            "/v1beta1/projects/my-project/locations/us-central1/endpoints/"
            f"{model}" in url
        )
        assert "openapi" not in url


def test_model_id_in_json_body_heuristic() -> None:
    assert (
        _vertex_model_garden_model_id_in_json_body("xai/grok-4.1-fast-reasoning")
        is True
    )
    assert _vertex_model_garden_model_id_in_json_body("5464397967697903616") is False


@pytest.fixture
def _reset_litellm_http_client_cache():
    from litellm import in_memory_llm_clients_cache

    in_memory_llm_clients_cache.flush_cache()
    yield
    in_memory_llm_clients_cache.flush_cache()


@pytest.fixture
def clean_vertex_env():
    saved_env = {}
    env_vars_to_clear = [
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
        "VERTEXAI_PROJECT",
        "VERTEXAI_LOCATION",
        "VERTEXAI_CREDENTIALS",
        "VERTEX_PROJECT",
        "VERTEX_LOCATION",
        "VERTEX_AI_PROJECT",
    ]
    for var in env_vars_to_clear:
        if var in os.environ:
            saved_env[var] = os.environ[var]
            del os.environ[var]

    yield

    for var, value in saved_env.items():
        os.environ[var] = value


def _mock_chat_completion_response(model_in_response: str) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.headers = {}
    response.json.return_value = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": model_in_response,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    return response


async def _invoke_model_garden_completion(
    *,
    model: str,
    api_base,
    mock_response: MagicMock,
):
    """Drive litellm.acompletion through the Vertex Model Garden route and return
    the patched AsyncHTTPHandler so callers can inspect the outbound HTTP call."""
    mock_vertexai = MagicMock()
    mock_vertexai.preview = MagicMock()
    mock_vertexai.preview.language_models = MagicMock()

    with (
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler"
        ) as mock_http_handler,
        patch(
            "litellm.llms.vertex_ai.vertex_model_garden.main.VertexAIModelGardenModels._ensure_access_token",
            return_value=("fake-token", "test-project"),
        ),
        patch.dict(
            sys.modules,
            {"vertexai": mock_vertexai, "vertexai.preview": mock_vertexai.preview},
        ),
    ):
        mock_http_handler.return_value.post = AsyncMock(return_value=mock_response)

        kwargs = dict(
            model=model,
            messages=[{"role": "user", "content": "hello"}],
            vertex_ai_location="us-central1",
            vertex_ai_project="test-project",
        )
        if api_base is not None:
            kwargs["api_base"] = api_base

        await litellm.acompletion(**kwargs)

        return mock_http_handler


@pytest.mark.asyncio
async def test_user_supplied_api_base_passes_through_unchanged(
    clean_vertex_env, _reset_litellm_http_client_cache
):
    """A user-supplied api_base must reach the OpenAI-like handler unchanged,
    with only its own '/chat/completions' suffix appended."""
    user_api_base = "https://my-endpoint.example.com/v1"
    mock_http_handler = await _invoke_model_garden_completion(
        model="vertex_ai/openai/5464397967697903616",
        api_base=user_api_base,
        mock_response=_mock_chat_completion_response("5464397967697903616"),
    )

    mock_http_handler.return_value.post.assert_called_once()
    call_args = mock_http_handler.return_value.post.call_args
    called_url = call_args.kwargs.get("url") or call_args.args[0]
    request_body = json.loads(call_args.kwargs["data"])

    assert called_url == f"{user_api_base}/chat/completions"
    assert ":" not in called_url.replace("https://", "")
    assert "aiplatform.googleapis.com" not in called_url
    assert request_body["model"] == ""


@pytest.mark.asyncio
async def test_user_supplied_api_base_passthrough_for_publisher_model(
    clean_vertex_env, _reset_litellm_http_client_cache
):
    """User-supplied api_base is forwarded unchanged for publisher/catalog
    models too; the publisher model id stays in the JSON body."""
    user_api_base = "https://my-endpoint.example.com/v1"
    mock_http_handler = await _invoke_model_garden_completion(
        model="vertex_ai/openai/xai/grok-4.1-fast-reasoning",
        api_base=user_api_base,
        mock_response=_mock_chat_completion_response("xai/grok-4.1-fast-reasoning"),
    )

    mock_http_handler.return_value.post.assert_called_once()
    call_args = mock_http_handler.return_value.post.call_args
    called_url = call_args.kwargs.get("url") or call_args.args[0]
    request_body = json.loads(call_args.kwargs["data"])

    assert called_url == f"{user_api_base}/chat/completions"
    assert "aiplatform.googleapis.com" not in called_url
    assert request_body["model"] == "xai/grok-4.1-fast-reasoning"


@pytest.mark.asyncio
async def test_default_api_base_when_none_provided_single_segment(
    clean_vertex_env, _reset_litellm_http_client_cache
):
    """With no api_base, single-segment endpoint ids must hit the per-endpoint
    Vertex URL and send an empty model field in the body."""
    mock_http_handler = await _invoke_model_garden_completion(
        model="vertex_ai/openai/5464397967697903616",
        api_base=None,
        mock_response=_mock_chat_completion_response("5464397967697903616"),
    )

    mock_http_handler.return_value.post.assert_called_once()
    call_args = mock_http_handler.return_value.post.call_args
    called_url = call_args.kwargs.get("url") or call_args.args[0]
    request_body = json.loads(call_args.kwargs["data"])

    assert called_url == (
        "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/"
        "test-project/locations/us-central1/endpoints/5464397967697903616/chat/completions"
    )
    assert request_body["model"] == ""


@pytest.mark.asyncio
async def test_default_api_base_when_none_provided_publisher_model(
    clean_vertex_env, _reset_litellm_http_client_cache
):
    """With no api_base, publisher/catalog models must hit the shared OpenAPI
    URL and send the publisher model id in the body."""
    mock_http_handler = await _invoke_model_garden_completion(
        model="vertex_ai/openai/xai/grok-4.1-fast-reasoning",
        api_base=None,
        mock_response=_mock_chat_completion_response("xai/grok-4.1-fast-reasoning"),
    )

    mock_http_handler.return_value.post.assert_called_once()
    call_args = mock_http_handler.return_value.post.call_args
    called_url = call_args.kwargs.get("url") or call_args.args[0]
    request_body = json.loads(call_args.kwargs["data"])

    assert called_url == (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/"
        "test-project/locations/us-central1/endpoints/openapi/chat/completions"
    )
    assert request_body["model"] == "xai/grok-4.1-fast-reasoning"
