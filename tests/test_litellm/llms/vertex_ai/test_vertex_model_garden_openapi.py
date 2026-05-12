"""Vertex Model Garden: OpenAPI base URL for publisher/model ids vs per-endpoint path."""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.llms.vertex_ai.common_utils import (
    VertexAIModelRoute,
    get_vertex_ai_model_route,
)
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.llms.vertex_ai.vertex_model_garden.main import (
    _vertex_model_garden_model_id_in_json_body,
    create_vertex_url,
)

GROK_MODELS = [
    "xai/grok-4.1-fast-non-reasoning",
    "xai/grok-4.1-fast-reasoning",
    "xai/grok-4.20-non-reasoning",
    "xai/grok-4.20-reasoning",
]


@pytest.fixture(autouse=True)
def _use_local_model_cost_map():
    previous_value = os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP")
    previous_model_cost = litellm.model_cost
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    yield
    if previous_value is None:
        os.environ.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
    else:
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = previous_value
    litellm.model_cost = previous_model_cost


@pytest.fixture(autouse=True)
def _reset_litellm_http_client_cache():
    litellm.in_memory_llm_clients_cache.flush_cache()


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


@pytest.mark.parametrize("model", GROK_MODELS)
def test_vertex_grok_models_route_to_model_garden(model: str) -> None:
    assert get_vertex_ai_model_route(model=model) == VertexAIModelRoute.MODEL_GARDEN


@pytest.mark.parametrize("model", GROK_MODELS)
def test_vertex_grok_models_resolve_global_region(model: str) -> None:
    vertex_base = VertexBase()
    assert vertex_base.get_vertex_region(vertex_region=None, model=model) == "global"
    assert (
        vertex_base.get_vertex_region(vertex_region="us-central1", model=model)
        == "global"
    )


@pytest.mark.parametrize("model", GROK_MODELS)
def test_create_vertex_url_uses_global_openapi_endpoint_for_grok(model: str) -> None:
    url = create_vertex_url(
        vertex_location="global",
        vertex_project="my-project",
        stream=False,
        model=model,
    )

    assert url == (
        "https://aiplatform.googleapis.com/v1/projects/my-project/"
        "locations/global/endpoints/openapi"
    )


@pytest.mark.parametrize(
    "model,expected_context,expected_reasoning,expected_input_cost,expected_output_cost,expected_cache_read_cost",
    [
        (
            "vertex_ai/xai/grok-4.1-fast-non-reasoning",
            128000,
            None,
            2e-07,
            5e-07,
            5e-08,
        ),
        (
            "vertex_ai/xai/grok-4.1-fast-reasoning",
            128000,
            True,
            2e-07,
            5e-07,
            5e-08,
        ),
        (
            "vertex_ai/xai/grok-4.20-non-reasoning",
            200000,
            None,
            2e-06,
            6e-06,
            2e-07,
        ),
        (
            "vertex_ai/xai/grok-4.20-reasoning",
            200000,
            True,
            2e-06,
            6e-06,
            2e-07,
        ),
    ],
)
def test_vertex_grok_model_metadata(
    model: str,
    expected_context: int,
    expected_reasoning: bool | None,
    expected_input_cost: float,
    expected_output_cost: float,
    expected_cache_read_cost: float,
) -> None:
    model_info = litellm.get_model_info(model=model, custom_llm_provider="vertex_ai")

    assert model_info["litellm_provider"] == "vertex_ai-xai_models"
    assert model_info["mode"] == "chat"
    assert litellm.model_cost[model]["supported_regions"] == ["global"]
    assert litellm.utils.get_supported_regions(
        model=model,
        custom_llm_provider="vertex_ai",
    ) == ["global"]
    assert model_info["max_input_tokens"] == expected_context
    assert model_info["max_output_tokens"] == expected_context
    assert model_info["max_tokens"] == expected_context
    assert model_info["input_cost_per_token"] == expected_input_cost
    assert model_info["output_cost_per_token"] == expected_output_cost
    assert model_info["cache_read_input_token_cost"] == expected_cache_read_cost
    assert model_info.get("supports_function_calling") is True
    assert model_info.get("supports_response_schema") is True
    assert model_info.get("supports_tool_choice") is True
    assert model_info.get("supports_vision") is True
    assert model_info.get("supports_reasoning") is expected_reasoning
    assert model_info.get("supports_web_search") is None
    assert model_info.get("supports_low_reasoning_effort") is None


def _mock_vertexai_module() -> MagicMock:
    mock_vertexai = MagicMock()
    mock_vertexai.preview = MagicMock()
    mock_vertexai.preview.language_models = MagicMock()
    return mock_vertexai


def _mock_chat_completion_response(model: str) -> MagicMock:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {}
    mock_response.json.return_value = {
        "id": "chatcmpl-grok-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello from Vertex Grok.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
    }
    return mock_response


def _get_mock_post_call_fields(call_args) -> tuple[str, dict, dict]:
    called_url = call_args.args[0]
    request_body = json.loads(call_args.kwargs["data"])
    headers = call_args.kwargs["headers"]
    return called_url, request_body, headers


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    ["xai/grok-4.1-fast-reasoning", "xai/grok-4.20-non-reasoning"],
)
async def test_vertex_grok_completion_request_uses_global_openapi_endpoint(
    model: str,
) -> None:
    mock_vertexai = _mock_vertexai_module()
    mock_response = _mock_chat_completion_response(model=model)

    with (
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler"
        ) as mock_http_handler,
        patch(
            "litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini.VertexLLM._ensure_access_token",
            return_value=("fake-token", "test-project"),
        ),
        patch.dict(
            "sys.modules",
            {"vertexai": mock_vertexai, "vertexai.preview": mock_vertexai.preview},
        ),
    ):
        mock_http_handler.return_value.post = AsyncMock(return_value=mock_response)

        response = await litellm.acompletion(
            model=f"vertex_ai/{model}",
            messages=[{"role": "user", "content": "Hello"}],
            vertex_ai_project="test-project",
        )

        mock_http_handler.return_value.post.assert_called_once()
        call_args = mock_http_handler.return_value.post.call_args
        called_url, request_body, headers = _get_mock_post_call_fields(call_args)

        assert called_url == (
            "https://aiplatform.googleapis.com/v1/projects/test-project/"
            "locations/global/endpoints/openapi/chat/completions"
        )
        assert request_body == {
            "model": model,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }
        assert headers["Authorization"] == "Bearer fake-token"
        assert response.model == f"vertex_ai/{model}"


@pytest.mark.asyncio
async def test_vertex_grok_streaming_request_uses_global_openapi_endpoint() -> None:
    model = "xai/grok-4.1-fast-reasoning"
    mock_vertexai = _mock_vertexai_module()
    mock_response = _mock_chat_completion_response(model=model)
    mock_response.aiter_lines.return_value = iter([])

    with (
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler"
        ) as mock_http_handler,
        patch(
            "litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini.VertexLLM._ensure_access_token",
            return_value=("fake-token", "test-project"),
        ),
        patch.dict(
            "sys.modules",
            {"vertexai": mock_vertexai, "vertexai.preview": mock_vertexai.preview},
        ),
    ):
        mock_http_handler.return_value.post = AsyncMock(return_value=mock_response)

        await litellm.acompletion(
            model=f"vertex_ai/{model}",
            messages=[{"role": "user", "content": "Hello"}],
            vertex_ai_project="test-project",
            stream=True,
        )

        mock_http_handler.return_value.post.assert_called_once()
        call_args = mock_http_handler.return_value.post.call_args
        called_url, request_body, _ = _get_mock_post_call_fields(call_args)

        assert called_url == (
            "https://aiplatform.googleapis.com/v1/projects/test-project/"
            "locations/global/endpoints/openapi/chat/completions"
        )
        assert request_body["model"] == model
        assert request_body["stream"] is True
        assert call_args.kwargs["stream"] is True
