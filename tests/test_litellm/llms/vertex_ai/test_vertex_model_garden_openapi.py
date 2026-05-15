"""Vertex Model Garden: OpenAPI base URL for publisher/model ids vs per-endpoint path."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from litellm.llms.vertex_ai.vertex_model_garden.main import (
    VertexAIModelGardenModels,
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


def _mock_vertexai_module() -> MagicMock:
    mock_vertexai = MagicMock()
    mock_vertexai.preview = MagicMock()
    mock_vertexai.preview.language_models = MagicMock()
    return mock_vertexai


def test_completion_preserves_custom_model_garden_api_base() -> None:
    custom_api_base = (
        "https://mg-endpoint-123.us-west1-456.prediction.vertexai.goog/v1/"
        "projects/123/locations/us-west1/endpoints/mg-endpoint-123"
    )
    mock_vertexai = _mock_vertexai_module()
    mock_openai_like_handler = MagicMock()
    mock_openai_like_handler.completion.return_value = "mock-response"

    with (
        patch.dict(
            sys.modules,
            {"vertexai": mock_vertexai, "vertexai.preview": mock_vertexai.preview},
        ),
        patch(
            "litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini."
            "VertexLLM._ensure_access_token",
            return_value=("fake-token", "resolved-project"),
        ),
        patch(
            "litellm.llms.openai_like.chat.handler.OpenAILikeChatHandler",
            return_value=mock_openai_like_handler,
        ),
    ):
        response = VertexAIModelGardenModels().completion(
            model="gemma4",
            messages=[{"role": "user", "content": "hi"}],
            model_response=MagicMock(),
            print_verbose=MagicMock(),
            encoding=None,
            logging_obj=MagicMock(),
            api_base=custom_api_base,
            optional_params={},
            custom_prompt_dict={},
            headers=None,
            timeout=30,
            litellm_params={},
            vertex_project="configured-project",
            vertex_location="us-west1",
        )

    assert response == "mock-response"
    mock_openai_like_handler.completion.assert_called_once()
    call_kwargs = mock_openai_like_handler.completion.call_args.kwargs
    assert call_kwargs["api_base"] == custom_api_base
    assert call_kwargs["model"] == ""
    assert call_kwargs["optional_params"]["stream"] is False


def test_completion_builds_openapi_base_for_publisher_model_without_custom_api_base() -> (
    None
):
    mock_vertexai = _mock_vertexai_module()
    mock_openai_like_handler = MagicMock()
    mock_openai_like_handler.completion.return_value = "mock-response"

    with (
        patch.dict(
            sys.modules,
            {"vertexai": mock_vertexai, "vertexai.preview": mock_vertexai.preview},
        ),
        patch(
            "litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini."
            "VertexLLM._ensure_access_token",
            return_value=("fake-token", "resolved-project"),
        ),
        patch(
            "litellm.llms.openai_like.chat.handler.OpenAILikeChatHandler",
            return_value=mock_openai_like_handler,
        ),
    ):
        response = VertexAIModelGardenModels().completion(
            model="xai/grok-4.1-fast-reasoning",
            messages=[{"role": "user", "content": "hi"}],
            model_response=MagicMock(),
            print_verbose=MagicMock(),
            encoding=None,
            logging_obj=MagicMock(),
            api_base=None,
            optional_params={},
            custom_prompt_dict={},
            headers=None,
            timeout=30,
            litellm_params={},
            vertex_project="configured-project",
            vertex_location="us-west1",
        )

    assert response == "mock-response"
    mock_openai_like_handler.completion.assert_called_once()
    call_kwargs = mock_openai_like_handler.completion.call_args.kwargs
    assert call_kwargs["api_base"] == (
        "https://us-west1-aiplatform.googleapis.com/v1/projects/configured-project/"
        "locations/us-west1/endpoints/openapi"
    )
    assert call_kwargs["model"] == "xai/grok-4.1-fast-reasoning"
    assert call_kwargs["optional_params"]["stream"] is False
