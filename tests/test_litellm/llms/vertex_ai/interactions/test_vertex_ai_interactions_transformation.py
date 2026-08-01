from typing import List, Optional, Tuple
from unittest.mock import MagicMock

import httpx

import litellm
import pytest

from litellm.interactions.utils import get_provider_interactions_api_config
from litellm.llms.vertex_ai.interactions.transformation import (
    VertexAIInteractionsConfig,
)
from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES
from litellm.types.router import GenericLiteLLMParams


class FakeVertexAuth:
    def __init__(self) -> None:
        self.calls: List[Tuple[Optional[VERTEX_CREDENTIALS_TYPES], Optional[str]]] = []

    def get_access_token(
        self,
        credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        project_id: Optional[str],
        _retry_reauth: bool = False,
    ) -> Tuple[str, str]:
        self.calls.append((credentials, project_id))
        return "test-access-token", project_id or "resolved-project"


def test_validate_environment_uses_vertex_bearer_token() -> None:
    vertex_auth = FakeVertexAuth()
    config = VertexAIInteractionsConfig(vertex_auth=vertex_auth)
    credentials = '{"type":"service_account"}'

    headers = config.validate_environment(
        headers={"X-Custom": "value"},
        model="lyria-3-pro-preview",
        litellm_params=GenericLiteLLMParams(
            vertex_project="music-project",
            vertex_credentials=credentials,
        ),
    )

    assert headers["Authorization"] == "Bearer test-access-token"
    assert headers["Content-Type"] == "application/json"
    assert headers["X-Custom"] == "value"
    assert "x-goog-api-key" not in headers
    assert "Api-Revision" not in headers
    assert vertex_auth.calls == [(credentials, "music-project")]


def test_get_complete_url_uses_global_vertex_interactions_endpoint() -> None:
    config = VertexAIInteractionsConfig(vertex_auth=FakeVertexAuth())

    url = config.get_complete_url(
        api_base=None,
        model="lyria-3-pro-preview",
        litellm_params={"vertex_project": "music-project"},
    )

    assert (
        url
        == "https://aiplatform.googleapis.com/v1beta1/projects/music-project/locations/global/interactions"
    )


def test_get_complete_url_appends_alt_sse_for_streaming() -> None:
    config = VertexAIInteractionsConfig(vertex_auth=FakeVertexAuth())

    url = config.get_complete_url(
        api_base="https://private.googleapis.com/",
        model="lyria-3-clip-preview",
        litellm_params={"vertex_project": "music-project"},
        stream=True,
    )

    assert (
        url
        == "https://private.googleapis.com/v1beta1/projects/music-project/locations/global/interactions?alt=sse"
    )


def test_get_complete_url_reuses_project_from_params_without_second_auth_call() -> None:
    vertex_auth = FakeVertexAuth()
    config = VertexAIInteractionsConfig(vertex_auth=vertex_auth)

    config.validate_environment(
        headers={},
        model="lyria-3-pro-preview",
        litellm_params=GenericLiteLLMParams(vertex_project="music-project"),
    )
    url = config.get_complete_url(
        api_base=None,
        model="lyria-3-pro-preview",
        litellm_params={"vertex_project": "music-project"},
    )

    assert (
        url
        == "https://aiplatform.googleapis.com/v1beta1/projects/music-project/locations/global/interactions"
    )
    assert vertex_auth.calls == [(None, "music-project")]


@pytest.mark.parametrize(
    "api_base",
    [
        "https://attacker.example",
        "http://aiplatform.googleapis.com",
    ],
)
def test_get_complete_url_rejects_untrusted_api_base(api_base: str) -> None:
    config = VertexAIInteractionsConfig(vertex_auth=FakeVertexAuth())

    with pytest.raises(
        ValueError,
        match="trusted Google API HTTPS endpoint",
    ):
        config.get_complete_url(
            api_base=api_base,
            model="lyria-3-pro-preview",
            litellm_params={"vertex_project": "music-project"},
        )


def test_interaction_id_urls_are_encoded() -> None:
    config = VertexAIInteractionsConfig(vertex_auth=FakeVertexAuth())

    url, params = config.transform_cancel_interaction_request(
        interaction_id="interaction/id with spaces",
        api_base="",
        litellm_params=GenericLiteLLMParams(vertex_project="music-project"),
        headers={},
    )

    assert params == {}
    assert (
        url
        == "https://aiplatform.googleapis.com/v1beta1/projects/music-project/locations/global/interactions/interaction%2Fid%20with%20spaces:cancel"
    )


def test_transform_request_strips_vertex_ai_prefix() -> None:
    config = VertexAIInteractionsConfig(vertex_auth=FakeVertexAuth())

    body = config.transform_request(
        model="vertex_ai/lyria-3-pro-preview",
        agent=None,
        input=[
            {"type": "text", "text": "Write a bright synth-pop track."},
            {
                "type": "image",
                "image": {
                    "image_bytes": "abc123",
                    "mime_type": "image/png",
                },
            },
        ],
        optional_params={"stream": False},
        litellm_params=GenericLiteLLMParams(vertex_project="music-project"),
        headers={},
    )

    assert body == {
        "model": "lyria-3-pro-preview",
        "input": [
            {"type": "text", "text": "Write a bright synth-pop track."},
            {
                "type": "image",
                "image": {
                    "image_bytes": "abc123",
                    "mime_type": "image/png",
                },
            },
        ],
        "stream": False,
    }


@pytest.mark.parametrize(
    ("model", "expected_cost"),
    [
        ("lyria-3-clip-preview", 0.04),
        ("vertex_ai/lyria-3-pro-preview", 0.08),
    ],
)
def test_transform_response_sets_lyria_generation_cost(
    model: str, expected_cost: float
) -> None:
    config = VertexAIInteractionsConfig(vertex_auth=FakeVertexAuth())

    response = config.transform_response(
        model=model,
        raw_response=httpx.Response(
            200,
            json={
                "status": "completed",
                "outputs": [
                    {
                        "type": "audio",
                        "data": "bHlyaWEtYXVkaW8=",
                        "mime_type": "audio/mpeg",
                    }
                ],
            },
        ),
        logging_obj=MagicMock(),
    )

    assert response._hidden_params["response_cost"] == expected_cost


def test_transform_streaming_response_sets_lyria_generation_cost() -> None:
    config = VertexAIInteractionsConfig(vertex_auth=FakeVertexAuth())

    response = config.transform_streaming_response(
        model="lyria-3-clip-preview",
        parsed_chunk={"event_type": "interaction.completed", "status": "completed"},
        logging_obj=MagicMock(),
    )

    assert response._hidden_params["response_cost"] == 0.04


def test_vertex_ai_models_with_interactions_endpoint_route_to_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        "vertex_ai/lyria-3-clip-preview",
        {"supported_endpoints": ["/v1beta/interactions"]},
    )
    monkeypatch.setitem(
        litellm.model_cost,
        "vertex_ai/lyria-3-pro-preview",
        {"supported_endpoints": ["/v1beta/interactions"]},
    )
    monkeypatch.setitem(
        litellm.model_cost,
        "vertex_ai/gemini-2.5-flash",
        {"supported_endpoints": ["/v1/chat/completions"]},
    )

    assert isinstance(
        get_provider_interactions_api_config(
            provider="vertex_ai",
            model="lyria-3-clip-preview",
        ),
        VertexAIInteractionsConfig,
    )
    assert isinstance(
        get_provider_interactions_api_config(
            provider="vertex_ai",
            model="vertex_ai/lyria-3-pro-preview",
        ),
        VertexAIInteractionsConfig,
    )
    assert (
        get_provider_interactions_api_config(
            provider="vertex_ai",
            model="gemini-2.5-flash",
        )
        is None
    )


def test_vertex_ai_interaction_operations_route_without_model() -> None:
    assert isinstance(
        get_provider_interactions_api_config(provider="vertex_ai"),
        VertexAIInteractionsConfig,
    )
