from typing import List, Optional, Tuple

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


def test_vertex_ai_lyria_models_route_to_interactions_config() -> None:
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
