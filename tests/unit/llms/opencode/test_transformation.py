from typing import Final

import pytest

from litellm.constants import SESSION_ID_GENERATED_METADATA_KEY
from litellm.litellm_core_utils.provider_affinity import add_provider_affinity_header
from litellm.llms.opencode.transformation import OpenCodeProviderAdapter


@pytest.mark.parametrize(
    "api_base",
    [
        "https://opencode.ai/zen/v1",
        "https://opencode.ai/zen/go/v1/",
        "https://opencode.ai/inference/openai/v1",
        "https://opencode.ai/inference/go/openai/v1",
        "https://opencode.ai/inference/go/anthropic/v1",
        "https://opencode.ai/inference/anthropic/v1",
    ],
)
def test_opencode_defaults_affinity_header_to_the_callers_session(api_base: str) -> None:
    headers: Final = {"Authorization": "Bearer test-key"}
    params: Final = {"metadata": {"session_id": "conversation-123"}}
    assert add_provider_affinity_header(
        headers, params, OpenCodeProviderAdapter().get_provider_affinity_header(api_base)
    ) == {
        **headers,
        "x-opencode-session": "conversation-123",
    }
    assert headers == {"Authorization": "Bearer test-key"}
    assert params == {"metadata": {"session_id": "conversation-123"}}


@pytest.mark.parametrize(
    "api_base",
    [
        None,
        "https://api.openai.com/v1",
        "https://opencode.ai.example/zen/go/v1",
        "https://opencode.ai@example.com/zen/go/v1",
        "https://example.com/opencode.ai/zen/go/v1",
        "https://example.com/v1?upstream=https://opencode.ai/zen/go/v1",
        "https://opencode.ai/console/v1",
        "http://opencode.ai/zen/go/v1",
    ],
)
def test_other_destinations_do_not_receive_an_automatic_session_header(api_base: str | None) -> None:
    headers: Final = {"Authorization": "Bearer test-key", "x-existing": "unchanged"}
    assert (
        add_provider_affinity_header(
            headers,
            {"litellm_session_id": "conversation-123"},
            OpenCodeProviderAdapter().get_provider_affinity_header(api_base),
        )
        == headers
    )


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"litellm_trace_id": "request-123"},
        {"metadata": {"trace_id": "request-123"}},
        {
            "litellm_session_id": "request-123",
            "litellm_metadata": {"session_id": "request-123", SESSION_ID_GENERATED_METADATA_KEY: True},
        },
    ],
)
def test_opencode_does_not_invent_a_conversation_id(params: dict[str, object]) -> None:
    assert (
        add_provider_affinity_header(
            {}, params, OpenCodeProviderAdapter().get_provider_affinity_header("https://opencode.ai/zen/go/v1")
        )
        == {}
    )


def test_opencode_preserves_explicit_header_and_affinity_configuration() -> None:
    assert add_provider_affinity_header(
        {"X-OpenCode-Session": "explicit"},
        {"litellm_session_id": "conversation-123"},
        OpenCodeProviderAdapter().get_provider_affinity_header("https://opencode.ai/zen/go/v1"),
    ) == {"X-OpenCode-Session": "explicit"}
    assert add_provider_affinity_header(
        {},
        {"litellm_session_id": "conversation-123", "provider_affinity_header": "x-custom-session"},
        OpenCodeProviderAdapter().get_provider_affinity_header("https://opencode.ai/zen/go/v1"),
    ) == {"x-custom-session": "conversation-123"}
