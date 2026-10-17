from collections.abc import Mapping
from typing import Final

import pytest
from pydantic import TypeAdapter

from litellm.router_utils.endpoint_eligibility import UnsupportedEndpointError, eligible_deployments, endpoint_selection

_DEPLOYMENT: Final = TypeAdapter(list[dict[str, object]])


def _deployment(info: Mapping[str, object], identity: str = "embedding") -> dict[str, object]:
    return _DEPLOYMENT.validate_python(
        [
            {
                "model_name": "shared",
                "litellm_params": {"model": "openai/test-model"},
                "model_info": {"id": identity, **info},
            }
        ]
    )[0]


@pytest.mark.parametrize("call_type", ["acompletion", "anthropic_messages", "aresponses", "acreate_interaction"])
def test_explicit_restriction_excludes_an_embedding_deployment_from_a_mixed_group(call_type: str) -> None:
    embedding: Final = _deployment({"enforce_supported_endpoints": True, "supported_endpoints": ["/v1/embeddings"]})
    chat: Final = _deployment({"supported_endpoints": ["/v1/chat/completions"]}, "chat")
    with endpoint_selection(call_type):
        result: Final = eligible_deployments("shared", (embedding, chat))
    assert result == (chat,)


@pytest.mark.parametrize(
    "info",
    [
        {"mode": "audio_speech", "supported_endpoints": ["/v1/audio/speech"]},
        {"enforce_supported_endpoints": False, "supported_endpoints": ["/v1/embeddings"]},
        {"enforce_supported_endpoints": True},
        {"enforce_supported_endpoints": True, "supported_endpoints": None},
        {"enforce_supported_endpoints": True, "supported_endpoints": "invalid"},
        {"enforce_supported_endpoints": True, "supported_endpoints": ["/v1/messages"]},
    ],
)
def test_advisory_or_incomplete_metadata_preserves_routing(info: Mapping[str, object]) -> None:
    deployment: Final = _deployment(info)
    with endpoint_selection("acompletion"):
        result: Final = eligible_deployments("shared", (deployment,))
    assert result == (deployment,)


def test_unsupported_error_explains_the_requested_operation_and_available_endpoints() -> None:
    deployment: Final = _deployment({"enforce_supported_endpoints": True, "supported_endpoints": ["/v1/embeddings"]})
    with endpoint_selection("acreate_interaction"):
        result: Final = eligible_deployments("shared", (deployment,))
    assert isinstance(result, UnsupportedEndpointError)
    assert result.status_code == 400
    assert "/v1beta/interactions" in result.message
    assert "/v1/embeddings" in result.message


@pytest.mark.parametrize("call_type", ["aembedding", "aspeech", "_arealtime", "aget_responses"])
def test_chat_restriction_does_not_block_other_operations(call_type: str) -> None:
    deployment: Final = _deployment({"enforce_supported_endpoints": True, "supported_endpoints": ["/v1/embeddings"]})
    with endpoint_selection(call_type):
        result: Final = eligible_deployments("shared", (deployment,))
    assert result == (deployment,)


def test_nested_non_chat_calls_do_not_inherit_the_outer_chat_restriction() -> None:
    deployment: Final = _deployment({"enforce_supported_endpoints": True, "supported_endpoints": ["/v1/embeddings"]})
    with endpoint_selection("acompletion"):
        with endpoint_selection("aembedding"):
            embedding_result: Final = eligible_deployments("shared", (deployment,))
        chat_result: Final = eligible_deployments("shared", (deployment,))
    assert embedding_result == (deployment,)
    assert isinstance(chat_result, UnsupportedEndpointError)
