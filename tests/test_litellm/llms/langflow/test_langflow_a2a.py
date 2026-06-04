from unittest.mock import AsyncMock, patch

import pytest

from litellm.a2a_protocol.litellm_completion_bridge.handler import (
    A2A_USER_API_KEY_HASH_PARAM,
)
from litellm.a2a_protocol.providers.config_manager import A2AProviderConfigManager
from litellm.llms.langflow.a2a import merge_a2a_session_into_litellm_params


def test_merge_a2a_session_into_litellm_params():
    merged = merge_a2a_session_into_litellm_params(
        {"custom_llm_provider": "langflow", "model": "langflow/flow-1"},
        {"message": {"contextId": "shared-session-99"}},
    )
    assert merged["session_id"] == "shared-session-99"


def test_merge_a2a_session_is_scoped_per_principal():
    """The LangFlow session must be bound to the authenticated key so two
    distinct keys cannot share memory by reusing the same A2A contextId, while
    the same key keeps a stable session across turns."""
    base = {"custom_llm_provider": "langflow", "model": "langflow/flow-1"}
    params = {"message": {"contextId": "ctx-1"}}

    key_a = merge_a2a_session_into_litellm_params(base, params, "hash-a")["session_id"]
    key_a_again = merge_a2a_session_into_litellm_params(base, params, "hash-a")[
        "session_id"
    ]
    key_b = merge_a2a_session_into_litellm_params(base, params, "hash-b")["session_id"]

    assert key_a == key_a_again, "same key + contextId must stay on one session"
    assert key_a != key_b, "different keys must not collide on the same contextId"
    assert key_a != "ctx-1", "raw client contextId must not be used verbatim"
    assert key_a.endswith("-ctx-1"), "original contextId kept for correlation"
    assert "hash-a" not in key_a, "raw principal must not be sent to LangFlow"


def test_merge_a2a_session_without_context_id_is_noop():
    merged = merge_a2a_session_into_litellm_params(
        {"custom_llm_provider": "langflow", "model": "langflow/flow-1"},
        {"message": {"role": "user"}},
    )
    assert "session_id" not in merged


def test_langflow_a2a_provider_config_registered():
    cfg = A2AProviderConfigManager.get_provider_config(
        custom_llm_provider="langflow",
        model="langflow/flow-1",
    )
    assert cfg is not None
    assert cfg.__class__.__name__ == "LangFlowA2AConfig"


@pytest.mark.asyncio
async def test_langflow_a2a_config_passes_session_id_to_completion():
    from litellm.a2a_protocol.providers.langflow.config import LangFlowA2AConfig

    mock_response = type(
        "R",
        (),
        {
            "choices": [
                type(
                    "C",
                    (),
                    {"message": type("M", (), {"content": "ok"})()},
                )()
            ]
        },
    )()

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_response

        await LangFlowA2AConfig().handle_non_streaming(
            request_id="req-1",
            params={
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "hi"}],
                    "contextId": "shared-session-99",
                }
            },
            litellm_params={
                "custom_llm_provider": "langflow",
                "model": "langflow/flow-1",
                "api_base": "http://localhost:7860",
            },
            api_base="http://localhost:7860",
        )

        assert (
            mock_acompletion.call_args.kwargs.get("session_id") == "shared-session-99"
        )


@pytest.mark.asyncio
async def test_langflow_a2a_config_scopes_session_by_authenticated_key():
    from litellm.a2a_protocol.providers.langflow.config import LangFlowA2AConfig

    mock_response = type(
        "R",
        (),
        {"choices": [type("C", (), {"message": type("M", (), {"content": "ok"})()})()]},
    )()

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_response

        await LangFlowA2AConfig().handle_non_streaming(
            request_id="req-1",
            params={
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "hi"}],
                    "contextId": "ctx-1",
                }
            },
            litellm_params={
                "custom_llm_provider": "langflow",
                "model": "langflow/flow-1",
                "api_base": "http://localhost:7860",
                A2A_USER_API_KEY_HASH_PARAM: "hashed-key-1",
            },
            api_base="http://localhost:7860",
        )

    forwarded = mock_acompletion.call_args.kwargs
    assert forwarded.get("session_id") != "ctx-1"
    assert forwarded.get("session_id").endswith("-ctx-1")
    assert (
        A2A_USER_API_KEY_HASH_PARAM not in forwarded
    ), "internal principal param must not leak to the LLM call"


@pytest.mark.asyncio
async def test_langflow_a2a_config_requires_litellm_params_non_streaming():
    from litellm.a2a_protocol.providers.langflow.config import LangFlowA2AConfig

    with pytest.raises(ValueError, match="litellm_params is required"):
        await LangFlowA2AConfig().handle_non_streaming(
            request_id="req-1",
            params={"message": {"contextId": "shared-session-99"}},
        )


@pytest.mark.asyncio
async def test_langflow_a2a_config_requires_litellm_params_streaming():
    from litellm.a2a_protocol.providers.langflow.config import LangFlowA2AConfig

    with pytest.raises(ValueError, match="litellm_params is required"):
        async for _ in LangFlowA2AConfig().handle_streaming(
            request_id="req-1",
            params={"message": {"contextId": "shared-session-99"}},
        ):
            pass
