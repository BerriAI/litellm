from unittest.mock import AsyncMock, patch

import pytest

from litellm.a2a_protocol.providers.config_manager import A2AProviderConfigManager
from litellm.llms.langflow.a2a import merge_a2a_session_into_litellm_params


def test_merge_a2a_session_into_litellm_params():
    merged = merge_a2a_session_into_litellm_params(
        {"custom_llm_provider": "langflow", "model": "langflow/flow-1"},
        {"message": {"contextId": "shared-session-99"}},
    )
    assert merged["session_id"] == "shared-session-99"


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
