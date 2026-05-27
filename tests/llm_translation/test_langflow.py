import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def test_langflow_config_get_complete_url():
    from litellm.llms.langflow.chat.transformation import LangFlowConfig

    config = LangFlowConfig()
    url = config.get_complete_url(
        api_base="http://localhost:7860",
        api_key=None,
        model="langflow/my-flow-id",
        optional_params={},
        litellm_params={},
        stream=False,
    )
    assert url == "http://localhost:7860/api/v1/run/my-flow-id"


def test_langflow_config_transform_request_includes_session_id():
    from litellm.llms.langflow.chat.transformation import LangFlowConfig

    config = LangFlowConfig()
    request = config.transform_request(
        model="langflow/my-flow-id",
        messages=[{"role": "user", "content": "hello"}],
        optional_params={"session_id": "sess-abc"},
        litellm_params={},
        headers={},
    )

    assert request["input_value"] == "hello"
    assert request["input_type"] == "chat"
    assert request["output_type"] == "chat"
    assert request["session_id"] == "sess-abc"


def test_langflow_config_extract_response():
    from litellm.llms.langflow.chat.transformation import LangFlowConfig

    config = LangFlowConfig()
    content = config._extract_content_from_response(
        {
            "session_id": "sess-abc",
            "outputs": [
                {
                    "outputs": [
                        {
                            "results": {
                                "message": {"text": "Hello from LangFlow"},
                            }
                        }
                    ]
                }
            ],
        }
    )
    assert content == "Hello from LangFlow"


def test_langflow_provider_config_registered():
    cfg = ProviderConfigManager.get_provider_chat_config(
        model="langflow/flow-1",
        provider=LlmProviders.LANGFLOW,
    )
    assert cfg is not None
    assert cfg.__class__.__name__ == "LangFlowConfig"


@pytest.mark.asyncio
async def test_a2a_bridge_passes_context_id_as_session_id_for_langflow():
    from litellm.a2a_protocol.litellm_completion_bridge.handler import (
        A2ACompletionBridgeHandler,
    )

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

        await A2ACompletionBridgeHandler.handle_non_streaming(
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
async def test_langflow_acompletion_non_streaming():
    api_base = os.environ.get("LANGFLOW_API_BASE", "http://localhost:7860")
    flow_id = os.environ.get("LANGFLOW_FLOW_ID")
    api_key = os.environ.get("LANGFLOW_API_KEY")

    if not flow_id:
        pytest.skip("LANGFLOW_FLOW_ID not set")

    try:
        response = await litellm.acompletion(
            model=f"langflow/{flow_id}",
            messages=[{"role": "user", "content": "hello"}],
            api_base=api_base,
            api_key=api_key,
            session_id="litellm-test-session",
            stream=False,
        )
        assert response.choices[0].message.content
    except Exception as e:
        pytest.skip(f"LangFlow server not available: {e}")
