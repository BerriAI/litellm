import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

from unittest.mock import AsyncMock, patch

import pytest

from litellm.a2a_protocol.providers.config_manager import A2AProviderConfigManager
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def test_flow_id_cannot_be_overridden_via_optional_params():
    from litellm.llms.langflow.chat.transformation import LangFlowConfig, LangFlowError

    config = LangFlowConfig()
    url = config.get_complete_url(
        api_base="http://localhost:7860",
        api_key=None,
        model="langflow/authorized-flow",
        optional_params={},
        litellm_params={},
        stream=False,
    )
    assert url.endswith("/api/v1/run/authorized-flow")

    with pytest.raises(LangFlowError):
        config.get_complete_url(
            api_base="http://localhost:7860",
            api_key=None,
            model="langflow/authorized-flow",
            optional_params={"flow_id": "malicious-flow"},
            litellm_params={},
            stream=False,
        )


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


def test_merge_a2a_session_into_litellm_params():
    from litellm.llms.langflow.a2a import merge_a2a_session_into_litellm_params

    merged = merge_a2a_session_into_litellm_params(
        {"custom_llm_provider": "langflow", "model": "langflow/flow-1"},
        {"message": {"contextId": "shared-session-99"}},
    )
    assert merged["session_id"] == "shared-session-99"


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
