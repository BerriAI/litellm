import json
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.llms.litellm_proxy.responses.transformation import LiteLLMProxyResponsesAPIConfig
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager
from tests.unit.proxy.conftest import httpx_transport

_PROXY_RESPONSES_URL: Final = "https://my-proxy.example.com/responses"


@pytest.mark.usefixtures(httpx_transport.__name__)
def test_litellm_proxy_responses_request_uses_the_openai_wire_format() -> None:
    with respx.mock(assert_all_called=True) as mock:
        route: Final = mock.post(_PROXY_RESPONSES_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "resp_proxy_1",
                    "object": "response",
                    "created_at": 1750000000,
                    "status": "completed",
                    "model": "gpt-5.5",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_proxy_1",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "pong", "annotations": []}],
                        }
                    ],
                    "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
                },
            )
        )
        response: Final = litellm.responses(
            model="litellm_proxy/gpt-5.5",
            input="ping",
            max_output_tokens=16,
            api_base="https://my-proxy.example.com/",
            api_key="sk-proxy-key",
        )
        request: Final = route.calls.last.request
    assert request.headers["authorization"] == "Bearer sk-proxy-key"
    assert json.loads(request.content) == {"model": "gpt-5.5", "input": "ping", "max_output_tokens": 16}
    assert (response.usage.input_tokens, response.usage.output_tokens) == (3, 1)
    assert response.output[0].content[0].text == "pong"


def test_provider_config_manager_returns_litellm_proxy_responses_config() -> None:
    config: Final = ProviderConfigManager.get_provider_responses_api_config(
        model="litellm_proxy/gpt-5.5", provider=LlmProviders.LITELLM_PROXY
    )
    assert isinstance(config, LiteLLMProxyResponsesAPIConfig)
    assert config.custom_llm_provider == LlmProviders.LITELLM_PROXY


@pytest.mark.parametrize("api_base", ["https://my-proxy.example.com", "https://my-proxy.example.com/"])
def test_get_complete_url_appends_responses_path(api_base: str) -> None:
    assert (
        LiteLLMProxyResponsesAPIConfig().get_complete_url(api_base=api_base, litellm_params={})
        == "https://my-proxy.example.com/responses"
    )


def test_get_complete_url_requires_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_PROXY_API_BASE", raising=False)
    with pytest.raises(ValueError, match="api_base not set"):
        LiteLLMProxyResponsesAPIConfig().get_complete_url(api_base=None, litellm_params={})
