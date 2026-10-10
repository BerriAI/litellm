import json
from typing import Final

import httpx
import pytest
import respx

import litellm

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.vllm.passthrough.transformation import VLLMPassthroughConfig
from litellm.passthrough.main import allm_passthrough_route
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

_API_BASE: Final = "http://vllm-upstream.test:8090"


@pytest.fixture(autouse=True)
def _httpx_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)


def test_hosted_vllm_resolves_vllm_passthrough_config() -> None:
    cfg: Final = ProviderConfigManager.get_provider_passthrough_config(
        model="hosted_vllm/my-deployment",
        provider=LlmProviders.HOSTED_VLLM,
    )
    assert isinstance(cfg, VLLMPassthroughConfig)


@pytest.mark.asyncio
async def test_allm_passthrough_route_hosted_vllm_sends_normalized_model(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(f"{_API_BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    client: Final = AsyncHTTPHandler()
    response: Final = await allm_passthrough_route(
        method="POST",
        endpoint="v1/chat/completions",
        model="hosted_vllm/my-deployment",
        api_base=_API_BASE,
        json={
            "model": "anything",
            "messages": [{"role": "user", "content": "Hello"}],
        },
        client=client,
    )
    assert response.status_code == 200
    assert route.call_count == 1
    outbound: Final = json.loads(route.calls[0].request.content)
    assert outbound["model"] == "my-deployment"
    assert outbound["messages"] == [{"role": "user", "content": "Hello"}]
