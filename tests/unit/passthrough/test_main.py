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


@pytest.mark.parametrize(
    ("model", "pinned_profile_id", "forwarded_profile_id"),
    [
        ("azure_ai/claude-sonnet-5-5", "uprof_pin", "uprof_pin"),
        ("azure_ai/gpt-5", "uprof_pin", "uprof_caller"),
        ("azure_ai/claude-sonnet-5-5", None, "uprof_caller"),
        ("azure_ai/gpt-5", None, "uprof_caller"),
        ("hosted_vllm/claude-sonnet-5-5", "uprof_pin", "uprof_caller"),
        ("azure_ai/claude-unlisted-deployment", "uprof_pin", "uprof_caller"),
    ],
    ids=[
        "claude_pin",
        "other_model_pin",
        "claude_no_pin",
        "other_model_no_pin",
        "other_provider_pin",
        "unmapped_claude_name_pin",
    ],
)
@pytest.mark.asyncio
async def test_allm_passthrough_route_puts_a_pin_over_the_x_pass_user_profile_id_for_flagged_models_only(
    respx_mock: respx.MockRouter, model: str, pinned_profile_id: str | None, forwarded_profile_id: str
) -> None:
    upstream: Final = respx_mock.route(host="example.services.ai.azure.com").respond(json={"ok": True})

    response: Final = await allm_passthrough_route(
        method="POST",
        endpoint="chat/completions",
        model=model,
        api_base="https://example.services.ai.azure.com/models",
        api_key="sk-foundry",
        request_headers={"content-type": "application/json", "X-Pass-Anthropic-User-Profile-Id": "uprof_caller"},
        json={"model": "anything", "messages": [{"role": "user", "content": "Hello"}]},
        client=AsyncHTTPHandler(),
        anthropic_user_profile_id_pin=pinned_profile_id,
    )

    assert response.status_code == 200
    assert upstream.calls.last.request.headers.get_list("anthropic-user-profile-id") == [forwarded_profile_id]


@pytest.mark.parametrize(
    ("model_info", "forwarded_profile_id"),
    [({"base_model": "azure_ai/claude-sonnet-5-5"}, "uprof_pin"), ({}, "uprof_caller")],
    ids=["base_model_flagged", "no_base_model"],
)
@pytest.mark.asyncio
async def test_allm_passthrough_route_pins_an_unlisted_claude_whose_base_model_the_cost_map_flags(
    respx_mock: respx.MockRouter, model_info: dict[str, str], forwarded_profile_id: str
) -> None:
    upstream: Final = respx_mock.route(host="example.services.ai.azure.com").respond(json={"ok": True})

    response: Final = await allm_passthrough_route(
        method="POST",
        endpoint="chat/completions",
        model="azure_ai/claude-unlisted-deployment",
        api_base="https://example.services.ai.azure.com/models",
        api_key="sk-foundry",
        request_headers={"content-type": "application/json", "X-Pass-Anthropic-User-Profile-Id": "uprof_caller"},
        json={"model": "anything", "messages": [{"role": "user", "content": "Hello"}]},
        client=AsyncHTTPHandler(),
        anthropic_user_profile_id_pin="uprof_pin",
        model_info=model_info,
    )

    assert response.status_code == 200
    assert upstream.calls.last.request.headers.get_list("anthropic-user-profile-id") == [forwarded_profile_id]
