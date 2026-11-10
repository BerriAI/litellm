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


@pytest.mark.asyncio
async def test_allm_passthrough_route_hosted_vllm_sends_deployment_api_key(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(f"{_API_BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    response: Final = await allm_passthrough_route(
        method="POST",
        endpoint="v1/chat/completions",
        model="hosted_vllm/my-deployment",
        api_base=_API_BASE,
        api_key="sk-upstream",
        json={"model": "anything", "messages": [{"role": "user", "content": "Hello"}]},
        client=AsyncHTTPHandler(),
    )
    assert response.status_code == 200
    assert route.calls[0].request.headers["authorization"] == "Bearer sk-upstream"
    assert "x-api-key" not in route.calls[0].request.headers


@pytest.mark.parametrize("endpoint", ["v1/chat/completions", "chat/completions"])
@pytest.mark.asyncio
async def test_allm_passthrough_route_hosted_vllm_reaches_v1_paths_under_an_api_base_ending_in_v1(
    respx_mock: respx.MockRouter, endpoint: str
) -> None:
    route: Final = respx_mock.post(f"{_API_BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    response: Final = await allm_passthrough_route(
        method="POST",
        endpoint=endpoint,
        model="hosted_vllm/my-deployment",
        api_base=f"{_API_BASE}/v1",
        api_key="sk-upstream",
        json={"model": "anything", "messages": [{"role": "user", "content": "Hello"}]},
        client=AsyncHTTPHandler(),
    )
    assert response.status_code == 200
    assert route.call_count == 1


@pytest.mark.asyncio
async def test_allm_passthrough_route_hosted_vllm_sends_no_auth_for_a_keyless_deployment(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_API_KEY", "sk-vllm-route-only")
    monkeypatch.delenv("HOSTED_VLLM_API_KEY", raising=False)
    route: Final = respx_mock.post(f"{_API_BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    response: Final = await allm_passthrough_route(
        method="POST",
        endpoint="v1/chat/completions",
        model="hosted_vllm/my-deployment",
        api_base=_API_BASE,
        json={"model": "anything", "messages": [{"role": "user", "content": "Hello"}]},
        client=AsyncHTTPHandler(),
    )
    assert response.status_code == 200
    assert "authorization" not in route.calls[0].request.headers
    assert "x-api-key" not in route.calls[0].request.headers


@pytest.mark.parametrize(
    ("vllm_api_key", "expected_authorization"),
    [("sk-vllm-env", "Bearer sk-vllm-env"), ("oidc/env/VLLM_OIDC_TOKEN", "Bearer sk-vllm-oidc")],
)
@pytest.mark.asyncio
async def test_allm_passthrough_route_keyless_hosted_vllm_without_api_base_sends_the_vllm_api_base_key(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, vllm_api_key: str, expected_authorization: str
) -> None:
    monkeypatch.setenv("VLLM_API_BASE", _API_BASE)
    monkeypatch.setenv("VLLM_API_KEY", vllm_api_key)
    monkeypatch.setenv("VLLM_OIDC_TOKEN", "sk-vllm-oidc")
    monkeypatch.delenv("HOSTED_VLLM_API_BASE", raising=False)
    monkeypatch.delenv("HOSTED_VLLM_API_KEY", raising=False)
    route: Final = respx_mock.post(f"{_API_BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    response: Final = await allm_passthrough_route(
        method="POST",
        endpoint="v1/chat/completions",
        model="hosted_vllm/my-deployment",
        json={"model": "anything", "messages": [{"role": "user", "content": "Hello"}]},
        client=AsyncHTTPHandler(),
    )
    assert response.status_code == 200
    assert route.calls[0].request.headers["authorization"] == expected_authorization


@pytest.mark.parametrize(
    ("model_api_base", "vllm_api_base"), [(f"{_API_BASE}/", _API_BASE), (_API_BASE, f"{_API_BASE}/")]
)
@pytest.mark.asyncio
async def test_allm_passthrough_route_keyless_hosted_vllm_matches_the_vllm_api_base_across_a_trailing_slash(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, model_api_base: str, vllm_api_base: str
) -> None:
    monkeypatch.setenv("VLLM_API_BASE", vllm_api_base)
    monkeypatch.setenv("VLLM_API_KEY", "sk-vllm-env")
    monkeypatch.delenv("HOSTED_VLLM_API_BASE", raising=False)
    monkeypatch.delenv("HOSTED_VLLM_API_KEY", raising=False)
    route: Final = respx_mock.post(url__regex=rf"^{_API_BASE}/+v1/chat/completions$").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    response: Final = await allm_passthrough_route(
        method="POST",
        endpoint="v1/chat/completions",
        model="hosted_vllm/my-deployment",
        api_base=model_api_base,
        json={"model": "anything", "messages": [{"role": "user", "content": "Hello"}]},
        client=AsyncHTTPHandler(),
    )
    assert response.status_code == 200
    assert route.calls[0].request.headers["authorization"] == "Bearer sk-vllm-env"


@pytest.mark.parametrize("vllm_api_key", ["", "oidc/google/audience"])
@pytest.mark.asyncio
async def test_allm_passthrough_route_unusable_vllm_api_key_sends_no_authorization_header(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, vllm_api_key: str
) -> None:
    monkeypatch.setenv("VLLM_API_BASE", _API_BASE)
    monkeypatch.setenv("VLLM_API_KEY", vllm_api_key)
    monkeypatch.delenv("HOSTED_VLLM_API_BASE", raising=False)
    monkeypatch.delenv("HOSTED_VLLM_API_KEY", raising=False)
    route: Final = respx_mock.post(f"{_API_BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    response: Final = await allm_passthrough_route(
        method="POST",
        endpoint="v1/chat/completions",
        model="hosted_vllm/my-deployment",
        json={"model": "anything", "messages": [{"role": "user", "content": "Hello"}]},
        client=AsyncHTTPHandler(),
    )
    assert response.status_code == 200
    assert "authorization" not in route.calls[0].request.headers


@pytest.mark.parametrize(
    ("api_base", "api_key", "expected_authorization"),
    [
        (None, "sk-upstream", "Bearer sk-upstream"),
        (None, "sk-hosted-env", "Bearer sk-hosted-env"),
        (_API_BASE, "sk-hosted-env", "Bearer sk-hosted-env"),
        (None, None, "Bearer sk-hosted-env"),
    ],
)
@pytest.mark.asyncio
async def test_allm_passthrough_route_hosted_vllm_on_the_vllm_api_base_sends_its_own_resolved_key(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    api_base: str | None,
    api_key: str | None,
    expected_authorization: str,
) -> None:
    monkeypatch.setenv("VLLM_API_BASE", _API_BASE)
    monkeypatch.setenv("VLLM_API_KEY", "sk-vllm-env")
    monkeypatch.delenv("HOSTED_VLLM_API_BASE", raising=False)
    monkeypatch.setenv("HOSTED_VLLM_API_KEY", "sk-hosted-env")
    route: Final = respx_mock.post(f"{_API_BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    response: Final = await allm_passthrough_route(
        method="POST",
        endpoint="v1/chat/completions",
        model="hosted_vllm/my-deployment",
        api_base=api_base,
        api_key=api_key,
        json={"model": "anything", "messages": [{"role": "user", "content": "Hello"}]},
        client=AsyncHTTPHandler(),
    )
    assert response.status_code == 200
    assert route.calls[0].request.headers["authorization"] == expected_authorization
