import json
from collections.abc import Mapping
from datetime import datetime
from types import SimpleNamespace
from typing import Final

import httpx
import pytest
import respx
from pydantic import JsonValue

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.llms.anthropic.count_tokens import handler as count_handler
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import DEFAULT_ANTHROPIC_API_VERSION
from litellm.llms.anthropic.prompt_cache_prediction import (
    NativePredictionTarget,
    cache_scope,
    count_prompt_tokens,
    parse_observed_cache,
    parse_prompt,
    resolve_prediction_target,
    supported_prediction_headers,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.models.credentials import CredentialItem
from litellm.proxy import proxy_server
from litellm.proxy.hooks.prompt_cache_prediction import PromptCacheObserver, lookup
from litellm.proxy.management_endpoints.prompt_cache_prediction import predict_arm
from litellm.proxy.utils import InternalUsageCache
from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo
from litellm.types.utils import CacheCreationTokenDetails, ModelResponse, PromptTokensDetailsWrapper, Usage

_MODEL: Final = "claude-sonnet-5"
_KEY: Final = "test-provider-key"
_CALLER: Final = "test-caller-hash"
_DEPLOYMENT: Final = "test-native-deployment"


def _body() -> dict[str, JsonValue]:
    return {
        "model": _MODEL,
        "system": "Keep this context",
        "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "A cacheable prefix", "cache_control": {"type": "ephemeral"}}
        ]}],
    }


@pytest.mark.parametrize("version", [None, "2099-01-01", DEFAULT_ANTHROPIC_API_VERSION])
@pytest.mark.asyncio
async def test_observer_records_only_version_supported_by_token_counter(version: str | None) -> None:
    cache: Final = DualCache()
    observer: Final = PromptCacheObserver(InternalUsageCache(dual_cache=cache), clock=lambda: 1010.0)
    body: Final = _body()
    prefix: Final = parse_prompt(body)
    assert prefix is not None
    headers: Final = {"x-api-key": _KEY, **({"anthropic-version": version} if version is not None else {})}
    wire: Final = httpx.Request("POST", "https://api.anthropic.com/v1/messages", headers=headers, json=body)
    response: Final = ModelResponse(
        model=_MODEL,
        usage=Usage(
            prompt_tokens=311,
            completion_tokens=2,
            total_tokens=313,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                cached_tokens=100,
                cache_creation_tokens=200,
                cache_creation_token_details=CacheCreationTokenDetails(
                    ephemeral_5m_input_tokens=200, ephemeral_1h_input_tokens=0
                ),
            ),
        ),
    )
    await observer.async_log_success_event(
        {
            "call_type": "anthropic_messages",
            "custom_llm_provider": "anthropic",
            "httpx_response": httpx.Response(200, request=wire),
            "first_api_call_start_time": datetime.fromtimestamp(1000.0),
            "standard_logging_object": {
                "status": "success", "model_id": _DEPLOYMENT,
                "metadata": {"user_api_key_hash": _CALLER},
            },
        },
        response,
        datetime.fromtimestamp(1010.0),
        datetime.fromtimestamp(1010.0),
    )
    default_scope: Final = cache_scope(_CALLER, _DEPLOYMENT, _KEY, _MODEL)
    found: Final = await lookup(cache, default_scope, prefix, now=1010.0)
    assert (found is not None) == (version == DEFAULT_ANTHROPIC_API_VERSION)
    if version != DEFAULT_ANTHROPIC_API_VERSION:
        other_scope: Final = cache_scope(_CALLER, _DEPLOYMENT, _KEY, _MODEL, version or "")
        assert await lookup(cache, other_scope, prefix, now=1010.0) is None


@pytest.mark.parametrize("headers, supported", [
    ({}, True),
    ({"Anthropic-Version": DEFAULT_ANTHROPIC_API_VERSION}, True),
    ({"anthropic-version": "2099-01-01"}, False),
    ({"Anthropic-Beta": ""}, False),
    ({"anthropic-beta": "future-feature"}, False),
])
def test_prediction_header_eligibility(headers: Mapping[str, str], supported: bool) -> None:
    assert supported_prediction_headers(headers) is supported


@pytest.mark.asyncio
async def test_provider_count_uses_same_version_and_preserves_native_input(monkeypatch: pytest.MonkeyPatch) -> None:
    body: Final = _body()
    requests: Final[list[httpx.Request]] = []

    def provider(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"input_tokens": 311})

    client: Final = AsyncHTTPHandler()
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(provider))
    monkeypatch.setattr(count_handler, "get_async_httpx_client", lambda **kwargs: client)
    try:
        assert await count_prompt_tokens(_MODEL, _KEY, body) == 311
    finally:
        await client.client.aclose()
    assert len(requests) == 1
    assert requests[0].headers["anthropic-version"] == DEFAULT_ANTHROPIC_API_VERSION
    assert requests[0].url == "https://api.anthropic.com/v1/messages/count_tokens"
    assert json.loads(requests[0].content) == body


@pytest.mark.parametrize("source", ["static", "database"])
@pytest.mark.asyncio
async def test_environment_credential_matches_native_count_and_observed_scope(
    source: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LIT7658_PROVIDER_KEY", _KEY)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    params: Final = {
        "model": f"anthropic/{_MODEL}", "api_key": "os.environ/LIT7658_PROVIDER_KEY",
        "api_base": "https://api.anthropic.com",
    }
    router: Final = litellm.Router(model_list=[{
        "model_name": "test-native", "litellm_params": dict(params), "model_info": {"id": _DEPLOYMENT},
    }] if source == "static" else [], num_retries=0)
    if source == "database":
        monkeypatch.setattr(proxy_server, "llm_router", router)
        assert proxy_server.ProxyConfig()._add_deployment([SimpleNamespace(
            model_id=_DEPLOYMENT, model_name="test-native", model_info={}, litellm_params=dict(params),
        )]) == 1
    deployment: Final = router.get_deployment(_DEPLOYMENT)
    assert deployment is not None
    target: Final = resolve_prediction_target(deployment.litellm_params)
    assert isinstance(target, NativePredictionTarget)
    body: Final = _body()
    with respx.mock() as upstream:
        native: Final = upstream.post("https://api.anthropic.com/v1/messages").respond(200, json={
            "id": "msg_test", "type": "message", "role": "assistant", "model": _MODEL,
            "content": [{"type": "text", "text": "Hello"}], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 11, "output_tokens": 1, "cache_read_input_tokens": 300},
        })
        counter: Final = upstream.post("https://api.anthropic.com/v1/messages/count_tokens").respond(
            200, json={"input_tokens": 311},
        )
        await router.aanthropic_messages(
            model="test-native", max_tokens=1, **{key: value for key, value in body.items() if key != "model"},
        )
        assert await count_prompt_tokens(target.model, target.api_key, body) == 311
    assert native.call_count == counter.call_count == 1
    assert native.calls.last.request.headers["x-api-key"] == counter.calls.last.request.headers["x-api-key"] == _KEY
    observed: Final = parse_observed_cache(native.calls.last.request, ModelResponse(
        model=_MODEL, usage=Usage(
            prompt_tokens=311, completion_tokens=1, total_tokens=312,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=300),
        ),
    ), _CALLER, _DEPLOYMENT)
    assert observed is not None
    assert observed.scope == cache_scope(_CALLER, _DEPLOYMENT, target.api_key, target.model)


@pytest.mark.parametrize("inline_key", [None, _KEY])
@pytest.mark.asyncio
async def test_named_credential_is_explicitly_unsupported_before_count(
    inline_key: str | None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "credential_list", [CredentialItem(
        credential_name="test-named", credential_info={}, credential_values={"api_key": "test-named-provider-key"},
    )])
    deployment: Final = Deployment(
        model_name="test-native",
        litellm_params=LiteLLM_Params(
            model=f"anthropic/{_MODEL}", api_key=inline_key, litellm_credential_name="test-named",
        ),
        model_info=ModelInfo(id=_DEPLOYMENT),
    )
    body: Final = _body()
    prefix: Final = parse_prompt(body)
    assert prefix is not None

    async def count(model: str, api_key: str, body: Mapping[str, JsonValue]) -> int | None:
        pytest.fail("Unsupported named credentials must not reach provider counting")

    arm: Final = await predict_arm(deployment, body, prefix, _CALLER, DualCache(), count)
    assert arm.cache_state == "unknown"
    assert arm.reason == "unsupported_deployment_configuration"
    assert arm.estimate is None and arm.cold is None and arm.warm is None
