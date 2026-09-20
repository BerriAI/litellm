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
    CountedPromptCachePlan,
    NativePredictionTarget,
    PromptCachePlan,
    UnsupportedCachePlan,
    cache_scope,
    count_cache_plan,
    count_prompt_tokens,
    parse_cache_plan,
    parse_observed_cache,
    parse_prompt,
    resolve_baseline_prediction_target,
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


def _cache_plan(body: Mapping[str, JsonValue]) -> PromptCachePlan:
    plan: Final = parse_cache_plan(body)
    assert isinstance(plan, PromptCachePlan)
    return plan


def _text(text: str, ttl: str | None = None) -> dict[str, JsonValue]:
    return {"type": "text", "text": text,
            **({"cache_control": {"type": "ephemeral", "ttl": ttl}} if ttl else {})}


def _prompt(*blocks: dict[str, JsonValue], role: str = "user", **options: JsonValue) -> dict[str, JsonValue]:
    return {**options, "messages": [{"role": role, "content": list(blocks)}]}


@pytest.mark.parametrize("text, supported", [("", False), (" \t", False), ("Context", True)])
def test_public_predictor_preserves_string_message_policy(text: str, supported: bool) -> None:
    body: Final = _body()
    messages: Final = body["messages"]
    assert isinstance(messages, list)
    request: Final[dict[str, JsonValue]] = {**body, "messages": [{"role": "user", "content": text}, *messages]}
    assert (parse_prompt(request) is not None) is supported


def test_cache_plan_preserves_hierarchical_prefixes_and_public_policy() -> None:
    body: Final = _prompt(
        _text("First turn", "5m"), system=[_text("Stable instructions", "1h")],
        tools=[{"name": "lookup", "input_schema": {"type": "object"},
                "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
    )
    plan: Final = _cache_plan(body)
    changed: Final = _cache_plan({**body, "system": [_text("Changed instructions", "1h")]})
    assert tuple(marker.ttl_seconds for marker in plan.breakpoints) == (3600, 3600, 300)
    assert plan.breakpoints[0].fingerprint == changed.breakpoints[0].fingerprint
    assert all(left.fingerprint != right.fingerprint for left, right
               in zip(plan.breakpoints[1:], changed.breakpoints[1:]))
    assert plan.breakpoints[0].prefix_body == {
        "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
        "messages": [],
    }
    assert parse_prompt(body) is None


@pytest.mark.parametrize("kind, added, matches", [
    ("text", 19, True), ("text", 20, False), ("tool_use", 30, True),
    ("tool_result", 30, True),
])
def test_cache_plan_lookback_counts_native_positions(
    kind: str, added: int, matches: bool,
) -> None:
    previous: Final = _cache_plan(_body())
    appended: Final[list[dict[str, JsonValue]]] = [
        {"type": "tool_use", "id": f"tool_{index}", "name": "lookup", "input": {}}
        if kind == "tool_use" else
        {"type": "tool_result", "tool_use_id": f"tool_{index}", "content": "done"}
        if kind == "tool_result" else
        {"type": "text", "text": f"Added {index}"}
        for index in range(added)
    ]
    current: Final = _cache_plan({**_body(), **_prompt(
        _text("A cacheable prefix"), *appended[:-1],
        {**appended[-1], "cache_control": {"type": "ephemeral"}},
    )})
    assert (previous.breakpoints[0].fingerprint
            in current.breakpoints[0].lookback_fingerprints) is matches


@pytest.mark.parametrize("change, same_prefix, same_content", [
    ("tool_order", False, False), ("effort", False, False),
    ("standard_speed", True, True), ("ttl", False, True),
])
def test_cache_plan_identity_respects_settings_and_preserves_content(
    change: str, same_prefix: bool, same_content: bool,
) -> None:
    tool_input: Final[dict[str, JsonValue]] = {"a": 1, "b": 2, "cache_control": {"ttl": "user-data"}}
    block: Final[dict[str, JsonValue]] = {
        "type": "tool_use", "id": "tool_1", "name": "lookup", "input": tool_input,
        "cache_control": {"type": "ephemeral", "ttl": "5m"},
    }
    changed_block: Final = (
        {**block, "input": dict(reversed(tool_input.items()))} if change == "tool_order" else
        {**block, "cache_control": {"type": "ephemeral", "ttl": "1h"}} if change == "ttl" else block
    )
    before: Final = _cache_plan(_prompt(block, role="assistant", output_config={"effort": "low"})).breakpoints[0]
    after: Final = _cache_plan(_prompt(
        changed_block, role="assistant", output_config={"effort": "high" if change == "effort" else "low"},
        **({"speed": "standard"} if change == "standard_speed" else {}),
    )).breakpoints[0]
    assert (before.fingerprint == after.fingerprint) is same_prefix
    assert (before.fingerprint in after.lookback_fingerprints) is same_prefix
    assert (before.content_fingerprint == after.content_fingerprint) is same_content
    assert (before.content_fingerprint in after.lookback_content_fingerprints) is same_content
    assert "user-data" in json.dumps(dict(before.prefix_body))
    assert not supported_prediction_headers({"anthropic-beta": "fast-mode-2026-02-01"})


def test_cache_plan_automatic_cache_and_thinking_use_last_cacheable_block() -> None:
    body: Final = _prompt(
        _text("A stable answer"), {"type": "thinking", "thinking": "Thinking", "signature": "signature"},
        role="assistant", thinking={"type": "adaptive"}, cache_control={"type": "ephemeral", "ttl": "1h"},
    )
    plan: Final = _cache_plan(body)
    assert len(plan.breakpoints) == 1
    assert plan.breakpoints[0].ttl_seconds == 3600
    assert plan.breakpoints[0].prefix_body == {
        "thinking": {"type": "adaptive"},
        "messages": [{"role": "assistant", "content": [
            {"type": "text", "text": "A stable answer"},
        ]}],
    }
    assert parse_prompt(body) is None


@pytest.mark.parametrize("body, reason", [
    (_prompt({"type": "image"}), "unsupported_prompt_shape"),
    ({**_body(), "unknown_native_setting": True}, "unsupported_prompt_shape"),
    ({**_body(), "cache_control": {"type": "ephemeral", "ttl": "1h"}},
     "conflicting_cache_ttl"),
    (_prompt(_text("five", "5m"), _text("hour", "1h")), "invalid_cache_ttl_order"),
])
def test_cache_plan_unsupported_is_explicit(
    body: Mapping[str, JsonValue], reason: str,
) -> None:
    result: Final = parse_cache_plan(body)
    assert isinstance(result, UnsupportedCachePlan)
    assert result.reason == reason


@pytest.mark.asyncio
@pytest.mark.parametrize("model, counts, reason", [
    (None, (100, 150, 200), None),
    (None, (100, 201, 200), "inconsistent_prefix_token_count"),
    (None, (151, 150, 200), "inconsistent_prefix_token_count"),
    (None, (None, 150, 200), "token_count_unavailable"),
    ("claude-opus-5", (100, 150, 200), None),
    ("claude-sonnet-5", (100, 150, 200), None),
    ("declared-cache-model", (100, 150, 200), None),
    ("unknown-cache-model", (100, 150, 200), "unsupported_thinking_cache_semantics"),
    ("claude-haiku-4-5", (100, 150, 200), "unsupported_thinking_cache_semantics"),
    ("claude-sonnet-4-5", (100, 150, 200), "unsupported_thinking_cache_semantics"),
])
async def test_cache_plan_count_conserves_total_and_rejects_unknown(
    model: str | None, counts: tuple[int | None, int | None, int], reason: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(litellm.model_cost, "declared-cache-model", {
        "litellm_provider": "anthropic", "mode": "chat", "supports_thinking_cache_preservation": True,
    })
    plan: Final = _cache_plan(_prompt(
        {"type": "thinking", "thinking": "Retained thought", "signature": "signature"}
        if model else _text("first", "5m"),
        _text("second", "5m"), _text("uncached"), role="assistant" if model else "user",
    ))

    async def count(model: str, api_key: str, body: Mapping[str, JsonValue]) -> int | None:
        assert reason != "unsupported_thinking_cache_semantics", "Unverified thinking retention must skip counting"
        if body is plan.full_body:
            return counts[2]
        return counts[0] if body is plan.breakpoints[0].prefix_body else counts[1]

    result: Final = await count_cache_plan(model or _MODEL, _KEY, plan, count)
    if reason is not None:
        assert isinstance(result, UnsupportedCachePlan)
        assert result.reason == reason
    else:
        assert isinstance(result, CountedPromptCachePlan)
        assert result.total_tokens == 200
        assert tuple(marker.prefix_tokens for marker in result.breakpoints) == ((100,) if model else (100, 150))


@pytest.mark.asyncio
@pytest.mark.parametrize("section", ["system", "tools"])
@pytest.mark.parametrize("rejects_prefix", (False, True))
async def test_native_count_preserves_settings_and_requires_every_prefix(
    section: str, rejects_prefix: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    params: Final = LiteLLM_Params(
        model=f"anthropic/{_MODEL}", api_key=_KEY,
        api_base="https://gateway.example/v1/messages",
    )
    target: Final = resolve_baseline_prediction_target(params)
    assert isinstance(target, NativePredictionTarget)
    assert target.api_base == params.api_base
    assert not isinstance(resolve_prediction_target(params), NativePredictionTarget)
    body: Final = _body()
    marker: Final[dict[str, JsonValue]] = {"type": "ephemeral", "ttl": "1h"}
    body[section] = ([_text("A cached system", "1h")] if section == "system" else [{
        "name": "lookup", "input_schema": {"type": "object"}, "cache_control": marker,
    }])
    plan: Final = _cache_plan({**body, **_prompt(
        _text("A later prefix", "5m"), _text("An uncached suffix"),
        thinking={"type": "adaptive"}, tool_choice={"type": "auto"}, output_config={"effort": "high"},
    )})
    assert len(plan.breakpoints) == 2
    assert plan.breakpoints[0].prefix_body["messages"] == []

    async def count(model: str, api_key: str, body: Mapping[str, JsonValue]) -> int | None:
        return await count_prompt_tokens(
            model, api_key, {**body, "max_tokens": 100}, api_base=target.api_base,
        )

    with respx.mock(assert_all_called=False) as upstream:
        endpoint: Final = "https://gateway.example/v1/messages/count_tokens"
        routes: Final = tuple(
            upstream.post(endpoint, json={**body, "model": _MODEL}).respond(
                400 if rejects_prefix and index == 1 else 200,
                json={"detail": {"error": "messages parameter is required"}}
                if rejects_prefix and index == 1 else {"input_tokens": tokens},
            )
            for index, (body, tokens) in enumerate((
                (plan.full_body, 6000), (plan.breakpoints[0].prefix_body, 5000),
                (plan.breakpoints[1].prefix_body, 5800),
            ))
        )
        unexpected: Final = upstream.post(endpoint).respond(200, json={"input_tokens": 1})
        result: Final = await count_cache_plan(target.model, target.api_key, plan, count)

    if rejects_prefix:
        assert result == UnsupportedCachePlan("token_count_unavailable")
    else:
        assert isinstance(result, CountedPromptCachePlan)
        assert result.total_tokens == 6000
        assert tuple(marker.prefix_tokens for marker in result.breakpoints) == (5000, 5800)
    assert tuple(route.call_count for route in routes) == (1, 1, 1)
    assert unexpected.call_count == 0
