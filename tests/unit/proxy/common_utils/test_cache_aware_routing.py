from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import pytest
from pydantic import JsonValue

from litellm import Router
from litellm.caching.dual_cache import DualCache
from litellm.llms.anthropic.prompt_cache_prediction import TokenCounter, cache_scope, parse_prompt
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.cache_aware_routing import (
    CacheAwareChoice,
    choose_cached_model,
    eligible_models,
    select_cached_model,
)
from litellm.proxy.hooks.prompt_cache_prediction import CacheObservation, _cache_key
from litellm.router_strategy.complexity_router.complexity_router import ComplexityRouter
from litellm.router_strategy.complexity_router.config import ComplexityRouterConfig
from litellm.types.router import PreRoutingHookResponse

_CALLER: Final = "test-cache-aware-caller"
_PROVIDER_KEY: Final = "test-cache-aware-provider"
_NOW: Final = 1000.0


@dataclass(frozen=True, slots=True)
class _Counts:
    total: int | None = 51000
    prefix: int | None = 50000

    async def __call__(self, model: str, api_key: str, body: Mapping[str, JsonValue]) -> int | None:
        return self.total if "max_tokens" in body else self.prefix


def _counter_for_model(model: str) -> TokenCounter:
    return _Counts()


def _forbidden_counter(model: str) -> TokenCounter:
    raise AssertionError("No provider counts should run without a warm eligible alternative")


def _body(text: str = "Stable cached context") -> dict[str, JsonValue]:
    return {
        "max_tokens": 20000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "What is 2 + 2?"},
                ],
            }
        ],
    }


def _router(
    strong_output_rate: float = 0.000015, *, free: bool = False, cheap_limit: int = 30000, strong_limit: int = 30000
) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "cheap",
                "litellm_params": {
                    "model": "anthropic/claude-haiku-4-5",
                    "api_key": _PROVIDER_KEY,
                    "input_cost_per_token": 0 if free else 0.000001,
                    "output_cost_per_token": 0 if free else 0.000004,
                    "cache_read_input_token_cost": 0 if free else 0.0000001,
                    "cache_creation_input_token_cost": 0 if free else 0.00000125,
                },
                "model_info": {"id": "test-cache-cheap", "max_input_tokens": 100000, "max_output_tokens": cheap_limit},
            },
            {
                "model_name": "strong",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-5",
                    "api_key": _PROVIDER_KEY,
                    "input_cost_per_token": 0 if free else 0.000003,
                    "output_cost_per_token": 0 if free else strong_output_rate,
                    "cache_read_input_token_cost": 0 if free else 0.0000003,
                    "cache_creation_input_token_cost": 0 if free else 0.00000375,
                },
                "model_info": {
                    "id": "test-cache-strong",
                    "max_input_tokens": 100000,
                    "max_output_tokens": strong_limit,
                },
            },
        ]
    )


def _config(**overrides: object) -> ComplexityRouterConfig:
    return ComplexityRouterConfig.model_validate(
        {
            "tiers": {"SIMPLE": "cheap", "COMPLEX": "strong"},
            "cache_aware_routing": True,
            **overrides,
        }
    )


def _response(tier: str = "SIMPLE", model: str = "cheap") -> PreRoutingHookResponse:
    return PreRoutingHookResponse(
        model=model,
        messages=None,
        routing_decision={
            "router_model_name": "smart",
            "router_type": "complexity",
            "routed_model": model,
            "tier": tier,
            "cause": "heuristic_scorer",
        },
    )


async def _observed(cache: DualCache, *, caller: str = _CALLER, expires_at: float = 1290.0) -> None:
    prefix: Final = parse_prompt(_body())
    assert prefix is not None
    scope: Final = cache_scope(caller, "test-cache-strong", _PROVIDER_KEY, "claude-sonnet-5")
    observation: Final = CacheObservation(
        fingerprint=prefix.fingerprint,
        cached_tokens=50000,
        observed_at=990.0,
        expires_at=expires_at,
    )
    await cache.async_set_cache(_cache_key(scope, prefix.fingerprint), observation.model_dump_json(), ttl=3600)


async def _select(
    *,
    router: Router,
    config: ComplexityRouterConfig,
    response: PreRoutingHookResponse,
    body: Mapping[str, JsonValue],
    request_kwargs: Mapping[str, object],
    messages: Sequence[Mapping[str, object]] | None,
    caller: UserAPIKeyAuth,
    cache: DualCache,
    counter_for_model: Callable[[str], TokenCounter],
    now: float,
) -> CacheAwareChoice | None:
    complexity: Final = ComplexityRouter("smart", router, config.model_dump())
    return await select_cached_model(
        router=router,
        config=config,
        params_for_model=complexity._litellm_params_for_model,
        response=response,
        body=body,
        request_kwargs=request_kwargs,
        messages=messages,
        caller=caller,
        cache=cache,
        counter_for_model=counter_for_model,
        now=now,
    )


@pytest.mark.asyncio
async def test_warm_stronger_model_wins_after_counting_input_and_output_cost() -> None:
    cache: Final = DualCache()
    await _observed(cache)
    choice: Final = await _select(
        router=_router(),
        config=_config(),
        response=_response(),
        body=_body(),
        request_kwargs={},
        messages=None,
        caller=UserAPIKeyAuth(api_key=_CALLER, models=["cheap", "strong"]),
        cache=cache,
        counter_for_model=_counter_for_model,
        now=_NOW,
    )
    assert choice is not None
    assert (choice.model, choice.tier, choice.deployment_id) == ("strong", "COMPLEX", "test-cache-strong")
    assert choice.original_cost == pytest.approx(50000 * 0.00000125 + 1000 * 0.000001 + 1024 * 0.000004)
    assert choice.estimated_cost == pytest.approx(50000 * 0.0000003 + 1000 * 0.000003 + 1024 * 0.000015)


@pytest.mark.asyncio
async def test_output_price_can_outweigh_the_cache_saving() -> None:
    cache: Final = DualCache()
    await _observed(cache)
    choice: Final = await _select(
        router=_router(strong_output_rate=0.001),
        config=_config(),
        response=_response(),
        body=_body(),
        request_kwargs={},
        messages=None,
        caller=UserAPIKeyAuth(api_key=_CALLER, models=["cheap", "strong"]),
        cache=cache,
        counter_for_model=_counter_for_model,
        now=_NOW,
    )
    assert choice is None


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["missing", "expired", "different_caller", "changed_prefix", "unauthorized"])
async def test_no_cache_discount_without_fresh_authorized_matching_evidence(case: str) -> None:
    cache: Final = DualCache()
    if case != "missing":
        await _observed(
            cache,
            caller="someone-else" if case == "different_caller" else _CALLER,
            expires_at=999.0 if case == "expired" else 1290.0,
        )
    choice: Final = await _select(
        router=_router(),
        config=_config(),
        response=_response(),
        body=_body("Changed context" if case == "changed_prefix" else "Stable cached context"),
        request_kwargs={},
        messages=None,
        caller=UserAPIKeyAuth(api_key=_CALLER, models=["cheap"] if case == "unauthorized" else ["cheap", "strong"]),
        cache=cache,
        counter_for_model=_forbidden_counter,
        now=_NOW,
    )
    assert choice is None


@pytest.mark.asyncio
async def test_disabled_setting_does_not_access_prediction_services() -> None:
    config: Final = ComplexityRouterConfig(tiers={"SIMPLE": "cheap"})
    assert config.cache_aware_routing is False
    assert (
        await choose_cached_model(
            router=_router(),
            config=config,
            params_for_model=ComplexityRouter("smart", _router(), config.model_dump())._litellm_params_for_model,
            response=_response(),
            request_kwargs={},
            messages=None,
        )
        is None
    )


def test_cache_prices_cannot_add_a_model_below_the_classified_tier() -> None:
    response: Final = _response("COMPLEX", "strong")
    assert response.routing_decision is not None
    assert eligible_models(_config(), response.routing_decision) == (("COMPLEX", "strong"),)


@pytest.mark.parametrize(
    "overrides", [{"adaptive": True}, {"session_affinity": True}, {"classification_mode": "user_turn"}]
)
def test_existing_pinned_or_adaptive_policies_are_preserved(overrides: Mapping[str, object]) -> None:
    response: Final = _response()
    assert response.routing_decision is not None
    assert eligible_models(_config(**overrides), response.routing_decision) == ()


@pytest.mark.parametrize("total,prefix", [(None, 50000), (51000, None), (1000, 50000)])
@pytest.mark.asyncio
async def test_unavailable_or_inconsistent_counts_keep_the_classified_model(
    total: int | None, prefix: int | None
) -> None:
    cache: Final = DualCache()
    await _observed(cache)
    assert (
        await _select(
            router=_router(),
            config=_config(),
            response=_response(),
            body=_body(),
            request_kwargs={},
            messages=None,
            caller=UserAPIKeyAuth(api_key=_CALLER, models=["cheap", "strong"]),
            cache=cache,
            counter_for_model=lambda _: _Counts(total, prefix),
            now=_NOW,
        )
        is None
    )


@pytest.mark.asyncio
async def test_output_estimate_is_capped_by_the_requested_limit() -> None:
    cache: Final = DualCache()
    await _observed(cache)
    choice: Final = await _select(
        router=_router(),
        config=_config(cache_aware_routing_output_tokens=100000, max_tokens_from_tier_model=False),
        response=_response(),
        body={**_body(), "max_tokens": 1},
        request_kwargs={},
        messages=None,
        caller=UserAPIKeyAuth(api_key=_CALLER, models=["cheap", "strong"]),
        cache=cache,
        counter_for_model=_counter_for_model,
        now=_NOW,
    )
    assert choice is not None
    assert choice.estimated_cost == pytest.approx(50000 * 0.0000003 + 1000 * 0.000003 + 0.000015)


@pytest.mark.asyncio
async def test_warm_model_that_cannot_fit_the_request_is_not_selected() -> None:
    cache: Final = DualCache()
    await _observed(cache)
    assert (
        await _select(
            router=_router(),
            config=_config(max_tokens_from_tier_model=False),
            response=_response(),
            body={**_body(), "max_tokens": 100000000},
            request_kwargs={},
            messages=None,
            caller=UserAPIKeyAuth(api_key=_CALLER, models=["cheap", "strong"]),
            cache=cache,
            counter_for_model=_counter_for_model,
            now=_NOW,
        )
        is None
    )


def test_repeated_model_in_multiple_tiers_is_only_considered_once() -> None:
    decision: Final = _response().routing_decision
    assert decision is not None
    assert eligible_models(_config(tiers={"SIMPLE": "cheap", "MEDIUM": "strong", "COMPLEX": "strong"}), decision) == (
        ("SIMPLE", "cheap"),
        ("MEDIUM", "strong"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "enabled,behavior,expected",
    [
        (False, "success", "cheap"),
        (True, "success", "strong"),
        (True, "tier_cost", "cheap"),
        (True, "tier_context", "cheap"),
        (True, "error", "cheap"),
        (True, "deadline", "cheap"),
        (True, "cancel", None),
        (True, "transformed", "cheap"),
        (True, "unsupported_shape", "cheap"),
        (True, "custom_endpoint", "cheap"),
        (True, "compaction", "cheap"),
        (True, "guardrail", "cheap"),
    ],
)
async def test_router_applies_opt_in_and_preserves_failure_semantics(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, behavior: str, expected: str | None
) -> None:
    import asyncio
    import json

    import httpx

    import litellm
    from litellm.caching.llm_caching_handler import LLMClientCache
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    from litellm.proxy import proxy_server
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.hooks.parallel_request_limiter_v3 import _PROXY_MaxParallelRequestsHandler_v3
    from litellm.proxy.utils import ProxyLogging
    from litellm.router_strategy.complexity_router.context_compaction import initialize_compaction_state

    config: Final = _config(
        cache_aware_routing=enabled, cache_aware_routing_timeout_ms=1 if behavior == "deadline" else 2000
    )
    models: Final = _router(
        strong_output_rate=0.000048 if behavior == "tier_cost" else 0.000015,
        cheap_limit=50 if behavior == "tier_cost" else 30000,
        strong_limit=60000 if behavior == "tier_context" else 30000,
    ).get_model_list()
    assert models is not None
    router: Final = Router(
        model_list=[
            *models,
            {
                "model_name": "smart",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": config.model_dump(),
                    **({"temperature": 0.1} if behavior == "transformed" else {}),
                },
            },
        ]
    )
    logging: Final = ProxyLogging(UserApiKeyCache())
    logging.proxy_hook_mapping["parallel_request_limiter"] = _PROXY_MaxParallelRequestsHandler_v3(
        logging.internal_usage_cache
    )
    await _observed(logging.internal_usage_cache.dual_cache, expires_at=1e100)
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", logging)
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())

    requests: Final = asyncio.Queue[httpx.Request]()

    async def count(request: httpx.Request) -> httpx.Response:
        requests.put_nowait(request)
        assert enabled and behavior not in ("transformed", "unsupported_shape", "custom_endpoint")
        if behavior == "error":
            return httpx.Response(503, json={"error": "Provider unavailable"})
        if behavior == "cancel":
            raise asyncio.CancelledError()
        if behavior == "deadline":
            await asyncio.Future()
        payload: Final = json.loads(request.content)
        assert request.url == "https://api.anthropic.com/v1/messages/count_tokens"
        assert request.headers["x-api-key"] == _PROVIDER_KEY
        return httpx.Response(200, json={"input_tokens": 51000 if "What is 2 + 2?" in str(payload) else 50000})

    async with httpx.AsyncClient(transport=httpx.MockTransport(count)) as client:
        handler: Final = AsyncHTTPHandler()
        await handler.client.aclose()
        handler.client = client
        litellm.in_memory_llm_clients_cache.set_cache("async_httpx_clientanthropic", handler)
        body: Final = {
            **_body(),
            **({"max_tokens": 1000} if behavior in ("tier_cost", "tier_context") else {}),
            **({"thinking": {"type": "enabled", "budget_tokens": 10000}} if behavior == "unsupported_shape" else {}),
        }
        kwargs: Final = {
            "litellm_metadata": {
                "user_api_key_auth": UserAPIKeyAuth(api_key=_CALLER, models=["smart", "cheap", "strong"])
            },
            "proxy_server_request": {"url": "http://localhost/v1/messages", "body": body, "headers": {}},
            **({"api_base": "https://custom.example"} if behavior == "custom_endpoint" else {}),
            **(
                {"_context_compaction_state": initialize_compaction_state({}, "messages")}
                if behavior == "compaction"
                else {}
            ),
            **({"guardrails": ["test-guardrail"]} if behavior == "guardrail" else {}),
        }
        if expected is None:
            with pytest.raises(asyncio.CancelledError):
                await router.async_pre_routing_hook(model="smart", request_kwargs=kwargs, messages=body["messages"])
            return
        response: Final = await router.async_pre_routing_hook(
            model="smart", request_kwargs=kwargs, messages=body["messages"]
        )
        assert response is not None
        assert response.model == expected
        if not enabled or behavior in (
            "transformed",
            "unsupported_shape",
            "custom_endpoint",
            "compaction",
            "guardrail",
        ):
            assert requests.qsize() == 0
        if enabled and behavior == "success":
            assert requests.qsize() == 4
        assert response.routing_decision is not None
        assert response.routing_decision["cause"] == (
            "prompt_cache_cost" if expected == "strong" else "heuristic_scorer"
        )


@pytest.mark.asyncio
async def test_equal_costs_keep_the_classified_model() -> None:
    cache: Final = DualCache()
    await _observed(cache)
    assert (
        await _select(
            router=_router(free=True),
            config=_config(),
            response=_response(),
            body=_body(),
            request_kwargs={},
            messages=None,
            caller=UserAPIKeyAuth(api_key=_CALLER, models=["cheap", "strong"]),
            cache=cache,
            counter_for_model=_counter_for_model,
            now=_NOW,
        )
        is None
    )


@pytest.mark.parametrize(
    "cause", ["llm_v2_classifier", "capability_classifier", "heuristic_first_short_circuit", "hybrid_short_circuit"]
)
def test_successful_classifiers_can_consider_cache_costs(cause: str) -> None:
    response: Final = PreRoutingHookResponse.model_validate(
        {
            "model": "cheap",
            "messages": None,
            "routing_decision": {"tier": "SIMPLE", "cause": cause},
        }
    )
    assert response.routing_decision is not None
    assert eligible_models(_config(), response.routing_decision) == (("SIMPLE", "cheap"), ("COMPLEX", "strong"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cheap_limit,strong_limit,requested,from_tier,output_rate,expected_limits",
    [
        (50, 1000, 1000, True, 0.000048, None),
        (30000, 30000, 1, True, 0.000049, None),
        (30000, 60000, 1, True, 0.000015, None),
        (50, 100, 20000, True, 0.000048, (50, 100)),
        (30000, 30000, 1, False, 0.000048, (1, 1)),
    ],
)
async def test_each_candidate_uses_its_effective_routed_output_limit(
    cheap_limit: int,
    strong_limit: int,
    requested: int,
    from_tier: bool,
    output_rate: float,
    expected_limits: tuple[int, int] | None,
) -> None:
    cache: Final = DualCache()
    await _observed(cache)
    choice: Final = await _select(
        router=_router(strong_output_rate=output_rate, cheap_limit=cheap_limit, strong_limit=strong_limit),
        config=_config(max_tokens_from_tier_model=from_tier),
        response=_response(),
        body={**_body(), "max_tokens": requested},
        request_kwargs={},
        messages=None,
        caller=UserAPIKeyAuth(api_key=_CALLER, models=["cheap", "strong"]),
        cache=cache,
        counter_for_model=_counter_for_model,
        now=_NOW,
    )
    if expected_limits is None:
        assert choice is None
        return
    assert choice is not None
    assert choice.original_cost == pytest.approx(50000 * 0.00000125 + 1000 * 0.000001 + expected_limits[0] * 0.000004)
    assert choice.estimated_cost == pytest.approx(
        50000 * 0.0000003 + 1000 * 0.000003 + expected_limits[1] * output_rate
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("warm,authorized", [(False, True), (True, True), (True, False)])
async def test_authorization_only_runs_for_original_and_warm_alternatives_before_provider_counts(
    monkeypatch: pytest.MonkeyPatch, warm: bool, authorized: bool
) -> None:
    from unittest.mock import AsyncMock

    from litellm.proxy.common_utils import cache_aware_routing

    cache: Final = DualCache()
    if warm:
        await _observed(cache)
    models: Final = _router().get_model_list()
    assert models is not None
    router: Final = Router(
        model_list=[
            *models,
            {**models[0], "model_name": "cold", "model_info": {"id": "test-cache-cold"}},
        ]
    )
    authorization: Final = AsyncMock(wraps=cache_aware_routing.can_key_call_resolved_model)
    monkeypatch.setattr(cache_aware_routing, "can_key_call_resolved_model", authorization)

    def counter_for_model(model: str) -> TokenCounter:
        assert warm and authorized
        assert authorization.await_count == 2
        return _Counts()

    choice: Final = await _select(
        router=router,
        config=_config(tiers={"SIMPLE": "cheap", "MEDIUM": "cold", "COMPLEX": "strong"}),
        response=_response(),
        body=_body(),
        request_kwargs={},
        messages=None,
        caller=UserAPIKeyAuth(api_key=_CALLER, models=["cheap", "strong", "cold"] if authorized else ["cheap", "cold"]),
        cache=cache,
        counter_for_model=counter_for_model,
        now=_NOW,
    )
    assert (choice is not None) == (warm and authorized)
    assert tuple(call.kwargs["model"] for call in authorization.await_args_list) == (
        ("cheap", "strong") if warm else ()
    )
