import asyncio
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Final, Literal

import httpx
import pytest
from fastapi import FastAPI, Request
from pydantic import JsonValue

import litellm
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body, _safe_set_request_parsed_body
from litellm.proxy.hooks.parallel_request_limiter_v3 import _PROXY_MaxParallelRequestsHandler_v3
from litellm.llms.anthropic.prompt_cache_prediction import PromptPrefix, cache_scope, parse_prompt
from litellm.proxy.hooks.prompt_cache_prediction import (
    CacheObservation,
    _cache_key,
)
from litellm.proxy.management_endpoints import prompt_cache_prediction as endpoint
from litellm.proxy.utils import InternalUsageCache
from litellm.types.management_endpoints.prompt_cache_prediction import CachePredictionResponse
from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo


_PROVIDER_KEY: Final = "cache-prediction-test-provider-key"
_CALLER: Final = "cache-prediction-test-caller-hash"


@pytest.fixture(autouse=True)
def anthropic_endpoint_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_BASE", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)


def _body(ttl: str = "5m", *, extended: bool = False) -> dict[str, JsonValue]:
    blocks: Final[list[JsonValue]] = [
        {"type": "text", "text": "Stable context"},
        *([{"type": "text", "text": "Appended context"}] if extended else []),
    ]
    return {
        "max_tokens": 10,
        "system": "Follow the project conventions",
        "messages": [
            {
                "role": "user",
                "content": [
                    *blocks[:-1],
                    {**blocks[-1], "cache_control": {"type": "ephemeral", "ttl": ttl}},
                    {"type": "text", "text": "Follow-up question"},
                ],
            }
        ],
    }


def _prefix(body: Mapping[str, JsonValue]) -> PromptPrefix:
    prefix: Final = parse_prompt(body)
    assert prefix is not None
    return prefix


def _deployment(
    deployment_id: str = "sonnet",
    model: str = "claude-sonnet-5",
    *,
    team_id: str | None = None,
    api_base: str | None = None,
) -> Deployment:
    return Deployment(
        model_name=deployment_id,
        litellm_params=LiteLLM_Params(model=f"anthropic/{model}", api_key=_PROVIDER_KEY, api_base=api_base),
        model_info=ModelInfo(id=deployment_id, team_id=team_id),
    )


@dataclass(frozen=True)
class Counts:
    total: int | None = 6_000
    prefix: int | None = 5_000

    async def __call__(self, model: str, api_key: str, body: Mapping[str, JsonValue]) -> int | None:
        assert api_key == _PROVIDER_KEY
        assert model.startswith("claude-")
        return self.total if "max_tokens" in body else self.prefix


async def _observe(
    cache: DualCache,
    body: Mapping[str, JsonValue],
    *,
    deployment_id: str = "sonnet",
    model: str = "claude-sonnet-5",
    cached_tokens: int = 5_000,
    expired: bool = False,
    caller: str = _CALLER,
) -> None:
    prefix: Final = _prefix(body)
    now: Final = time.time()
    observation: Final = CacheObservation(
        fingerprint=prefix.fingerprint,
        cached_tokens=cached_tokens,
        observed_at=now - 400 if expired else now - 10,
        expires_at=now - 100 if expired else now + 290,
    )
    scope: Final = cache_scope(caller, deployment_id, _PROVIDER_KEY, model)
    await cache.async_set_cache(_cache_key(scope, prefix.fingerprint), observation.model_dump_json(), ttl=3_600)


@pytest.mark.asyncio
@pytest.mark.parametrize(("ttl", "cold_cost"), [("5m", 0.0145), ("1h", 0.022)])
async def test_unobserved_cache_prices_cold_and_warm_bounds(ttl: str, cold_cost: float) -> None:
    body: Final = _body(ttl)
    arm: Final = await endpoint.predict_arm(_deployment(), body, _prefix(body), _CALLER, DualCache(), Counts())

    assert arm.cache_state == "unknown"
    assert arm.reason == "no_compatible_observation"
    assert arm.evidence is None
    assert arm.estimate is not None and arm.cold is not None and arm.warm is not None
    assert arm.estimate.input_cost == pytest.approx(cold_cost)
    assert arm.cold.input_cost == pytest.approx(cold_cost)
    assert arm.warm.input_cost == pytest.approx(0.003)
    assert arm.cold.tokens.uncached_input_tokens == 1_000
    assert arm.cold.tokens.cache_read_input_tokens == 0
    assert arm.cold.tokens.cache_creation_5m_input_tokens == (5_000 if ttl == "5m" else 0)
    assert arm.cold.tokens.cache_creation_1h_input_tokens == (5_000 if ttl == "1h" else 0)
    assert arm.warm.tokens.cache_read_input_tokens == 5_000


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cached_tokens", "warm_cost", "cold_cost"), [(5_400, 0.00228, 0.0147), (4_600, 0.00372, 0.0143)]
)
@pytest.mark.parametrize("expired", [False, True])
async def test_exact_prefix_conserves_total_with_observed_count_in_all_scenarios(
    cached_tokens: int, warm_cost: float, cold_cost: float, expired: bool
) -> None:
    cache: Final = DualCache()
    body: Final = _body()
    await _observe(cache, body, cached_tokens=cached_tokens, expired=expired)
    arm: Final = await endpoint.predict_arm(_deployment(), body, _prefix(body), _CALLER, cache, Counts())

    assert arm.cache_state == ("stale" if expired else "warm")
    assert arm.evidence is not None
    assert arm.estimate is not None and arm.warm is not None and arm.cold is not None
    assert arm.warm.tokens.cache_read_input_tokens == cached_tokens
    assert arm.warm.tokens.cache_creation_5m_input_tokens == 0
    assert arm.cold.tokens.cache_creation_5m_input_tokens == cached_tokens
    assert arm.cold.tokens.cache_read_input_tokens == 0
    for scenario in (arm.estimate, arm.cold, arm.warm):
        assert scenario.tokens.total_tokens == 6_000
        assert scenario.tokens.uncached_input_tokens == 6_000 - cached_tokens
    assert arm.warm.input_cost == pytest.approx(warm_cost)
    assert arm.cold.input_cost == pytest.approx(cold_cost)
    assert arm.estimate.input_cost == pytest.approx(cold_cost if expired else warm_cost)


@pytest.mark.asyncio
async def test_observed_prefix_larger_than_full_request_returns_unknown() -> None:
    cache: Final = DualCache()
    body: Final = _body()
    await _observe(cache, body, cached_tokens=6_001)
    arm: Final = await endpoint.predict_arm(_deployment(), body, _prefix(body), _CALLER, cache, Counts())

    assert arm.cache_state == "unknown"
    assert arm.reason == "inconsistent_prefix_token_count"
    assert arm.estimate is None and arm.cold is None and arm.warm is None


@pytest.mark.asyncio
@pytest.mark.parametrize(("ttl", "expected"), [("5m", 0.0053), ("1h", 0.0068)])
async def test_append_only_prefix_reads_old_tokens_and_writes_extension(ttl: str, expected: float) -> None:
    cache: Final = DualCache()
    await _observe(cache, _body(ttl), cached_tokens=4_000)
    body: Final = _body(ttl, extended=True)
    arm: Final = await endpoint.predict_arm(_deployment(), body, _prefix(body), _CALLER, cache, Counts())

    assert arm.cache_state == "partial"
    assert arm.estimate is not None
    assert arm.estimate.tokens.cache_read_input_tokens == 4_000
    assert arm.estimate.tokens.cache_creation_5m_input_tokens == (1_000 if ttl == "5m" else 0)
    assert arm.estimate.tokens.cache_creation_1h_input_tokens == (1_000 if ttl == "1h" else 0)
    assert arm.estimate.input_cost == pytest.approx(expected)


@pytest.mark.asyncio
async def test_expired_observation_estimates_a_cold_rebuild() -> None:
    cache: Final = DualCache()
    body: Final = _body()
    await _observe(cache, body, expired=True)
    arm: Final = await endpoint.predict_arm(_deployment(), body, _prefix(body), _CALLER, cache, Counts())

    assert arm.cache_state == "stale"
    assert arm.reason == "observation_expired"
    assert arm.evidence is not None and arm.evidence.expires_at < time.time()
    assert arm.estimate is not None and arm.cold is not None
    assert arm.estimate.tokens.cache_read_input_tokens == 0
    assert arm.estimate.tokens.cache_creation_5m_input_tokens == 5_000
    assert arm.estimate.input_cost == arm.cold.input_cost


@pytest.mark.asyncio
async def test_below_model_minimum_prices_all_input_as_uncached() -> None:
    body: Final = _body()
    arm: Final = await endpoint.predict_arm(
        _deployment(), body, _prefix(body), _CALLER, DualCache(), Counts(total=1_500, prefix=1_000)
    )

    assert arm.cache_state == "disabled"
    assert arm.reason == "below_cache_minimum"
    assert arm.estimate is not None
    assert arm.estimate.tokens.uncached_input_tokens == 1_500
    assert arm.estimate.tokens.cache_read_input_tokens == 0
    assert arm.estimate.tokens.cache_creation_5m_input_tokens == 0
    assert arm.estimate.input_cost == pytest.approx(0.003)


@pytest.mark.asyncio
@pytest.mark.parametrize("counts", [Counts(total=None), Counts(prefix=None), Counts(total=4_000)])
async def test_unavailable_or_inconsistent_token_counts_return_null_estimates(counts: Counts) -> None:
    body: Final = _body()
    arm: Final = await endpoint.predict_arm(_deployment(), body, _prefix(body), _CALLER, DualCache(), counts)

    assert arm.cache_state == "unknown"
    assert arm.reason == "token_count_unavailable"
    assert arm.estimate is None and arm.cold is None and arm.warm is None


@pytest.mark.asyncio
@pytest.mark.parametrize("counts", [Counts(), Counts(total=1_500, prefix=1_000)])
async def test_missing_prices_return_unknown_and_null_estimates(
    monkeypatch: pytest.MonkeyPatch, counts: Counts
) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        "claude-cache-unpriced-5",
        {"litellm_provider": "anthropic", "mode": "chat"},
    )
    body: Final = _body()
    arm: Final = await endpoint.predict_arm(
        _deployment("cache-prediction-unpriced", "claude-cache-unpriced-5"),
        body,
        _prefix(body),
        _CALLER,
        DualCache(),
        counts,
    )

    assert arm.cache_state == "unknown"
    assert arm.reason == "pricing_unavailable"
    assert arm.estimate is None and arm.cold is None and arm.warm is None


@pytest.mark.asyncio
async def test_custom_api_base_from_environment_returns_unknown_before_counting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_BASE", "https://custom.invalid")
    body: Final = _body()
    arm: Final = await endpoint.predict_arm(
        _deployment(), body, _prefix(body), _CALLER, DualCache(), _unexpected_count
    )

    assert arm.cache_state == "unknown"
    assert arm.reason == "unsupported_provider_endpoint"
    assert arm.estimate is None and arm.cold is None and arm.warm is None


@pytest.mark.asyncio
async def test_explicit_official_api_base_overrides_custom_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_BASE", "https://custom.invalid")
    body: Final = _body()
    arm: Final = await endpoint.predict_arm(
        _deployment(api_base="https://api.anthropic.com"), body, _prefix(body), _CALLER, DualCache(), Counts()
    )

    assert arm.cache_state == "unknown"
    assert arm.reason == "no_compatible_observation"
    assert arm.estimate is not None
    assert arm.estimate.input_cost == pytest.approx(0.0145)


@dataclass(frozen=True)
class _ProxyLogging:
    internal_usage_cache: InternalUsageCache
    parallel_limiter: CustomLogger | None

    def get_proxy_hook(self, hook: str) -> CustomLogger | None:
        return self.parallel_limiter if hook == "parallel_request_limiter" else None


def _app(
    monkeypatch: pytest.MonkeyPatch,
    cache: DualCache,
    *,
    caller: UserAPIKeyAuth | None = None,
    current_team: str | None = None,
    candidate_team: str | None = None,
    counts: endpoint.TokenCounter = Counts(),
    limiter: CustomLogger | Literal["default"] | None = "default",
) -> FastAPI:
    import litellm.proxy.proxy_server as proxy_server

    model_list: Final = [
        _deployment("opus", "claude-opus-5", team_id=current_team).model_dump(exclude_unset=True),
        _deployment("sonnet", team_id=candidate_team).model_dump(exclude_unset=True),
    ]
    router: Final = litellm.Router(model_list=model_list)
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "llm_model_list", model_list)
    monkeypatch.setattr(endpoint, "count_prompt_tokens", counts)
    app: Final = FastAPI()
    app.include_router(endpoint.router)
    app.add_exception_handler(ProxyException, proxy_server.openai_exception_handler)
    if caller is not None:
        usage_cache: Final = InternalUsageCache(cache)
        configured_limiter: Final = (
            _PROXY_MaxParallelRequestsHandler_v3(usage_cache) if isinstance(limiter, str) else limiter
        )
        monkeypatch.setattr(proxy_server, "proxy_logging_obj", _ProxyLogging(usage_cache, configured_limiter))
        app.dependency_overrides[endpoint.user_api_key_auth] = lambda: caller
    return app


async def _post(
    app: FastAPI,
    body: Mapping[str, JsonValue],
    *,
    current_deployment_id: str = "opus",
    candidate_deployment_id: str = "sonnet",
) -> httpx.Response:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        return await client.post(
            "/cost/predict-cache",
            json={
                "current_deployment_id": current_deployment_id,
                "candidate_deployment_id": candidate_deployment_id,
                "request": body,
            },
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("warm_deployment", "warm_model", "expected_delta", "expected_penalty"),
    [("sonnet", "claude-sonnet-5", -0.03325, 0.0), ("opus", "claude-opus-5", 0.007, 0.0115)],
)
async def test_switch_delta_accounts_for_each_deployment_cache(
    monkeypatch: pytest.MonkeyPatch,
    warm_deployment: str,
    warm_model: str,
    expected_delta: float,
    expected_penalty: float,
) -> None:
    cache: Final = DualCache()
    body: Final = _body()
    await _observe(cache, body, deployment_id=warm_deployment, model=warm_model)
    app: Final = _app(monkeypatch, cache, caller=UserAPIKeyAuth(api_key=_CALLER))
    response: Final = await _post(app, body)

    assert response.status_code == 200, response.text
    result: Final = CachePredictionResponse.model_validate(response.json())
    assert result.switch_delta == pytest.approx(expected_delta)
    assert result.cache_rebuild_penalty == pytest.approx(expected_penalty)
    assert result.cache_guarantee is False
    assert result.pricing_basis == "input_before_discounts_and_margins"
    if warm_deployment == "sonnet":
        assert result.switch.cache_state == "warm"
        assert result.stay.cache_state == "unknown"
    else:
        assert result.stay.cache_state == "warm"
        assert result.switch.cache_state == "unknown"


@pytest.mark.asyncio
async def test_missing_caller_identity_cannot_reuse_observations(monkeypatch: pytest.MonkeyPatch) -> None:
    cache: Final = DualCache()
    body: Final = _body()
    await _observe(cache, body)
    response: Final = await _post(
        _app(monkeypatch, cache, caller=UserAPIKeyAuth(api_key=None), counts=_unexpected_count), body
    )

    assert response.status_code == 200, response.text
    result: Final = CachePredictionResponse.model_validate(response.json())
    assert result.stay.reason == result.switch.reason == "caller_identity_unavailable"
    assert result.stay.estimate is None and result.switch.estimate is None
    assert result.switch_delta is None and result.cache_rebuild_penalty is None


@pytest.mark.asyncio
async def test_unauthenticated_request_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "master_key", "cache-prediction-test-master-key")
    response: Final = await _post(_app(monkeypatch, DualCache()), _body())
    assert response.status_code == 401, response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("arm", ["current", "candidate"])
@pytest.mark.parametrize("caller_team", [None, "own-team"])
@pytest.mark.parametrize("restricted", [False, True])
async def test_foreign_and_missing_deployments_have_identical_authenticated_responses(
    monkeypatch: pytest.MonkeyPatch, arm: str, caller_team: str | None, restricted: bool
) -> None:
    allowed: Final = ("sonnet",) if arm == "current" else ("opus",)
    app: Final = _app(
        monkeypatch,
        DualCache(),
        caller=UserAPIKeyAuth(api_key=_CALLER, team_id=caller_team, models=list(allowed) if restricted else []),
        current_team="foreign-team" if arm == "current" else None,
        candidate_team="foreign-team" if arm == "candidate" else None,
        counts=_unexpected_count,
    )
    foreign: Final = await _post(app, _body())
    missing: Final = await _post(
        app,
        _body(),
        current_deployment_id="missing-deployment" if arm == "current" else "opus",
        candidate_deployment_id="missing-deployment" if arm == "candidate" else "sonnet",
    )

    assert foreign.status_code == missing.status_code == 404
    assert foreign.json() == missing.json() == {"detail": "Deployment not found"}


@pytest.mark.asyncio
@pytest.mark.parametrize("deployment_team", [None, "own-team"])
async def test_visible_public_and_own_team_deployments_remain_available(
    monkeypatch: pytest.MonkeyPatch, deployment_team: str | None
) -> None:
    app: Final = _app(
        monkeypatch,
        DualCache(),
        caller=UserAPIKeyAuth(api_key=_CALLER, team_id="own-team"),
        current_team=deployment_team,
        candidate_team=deployment_team,
    )
    response: Final = await _post(app, _body())

    assert response.status_code == 200, response.text
    result: Final = CachePredictionResponse.model_validate(response.json())
    assert result.stay.estimate is not None and result.switch.estimate is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("arm", ["current", "candidate"])
async def test_visible_deployment_outside_key_model_permissions_is_forbidden(
    monkeypatch: pytest.MonkeyPatch, arm: str
) -> None:
    allowed: Final = "sonnet" if arm == "current" else "opus"
    denied: Final = "opus" if arm == "current" else "sonnet"
    app: Final = _app(monkeypatch, DualCache(), caller=UserAPIKeyAuth(api_key=_CALLER, models=[allowed]))
    response: Final = await _post(app, _body())
    assert response.status_code == 403, response.text
    assert denied in response.text


@pytest.mark.asyncio
async def test_other_callers_warm_cache_is_not_prediction_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    cache: Final = DualCache()
    body: Final = _body()
    await _observe(cache, body, caller="other-caller")
    response: Final = await _post(_app(monkeypatch, cache, caller=UserAPIKeyAuth(api_key=_CALLER)), body)

    assert response.status_code == 200, response.text
    result: Final = CachePredictionResponse.model_validate(response.json())
    assert result.switch.cache_state == "unknown"
    assert result.switch.reason == "no_compatible_observation"
    assert result.switch.evidence is None
    assert result.switch.estimate is not None
    assert result.switch.estimate.tokens.cache_read_input_tokens == 0


@pytest.mark.asyncio
async def test_count_failure_nulls_switch_comparison(monkeypatch: pytest.MonkeyPatch) -> None:
    app: Final = _app(
        monkeypatch, DualCache(), caller=UserAPIKeyAuth(api_key=_CALLER), counts=Counts(total=None)
    )
    response: Final = await _post(app, _body())

    assert response.status_code == 200, response.text
    result: Final = CachePredictionResponse.model_validate(response.json())
    assert result.stay.reason == result.switch.reason == "token_count_unavailable"
    assert result.stay.estimate is None and result.switch.estimate is None
    assert result.switch_delta is None and result.cache_rebuild_penalty is None


@pytest.mark.asyncio
@pytest.mark.parametrize("limiter", [None, CustomLogger()])
async def test_missing_or_unsupported_limiter_returns_unknown_before_counting(
    monkeypatch: pytest.MonkeyPatch, limiter: CustomLogger | None
) -> None:
    app: Final = _app(
        monkeypatch, DualCache(), caller=UserAPIKeyAuth(api_key=_CALLER), counts=_unexpected_count, limiter=limiter
    )
    response: Final = await _post(app, _body())

    assert response.status_code == 200, response.text
    result: Final = CachePredictionResponse.model_validate(response.json())
    assert result.stay.reason == result.switch.reason == "limiter_unavailable"
    assert result.stay.estimate is None and result.switch.estimate is None
    assert result.switch_delta is None and result.cache_rebuild_penalty is None


@pytest.mark.asyncio
async def test_occupied_parallel_capacity_rejects_before_provider_count(monkeypatch: pytest.MonkeyPatch) -> None:
    cache: Final = DualCache()
    limiter: Final = _PROXY_MaxParallelRequestsHandler_v3(InternalUsageCache(cache))
    caller: Final = UserAPIKeyAuth(api_key=_CALLER, max_parallel_requests=1)
    app: Final = _app(monkeypatch, cache, caller=caller, counts=_unexpected_count, limiter=limiter)
    async with limiter.request_capacity(caller, "opus"):
        response: Final = await _post(app, _body())

    assert response.status_code == 429, response.text
    assert "max_parallel_requests" in response.text
    recovered: Final = await _post(_app(monkeypatch, cache, caller=caller, limiter=limiter), _body())
    assert recovered.status_code == 200, recovered.text


@pytest.mark.asyncio
async def test_each_count_consumes_the_deployment_group_rpm_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: Final = asyncio.Queue[str]()

    async def count(model: str, api_key: str, body: Mapping[str, JsonValue]) -> int | None:
        calls.put_nowait(model)
        return await Counts()(model, api_key, body)

    caller: Final = UserAPIKeyAuth(api_key=_CALLER, metadata={"model_rpm_limit": {"sonnet": 1}})
    app: Final = _app(monkeypatch, DualCache(), caller=caller, counts=count)
    response: Final = await _post(app, _body())

    assert response.status_code == 429, response.text
    assert calls.qsize() == 3
    assert tuple(calls.get_nowait() for _ in range(3)) == (
        "claude-opus-5", "claude-opus-5", "claude-sonnet-5"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_key", ["metadata", "litellm_metadata"])
async def test_each_count_preserves_auth_cached_request_tag_limits(
    monkeypatch: pytest.MonkeyPatch, metadata_key: str
) -> None:
    calls: Final = asyncio.Queue[str]()
    caller: Final = UserAPIKeyAuth(api_key=_CALLER, metadata={"tag_rpm_limit": {"cache-cost": 1}})

    async def count(model: str, api_key: str, body: Mapping[str, JsonValue]) -> int | None:
        calls.put_nowait(model)
        return await Counts()(model, api_key, body)

    async def authenticated_request(request: Request) -> UserAPIKeyAuth:
        data: Final = await _read_request_body(request)
        _safe_set_request_parsed_body(request, {**data, metadata_key: {"tags": ["cache-cost"]}})
        return caller

    app: Final = _app(monkeypatch, DualCache(), caller=caller, counts=count)
    app.dependency_overrides[endpoint.user_api_key_auth] = authenticated_request
    response: Final = await _post(app, _body())

    assert response.status_code == 429, response.text
    assert "tag_per_key" in response.text
    assert calls.qsize() == 1
    assert calls.get_nowait() == "claude-opus-5"


@pytest.mark.asyncio
async def test_provider_counter_failure_releases_parallel_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    cache: Final = DualCache()
    limiter: Final = _PROXY_MaxParallelRequestsHandler_v3(InternalUsageCache(cache))
    caller: Final = UserAPIKeyAuth(api_key=_CALLER, max_parallel_requests=1)

    async def fail_count(model: str, api_key: str, body: Mapping[str, JsonValue]) -> int | None:
        raise RuntimeError("provider counter failed")

    app: Final = _app(monkeypatch, cache, caller=caller, counts=fail_count, limiter=limiter)
    with pytest.raises(RuntimeError, match="provider counter failed"):
        await _post(app, _body())
    recovered: Final = await _post(_app(monkeypatch, cache, caller=caller, limiter=limiter), _body())
    assert recovered.status_code == 200, recovered.text
    assert recovered.json()["switch"]["estimate"]["input_cost"] == pytest.approx(0.0145)


@pytest.mark.asyncio
async def test_cancelled_provider_counter_releases_parallel_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    cache: Final = DualCache()
    limiter: Final = _PROXY_MaxParallelRequestsHandler_v3(InternalUsageCache(cache))
    caller: Final = UserAPIKeyAuth(api_key=_CALLER, max_parallel_requests=1)
    started: Final = asyncio.Event()
    release: Final = asyncio.Event()

    async def wait_count(model: str, api_key: str, body: Mapping[str, JsonValue]) -> int | None:
        started.set()
        await release.wait()
        return await Counts()(model, api_key, body)

    app: Final = _app(monkeypatch, cache, caller=caller, counts=wait_count, limiter=limiter)
    pending: Final = asyncio.create_task(_post(app, _body()))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        release.set()
        recovered: Final = await asyncio.wait_for(_post(app, _body()), timeout=5)
        assert recovered.status_code == 200, recovered.text
        assert recovered.json()["switch"]["estimate"]["input_cost"] == pytest.approx(0.0145)
    finally:
        pending.cancel()
        release.set()
        await asyncio.gather(pending, return_exceptions=True)


async def _unexpected_count(model: str, api_key: str, body: Mapping[str, JsonValue]) -> int | None:
    pytest.fail("Unsupported prediction must return before contacting the token counter")


class RequestMutator(CustomLogger):
    async def async_pre_call_hook(
        self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict[str, object], call_type: str
    ) -> dict[str, object]:
        return {**data, "system": "Injected policy"}


@pytest.fixture
def request_mutator() -> Iterator[RequestMutator]:
    callback: Final = RequestMutator()
    litellm.logging_callback_manager.add_litellm_callback(callback)
    try:
        yield callback
    finally:
        litellm.logging_callback_manager.remove_callback_from_all_lists(callback)


@pytest.mark.asyncio
async def test_request_transform_callback_returns_unknown_before_token_counting(
    monkeypatch: pytest.MonkeyPatch, request_mutator: RequestMutator
) -> None:
    app: Final = _app(
        monkeypatch, DualCache(), caller=UserAPIKeyAuth(api_key=_CALLER), counts=_unexpected_count
    )
    response: Final = await _post(app, _body())

    assert response.status_code == 200, response.text
    result: Final = CachePredictionResponse.model_validate(response.json())
    assert result.stay.cache_state == result.switch.cache_state == "unknown"
    assert result.stay.reason == result.switch.reason == "unsupported_request_transform"
    assert result.stay.estimate is None and result.switch.estimate is None
    assert result.switch_delta is None and result.cache_rebuild_penalty is None


@pytest.mark.asyncio
async def test_key_config_returns_unknown_before_token_counting(monkeypatch: pytest.MonkeyPatch) -> None:
    app: Final = _app(
        monkeypatch,
        DualCache(),
        caller=UserAPIKeyAuth(api_key=_CALLER, config={"model_list": []}),
        counts=_unexpected_count,
    )
    response: Final = await _post(app, _body())

    assert response.status_code == 200, response.text
    result: Final = CachePredictionResponse.model_validate(response.json())
    assert result.stay.cache_state == result.switch.cache_state == "unknown"
    assert result.stay.reason == result.switch.reason == "unsupported_request_transform"
    assert result.stay.estimate is None and result.switch.estimate is None
    assert result.switch_delta is None and result.cache_rebuild_penalty is None


@pytest.mark.parametrize("headers", [
    {"anthropic-version": "2099-01-01"},
    {"anthropic-beta": "future-feature"},
])
@pytest.mark.asyncio
async def test_unsupported_provider_headers_cannot_reuse_default_version_evidence(
    monkeypatch: pytest.MonkeyPatch, headers: dict[str, str]
) -> None:
    cache: Final = DualCache()
    await _observe(cache, _body(), deployment_id="sonnet")
    app: Final = _app(
        monkeypatch, cache, caller=UserAPIKeyAuth(api_key=_CALLER), counts=_unexpected_count
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response: Final = await client.post(
            "/cost/predict-cache",
            headers=headers,
            json={"current_deployment_id": "opus", "candidate_deployment_id": "sonnet", "request": _body()},
        )
    assert response.status_code == 200, response.text
    result: Final = CachePredictionResponse.model_validate(response.json())
    assert result.stay.cache_state == result.switch.cache_state == "unknown"
    assert result.stay.reason == result.switch.reason == "unsupported_provider_headers"
    assert result.stay.estimate is None and result.switch.estimate is None
    assert result.switch_delta is None and result.cache_rebuild_penalty is None
