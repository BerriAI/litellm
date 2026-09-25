"""
Tests for order-based fallback routing.

When deployments have `order` set in litellm_params, lower order deployments
should be tried first, and higher order deployments should be used as fallbacks
when lower order deployments fail.
"""

import json
from typing import Final, Optional, cast
from unittest.mock import AsyncMock, patch

import httpx
import openai
import pytest
from openai import AsyncOpenAI
from pydantic import ValidationError

import litellm
from litellm import Router
from litellm.integrations.custom_logger import CustomLogger
from litellm.router import (
    _coerce_http_status_code,
    _evaluate_order_fallback_status_policy,
    _log_order_fallback_status_policy_decision,
    _normalize_order_fallback_status_codes,
    _order_fallback_policy_decision,
    _order_fallback_provider_status_code,
    _order_fallback_status_matches,
)
from litellm.router_utils.prompt_caching_cache import PromptCachingCache
from litellm.types.router import (
    OrderFallbackStatusCodes,
    RouterRateLimitError,
    RouterRateLimitErrorBasic,
    UpdateRouterConfig,
)
from litellm.utils import _get_deployment_order, get_order_filtered_deployments

# ---------------------------------------------------------------------------
# Unit tests for get_order_filtered_deployments
# ---------------------------------------------------------------------------


class TestGetOrderFilteredDeployments:
    def _make_deployment(self, order: Optional[int], dep_id: str) -> dict:
        params: dict = {"model": "gpt-4o", "api_key": "key"}
        if order is not None:
            params["order"] = order
        return {
            "model_name": "test-model",
            "litellm_params": params,
            "model_info": {"id": dep_id},
        }

    def test_returns_min_order_group(self):
        deps = [
            self._make_deployment(1, "a"),
            self._make_deployment(2, "b"),
            self._make_deployment(1, "c"),
        ]
        result = get_order_filtered_deployments(deps)
        assert len(result) == 2
        assert all(d["model_info"]["id"] in ("a", "c") for d in result)

    def test_target_order_filters_to_exact_level(self):
        deps = [
            self._make_deployment(1, "a"),
            self._make_deployment(2, "b"),
            self._make_deployment(3, "c"),
        ]
        result = get_order_filtered_deployments(deps, target_order=2)
        assert len(result) == 1
        assert result[0]["model_info"]["id"] == "b"

    def test_target_order_no_match_returns_empty(self):
        deps = [
            self._make_deployment(1, "a"),
            self._make_deployment(2, "b"),
        ]
        result = get_order_filtered_deployments(deps, target_order=99)
        assert result == []

    def test_target_order_no_match_does_not_reselect_lower_order(self):
        deps = [
            self._make_deployment(1, "a"),
            self._make_deployment(2, "b"),
        ]
        remaining_after_pre_call = [deps[0]]
        result = get_order_filtered_deployments(remaining_after_pre_call, target_order=2)
        assert result == []

    def test_no_order_set_returns_all(self):
        deps = [
            self._make_deployment(None, "a"),
            self._make_deployment(None, "b"),
        ]
        result = get_order_filtered_deployments(deps)
        assert len(result) == 2

    def test_empty_list(self):
        result = get_order_filtered_deployments([])
        assert result == []

    def test_single_order_returns_all_with_that_order(self):
        deps = [
            self._make_deployment(1, "a"),
            self._make_deployment(1, "b"),
        ]
        result = get_order_filtered_deployments(deps)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Integration tests for order-based fallback in Router
# ---------------------------------------------------------------------------


class _StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(status_code)


class _ResponseStatusError(Exception):
    def __init__(self, status_code: int) -> None:
        self.response = httpx.Response(
            status_code=status_code,
            request=httpx.Request("POST", "https://provider.example/v1"),
        )
        super().__init__(status_code)


def _status_policy_router(
    *,
    status_code: int,
    order_fallback_status_codes: OrderFallbackStatusCodes | None,
    external_fallback: bool = False,
    enable_weighted_failover: bool = False,
) -> Router:
    model_list: list[dict[str, object]] = [
        {
            "model_name": "test-model",
            "litellm_params": {
                "model": "gpt-4o",
                "api_key": "bad-key",
                "mock_response": _StatusError(status_code),
                "order": 1,
            },
            "model_info": {"id": "order-1"},
        },
        {
            "model_name": "test-model",
            "litellm_params": {
                "model": "gpt-4o",
                "api_key": "good-key",
                "mock_response": "success from order 2",
                "order": 2,
            },
            "model_info": {"id": "order-2"},
        },
    ]
    if external_fallback:
        model_list.append(
            {
                "model_name": "external-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "external-key",
                    "mock_response": "success from external fallback",
                },
                "model_info": {"id": "external"},
            }
        )
    return Router(
        model_list=model_list,
        num_retries=0,
        order_fallback_status_codes=order_fallback_status_codes,
        fallbacks=[{"test-model": ["external-model"]}] if external_fallback else [],
        enable_weighted_failover=enable_weighted_failover,
    )


def test_order_fallback_status_codes_config_round_trip_and_update() -> None:
    config = UpdateRouterConfig(order_fallback_status_codes=[429, "5xx"])
    router = Router(model_list=[], order_fallback_status_codes=config.order_fallback_status_codes)

    assert router.order_fallback_status_codes == (429, "5xx")
    assert router.get_settings()["order_fallback_status_codes"] == [429, "5xx"]

    router.update_settings(order_fallback_status_codes=[])
    assert router.order_fallback_status_codes == ()
    assert router.get_settings()["order_fallback_status_codes"] == []

    router.update_settings(order_fallback_status_codes=None)
    assert router.order_fallback_status_codes is None
    assert router.get_settings()["order_fallback_status_codes"] is None


@pytest.mark.parametrize(
    ("raw_status", "expected"),
    [(429, 429), ("503", 503), (True, None), (99, None), (600, None), ("5xx", None)],
)
def test_coerce_http_status_code(raw_status: object, expected: int | None) -> None:
    assert _coerce_http_status_code(raw_status) == expected


def test_order_fallback_status_policy_helpers() -> None:
    configured = _normalize_order_fallback_status_codes([429, "5xx"])
    assert configured == (429, "5xx")
    assert configured is not None
    assert _order_fallback_status_matches(configured, 429)
    assert _order_fallback_status_matches(configured, 599)
    assert not _order_fallback_status_matches(configured, 401)

    assert _order_fallback_provider_status_code(_StatusError(409)) == 409
    assert _order_fallback_provider_status_code(_ResponseStatusError(503)) == 503
    assert _order_fallback_provider_status_code(RouterRateLimitErrorBasic(model="test-model")) is None
    assert (
        _order_fallback_provider_status_code(
            litellm.APIConnectionError(message="connection failed", model="test-model", llm_provider="openai")
        )
        is None
    )
    assert (
        _order_fallback_provider_status_code(
            litellm.Timeout(message="request timed out", model="test-model", llm_provider="openai")
        )
        is None
    )
    timeout_response: Final = httpx.Response(
        status_code=408,
        request=httpx.Request("POST", "https://provider.example/v1"),
    )
    assert (
        _order_fallback_provider_status_code(
            litellm.Timeout(
                message="provider request timeout",
                model="test-model",
                llm_provider="openai",
                response=timeout_response,
            )
        )
        == 408
    )

    dedicated, status_code, policy_match, skip = _evaluate_order_fallback_status_policy(
        error=_StatusError(401),
        configured_status_codes=configured,
    )
    assert (dedicated, status_code, policy_match, skip) == (False, 401, False, True)
    assert _order_fallback_policy_decision(
        dedicated_fallback=False,
        router_error=False,
        policy_match=policy_match,
        provider_status_code=status_code,
        target_order=2,
    ) == ("not_matched", "status_code_not_allowed")


def test_log_order_fallback_status_policy_decision() -> None:
    with patch("litellm.router.verbose_router_logger.debug") as debug:
        _log_order_fallback_status_policy_decision(
            error=_StatusError(429),
            configured_status_codes=(429,),
            policy_match=True,
            provider_status_code=429,
            dedicated_fallback=False,
            order_values=[1, 2],
            current_target=None,
        )

    debug.assert_called_once()


@pytest.mark.parametrize("invalid_value", [True, "429", 99, 600, "5XX", "400-499", "4xx"])
def test_order_fallback_status_codes_reject_invalid_values(invalid_value: object) -> None:
    invalid_status_codes = cast(OrderFallbackStatusCodes, [invalid_value])

    with pytest.raises(ValidationError):
        Router(model_list=[], order_fallback_status_codes=invalid_status_codes)
    with pytest.raises(ValidationError):
        UpdateRouterConfig(order_fallback_status_codes=invalid_status_codes)


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 401, 403, 404])
async def test_unset_order_fallback_status_codes_preserves_existing_4xx_fallback(status_code: int) -> None:
    router = _status_policy_router(status_code=status_code, order_fallback_status_codes=None)

    response = await router.acompletion(model="test-model", messages=[{"role": "user", "content": "hi"}])

    assert response._hidden_params["model_id"] == "order-2"


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [408, 409, 429, 500, 502, 503, 504, 599])
async def test_order_fallback_status_codes_allow_matching_statuses(status_code: int) -> None:
    router = _status_policy_router(
        status_code=status_code,
        order_fallback_status_codes=[408, 409, 429, "5xx"],
    )

    response = await router.acompletion(model="test-model", messages=[{"role": "user", "content": "hi"}])

    assert response._hidden_params["model_id"] == "order-2"


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 401, 403, 404])
async def test_order_fallback_status_codes_reject_non_matching_statuses(status_code: int) -> None:
    router = _status_policy_router(
        status_code=status_code,
        order_fallback_status_codes=[408, 409, 429, "5xx"],
    )

    with pytest.raises(openai.APIError) as exc_info:
        await router.acompletion(model="test-model", messages=[{"role": "user", "content": "hi"}])

    assert exc_info.value.status_code == status_code


@pytest.mark.asyncio
async def test_empty_order_fallback_status_codes_disable_order_fallback() -> None:
    router = _status_policy_router(status_code=429, order_fallback_status_codes=[])

    with pytest.raises(openai.APIError) as exc_info:
        await router.acompletion(model="test-model", messages=[{"role": "user", "content": "hi"}])

    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_kind",
    ["connection", "timeout"],
    ids=["connection-error-500", "timeout-408"],
)
async def test_synthetic_status_codes_do_not_trigger_order_fallback(error_kind: str) -> None:
    synthetic_error: Final[Exception] = (
        litellm.APIConnectionError(message="connection failed", model="test-model", llm_provider="openai")
        if error_kind == "connection"
        else litellm.Timeout(message="request timed out", model="test-model", llm_provider="openai")
    )
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "bad",
                    "order": 1,
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "good",
                    "order": 2,
                },
                "model_info": {"id": "2"},
            },
        ],
        num_retries=0,
        order_fallback_status_codes=[408, "5xx"],
    )

    provider_call: Final = AsyncMock(side_effect=synthetic_error)
    with patch("litellm.acompletion", provider_call):
        with pytest.raises(openai.APIConnectionError):
            await router.acompletion(model="test-model", messages=[{"role": "user", "content": "hi"}])

    provider_call.assert_awaited_once()


def test_order_fallback_status_codes_apply_to_sync_completion() -> None:
    router = _status_policy_router(status_code=429, order_fallback_status_codes=[429])

    response = router.completion(model="test-model", messages=[{"role": "user", "content": "hi"}])

    assert response._hidden_params["model_id"] == "order-2"


@pytest.mark.asyncio
async def test_non_matching_status_still_uses_external_fallback() -> None:
    router = _status_policy_router(
        status_code=400,
        order_fallback_status_codes=[429, "5xx"],
        external_fallback=True,
    )

    response = await router.acompletion(model="test-model", messages=[{"role": "user", "content": "hi"}])

    assert response._hidden_params["model_id"] == "external"


@pytest.mark.asyncio
async def test_order_fallback_status_codes_do_not_control_weighted_failover() -> None:
    router = _status_policy_router(
        status_code=400,
        order_fallback_status_codes=[429],
        enable_weighted_failover=True,
    )

    weighted_failover = AsyncMock(return_value=None)
    with patch.object(router, "_maybe_run_weighted_failover", new=weighted_failover):
        with pytest.raises(openai.APIError):
            await router.acompletion(model="test-model", messages=[{"role": "user", "content": "hi"}])

    weighted_failover.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mock_response", "context_window_fallbacks", "content_policy_fallbacks"),
    [
        ("litellm.ContextWindowExceededError", [{"test-model": ["dedicated-model"]}], []),
        (
            "Exception: content_filter_policy invalid_request_error content_policy_violation",
            [],
            [{"test-model": ["dedicated-model"]}],
        ),
    ],
)
async def test_status_policy_preserves_dedicated_fallbacks(
    mock_response: str,
    context_window_fallbacks: list[dict[str, list[str]]],
    content_policy_fallbacks: list[dict[str, list[str]]],
) -> None:
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad-key",
                    "mock_response": mock_response,
                    "order": 1,
                },
                "model_info": {"id": "order-1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "order-key",
                    "mock_response": "order fallback should not run",
                    "order": 2,
                },
                "model_info": {"id": "order-2"},
            },
            {
                "model_name": "dedicated-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "dedicated-key",
                    "mock_response": "dedicated fallback succeeded",
                },
                "model_info": {"id": "dedicated"},
            },
        ],
        num_retries=0,
        order_fallback_status_codes=[429],
        context_window_fallbacks=context_window_fallbacks,
        content_policy_fallbacks=content_policy_fallbacks,
    )

    response = await router.acompletion(model="test-model", messages=[{"role": "user", "content": "hi"}])

    assert response._hidden_params["model_id"] == "dedicated"


def test_router_order_without_pre_call_checks():
    """Order filtering should work even when enable_pre_call_checks=False (default)."""
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key",
                    "mock_response": "from order 1",
                    "order": 1,
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key",
                    "mock_response": "from order 2",
                    "order": 2,
                },
                "model_info": {"id": "2"},
            },
        ],
        num_retries=0,
        enable_pre_call_checks=False,
    )

    for _ in range(20):
        response = router.completion(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert response._hidden_params["model_id"] == "1"


def test_router_order_no_fallback_when_healthy():
    """When order=1 is healthy, order=2 should never be used."""
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key",
                    "mock_response": "from order 1",
                    "order": 1,
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key",
                    "mock_response": "from order 2",
                    "order": 2,
                },
                "model_info": {"id": "2"},
            },
        ],
        num_retries=0,
    )

    for _ in range(50):
        response = router.completion(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert response._hidden_params["model_id"] == "1"


@pytest.mark.asyncio
async def test_router_order_fallback_on_failure():
    """When order=1 fails, order=2 should be tried as fallback."""
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad-key",
                    "mock_response": Exception("connection error"),
                    "order": 1,
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "good-key",
                    "mock_response": "success from order 2",
                    "order": 2,
                },
                "model_info": {"id": "2"},
            },
        ],
        num_retries=0,
    )

    response = await router.acompletion(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert response._hidden_params["model_id"] == "2"


@pytest.mark.asyncio
async def test_router_order_fallback_three_levels():
    """When order=1 and order=2 both fail, order=3 should be tried."""
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("fail 1"),
                    "order": 1,
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("fail 2"),
                    "order": 2,
                },
                "model_info": {"id": "2"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "good",
                    "mock_response": "success from order 3",
                    "order": 3,
                },
                "model_info": {"id": "3"},
            },
        ],
        num_retries=0,
        order_fallback_status_codes=["5xx"],
    )

    response = await router.acompletion(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert response._hidden_params["model_id"] == "3"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("second_status_code", "should_reach_order_3"),
    [(401, False), (503, True)],
)
async def test_order_fallback_status_policy_is_enforced_for_each_order_hop(
    second_status_code: int,
    should_reach_order_3: bool,
) -> None:
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": _StatusError(429),
                    "order": 1,
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": _StatusError(second_status_code),
                    "order": 2,
                },
                "model_info": {"id": "2"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "good",
                    "mock_response": "success from order 3",
                    "order": 3,
                },
                "model_info": {"id": "3"},
            },
        ],
        num_retries=0,
        order_fallback_status_codes=[429, "5xx"],
    )

    if should_reach_order_3:
        response = await router.acompletion(model="test-model", messages=[{"role": "user", "content": "hi"}])
        assert response._hidden_params["model_id"] == "3"
        return

    with pytest.raises(openai.APIError) as exc_info:
        await router.acompletion(model="test-model", messages=[{"role": "user", "content": "hi"}])
    assert exc_info.value.status_code == second_status_code


@pytest.mark.asyncio
async def test_router_order_fallback_then_external_fallback():
    """When all order levels fail, external fallbacks should be tried."""
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("fail order 1"),
                    "order": 1,
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("fail order 2"),
                    "order": 2,
                },
                "model_info": {"id": "2"},
            },
            {
                "model_name": "fallback-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "good",
                    "mock_response": "success from external fallback",
                },
                "model_info": {"id": "fallback"},
            },
        ],
        fallbacks=[{"test-model": ["fallback-model"]}],
        num_retries=0,
    )

    response = await router.acompletion(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert response._hidden_params["model_id"] == "fallback"


@pytest.mark.asyncio
async def test_router_order_fallback_with_non_standard_fallbacks():
    """Non-standard fallback formats (e.g. fallbacks=["model-name"]) passed
    per-request should still be tried after all order levels are exhausted."""
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("fail order 1"),
                    "order": 1,
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("fail order 2"),
                    "order": 2,
                },
                "model_info": {"id": "2"},
            },
            {
                "model_name": "fallback-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "good",
                    "mock_response": "success from non-standard fallback",
                },
                "model_info": {"id": "fallback"},
            },
        ],
        num_retries=0,
    )

    response = await router.acompletion(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        fallbacks=["fallback-model"],  # non-standard format, passed per-request
    )
    assert response._hidden_params["model_id"] == "fallback"


@pytest.mark.asyncio
async def test_router_order_fallback_with_wildcard_model_group():
    """Wildcard model groups should also advance across order levels."""
    router = Router(
        model_list=[
            {
                "model_name": "openai/*",
                "litellm_params": {
                    "model": "openai/*",
                    "api_key": "bad",
                    "mock_response": Exception("fail order 1"),
                    "order": 1,
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": "openai/*",
                "litellm_params": {
                    "model": "openai/*",
                    "api_key": "good",
                    "mock_response": "success from wildcard order 2",
                    "order": 2,
                },
                "model_info": {"id": "2"},
            },
        ],
        num_retries=0,
    )

    response = await router.acompletion(
        model="openai/gpt-4.1-mini",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert response._hidden_params["model_id"] == "2"


@pytest.mark.asyncio
async def test_router_order_fallback_with_hidden_model_group_alias():
    router = Router(
        model_list=[
            {
                "model_name": "canonical-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("fail order 1"),
                    "order": 1,
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": "canonical-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "good",
                    "mock_response": "success from order 2",
                    "order": 2,
                },
                "model_info": {"id": "2"},
            },
        ],
        model_group_alias={"hidden-alias": {"model": "canonical-model", "hidden": True}},
        num_retries=0,
    )

    assert "hidden-alias" not in {deployment["model_name"] for deployment in router.get_model_list() or []}

    response = await router.acompletion(
        model="hidden-alias",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert response._hidden_params["model_id"] == "2"


@pytest.mark.asyncio
async def test_router_order_fallback_does_not_reselect_order_1_when_order_2_is_filtered_out():
    class _DropOrder2(CustomLogger):
        async def async_filter_deployments(
            self, model, healthy_deployments, messages, request_kwargs=None, parent_otel_span=None
        ):
            return [d for d in healthy_deployments if _get_deployment_order(d) != 2]

    drop_order_2: Final = _DropOrder2()
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key",
                    "mock_response": "litellm.RateLimitError",
                    "order": 1,
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key",
                    "mock_response": "success from order 2",
                    "order": 2,
                },
                "model_info": {"id": "2"},
            },
        ],
        num_retries=0,
        order_fallback_status_codes=[429],
    )
    litellm.callbacks.append(drop_order_2)
    try:
        with pytest.raises(RouterRateLimitError, match="No deployments available") as exc_info:
            await router.acompletion(
                model="test-model",
                messages=[{"role": "user", "content": "hi"}],
            )
        assert "success from order 2" not in str(exc_info.value)
    finally:
        litellm.callbacks.remove(drop_order_2)


@pytest.mark.asyncio
async def test_router_order_fallback_ignores_prompt_cache_pin_on_target_order():
    messages = [{"role": "user", "content": "word " * 5000}]
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("azure peak load"),
                    "order": 1,
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "good",
                    "mock_response": "success from order 2",
                    "order": 2,
                },
                "model_info": {"id": "2"},
            },
        ],
        num_retries=0,
        optional_pre_call_checks=["prompt_caching"],
    )
    await PromptCachingCache(cache=router.cache).async_add_model_id(
        model_id="1",
        messages=messages,
        tools=None,
    )
    response = await router.acompletion(model="test-model", messages=messages)
    assert response._hidden_params["model_id"] == "2"


@pytest.mark.asyncio
async def test_router_order_fallback_retries_keep_target_order():
    seen_target_orders: Final = []

    class _RecordTargetOrder(CustomLogger):
        async def async_filter_deployments(
            self, model, healthy_deployments, messages, request_kwargs=None, parent_otel_span=None
        ):
            seen_target_orders.append((request_kwargs or {}).get("_target_order"))
            return healthy_deployments

    recorder: Final = _RecordTargetOrder()
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("fail order 1"),
                    "order": 1,
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("fail order 2"),
                    "order": 2,
                },
                "model_info": {"id": "2"},
            },
        ],
        num_retries=1,
        order_fallback_status_codes=["5xx"],
    )
    litellm.callbacks.append(recorder)
    try:
        with pytest.raises(Exception, match="fail order 2"):
            await router.acompletion(
                model="test-model",
                messages=[{"role": "user", "content": "hi"}],
            )
    finally:
        litellm.callbacks.remove(recorder)
    assert seen_target_orders.count(2) >= 2


@pytest.mark.asyncio
async def test_generic_api_call_strips_target_order_from_provider_kwargs():
    captured: Final = {}

    async def _fake_provider(**provider_kwargs):
        captured.update(provider_kwargs)
        return "ok"

    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-4o", "api_key": "key", "order": 2},
                "model_info": {"id": "2"},
            },
        ],
    )
    response = await router._ageneric_api_call_with_fallbacks_helper(
        model="test-model",
        original_generic_function=_fake_provider,
        _target_order=2,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert response == "ok"
    assert captured["model"] == "gpt-4o"
    assert "_target_order" not in captured


@pytest.mark.asyncio
async def test_text_completion_order_fallback_hop_does_not_send_target_order_upstream():
    upstream_bodies: Final[list[dict]] = []

    def _upstream(request: httpx.Request) -> httpx.Response:
        upstream_bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "cmpl-1",
                "object": "text_completion",
                "created": 0,
                "model": "gpt-3.5-turbo-instruct",
                "choices": [{"text": "ok from order 2", "index": 0, "logprobs": None, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    upstream_client: Final = AsyncOpenAI(
        api_key="key",
        base_url="http://upstream.test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_upstream)),
    )
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "text-completion-openai/gpt-3.5-turbo-instruct",
                    "api_key": "key",
                    "mock_response": Exception("fail order 1"),
                    "order": 1,
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "text-completion-openai/gpt-3.5-turbo-instruct",
                    "api_key": "key",
                    "api_base": "http://upstream.test",
                    "order": 2,
                },
                "model_info": {"id": "2"},
            },
        ],
        num_retries=0,
    )
    try:
        response = await router.atext_completion(model="test-model", prompt="hi", client=upstream_client)
    finally:
        await upstream_client.close()

    assert response._hidden_params["model_id"] == "2"
    assert upstream_bodies
    assert all("_target_order" not in body for body in upstream_bodies)


def test_check_non_standard_fallback_format():
    from litellm.router_utils.fallback_event_handlers import (
        _check_non_standard_fallback_format,
    )

    # Standard formats
    assert _check_non_standard_fallback_format([{"gpt-3.5-turbo": ["claude-3-haiku"]}]) == False
    assert _check_non_standard_fallback_format([{"model": ["qwen-backup"]}]) == False
    assert _check_non_standard_fallback_format([{"model": ["qwen-backup"], "region": ["us-east-1"]}]) == False

    # Non-standard formats
    assert _check_non_standard_fallback_format([{"model": "qwen-backup"}]) == True
    assert (
        _check_non_standard_fallback_format([{"model": "qwen-backup", "messages": [{"role": "user", "content": "hi"}]}])
        == True
    )
    assert _check_non_standard_fallback_format([{"model": ["qwen-backup"], "api_key": "some-key"}]) == True
