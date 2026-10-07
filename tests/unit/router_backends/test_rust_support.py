from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.router_backends.rust_support import bind_arguments, unsupported_reason


def _deployment(overrides: Mapping[str, object] = MappingProxyType({})) -> Mapping[str, object]:
    return {"model_name": "gpt", "litellm_params": {"model": "openai/gpt-4o-mini"}, **overrides}


@pytest.mark.parametrize(
    "arguments",
    (
        {},
        {"model_list": [_deployment()]},
        {"model_list": [_deployment(), _deployment({"litellm_params": {"model": "openai/gpt-4o", "weight": 2}})]},
        {
            "model_list": [_deployment({"model_info": {"id": "a", "cooldown_time": 3}})],
            "num_retries": 3,
            "fallbacks": [{"gpt": ["other"]}],
            "context_window_fallbacks": [{"gpt": ["big"]}],
            "content_policy_fallbacks": [{"gpt": ["safe"]}],
            "retry_policy": {"RateLimitErrorRetries": 2},
            "model_group_retry_policy": {"gpt": {"TimeoutErrorRetries": 1}},
            "allowed_fails": 2,
            "cooldown_time": 9,
            "disable_cooldowns": False,
            "redis_host": "localhost",
            "redis_port": 6379,
        },
        {"routing_strategy": "simple-shuffle", "enable_pre_call_checks": False, "fallbacks": []},
        {"model_list": [_deployment({"model_name": "team/gpt"})], "fallbacks": [{"team/gpt": ["other"]}]},
    ),
)
def test_supported_configs_have_no_reason(arguments: Mapping[str, object]) -> None:
    assert unsupported_reason(arguments) is None


@pytest.mark.parametrize(
    ("arguments", "named"),
    (
        ({"routing_strategy": "latency-based-routing"}, "'routing_strategy'"),
        ({"enable_pre_call_checks": True}, "'enable_pre_call_checks'"),
        ({"model_group_alias": {"a": "gpt"}}, "'model_group_alias'"),
        ({"model_list": [_deployment({"model_name": "openai/*"})]}, "wildcard"),
        ({"model_list": [_deployment({"litellm_params": {"model": "auto_router/x"}})]}, "strategy router"),
        ({"model_list": [_deployment({"litellm_params": {"model": "m", "order": 1}})]}, "litellm_params.order"),
        ({"model_list": [_deployment({"litellm_params": {"model": "m", "rpm": 10}})]}, "litellm_params.rpm"),
        ({"model_list": [_deployment({"model_info": {"team_id": "t"}})]}, "model_info.team_id"),
        ({"model_list": [_deployment({"tpm": 100})]}, "tpm"),
        ({"model_list": [{"litellm_params": {"model": "m"}}]}, "cannot read"),
        ({"model_list": [_deployment()], "fallbacks": [{"openai/gpt": ["other"]}]}, "fallbacks entries"),
        ({"fallbacks": [{"gpt": [{"model": "other"}]}]}, "fallbacks entries"),
        ({"context_window_fallbacks": [{"gpt": "big"}]}, "context_window_fallbacks entries"),
    ),
)
def test_unsupported_configs_name_the_reason(arguments: Mapping[str, object], named: str) -> None:
    reason: Final = unsupported_reason(arguments)

    assert reason is not None
    assert named in reason


def test_global_model_alias_map_is_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "model_alias_map", {"a": "gpt"})

    assert unsupported_reason({}) == "litellm.model_alias_map is set"


async def _filter_nothing(*_: object) -> tuple[()]:
    return ()


_Filtering: Final = type("_Filtering", (CustomLogger,), {"async_filter_deployments": _filter_nothing})


def test_callbacks_that_filter_deployments_are_unsupported_but_plain_loggers_are_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [CustomLogger()])
    assert unsupported_reason({}) is None

    monkeypatch.setattr(litellm, "callbacks", [CustomLogger(), _Filtering()])
    assert unsupported_reason({}) == "callbacks _Filtering filter or pre-check deployments"


def test_bind_arguments_maps_positional_and_keyword_arguments_by_name() -> None:
    assert bind_arguments(([_deployment()],), {"num_retries": 1}) == {"model_list": [_deployment()], "num_retries": 1}
