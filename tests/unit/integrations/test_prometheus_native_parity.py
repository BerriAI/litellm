import asyncio
import copy
import json
import logging
import os
import re
import subprocess
import sys
import traceback
from collections.abc import Awaitable, Callable, Generator, Mapping
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Final, Literal, TypeAlias
from zoneinfo import ZoneInfo

import pytest
from prometheus_client import REGISTRY, CollectorRegistry, generate_latest, multiprocess
from prometheus_client.openmetrics.exposition import generate_latest as generate_openmetrics
from prometheus_client.registry import Collector

import litellm
from litellm._logging import verbose_logger
from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_logger_v2.enums import Hook
from litellm.integrations.custom_logger_v2.call import CallContext
from litellm.integrations.custom_logger_v2.derivations import DERIVED_ATTEMPT_FIELDS
from litellm.integrations.custom_logger_v2.field_sources import HookArgs, with_sources
from litellm.integrations.custom_logger_v2.legacy import Admit, KwargsFields, hook_subscribers, run_subscribers
from litellm.integrations.prometheus import PrometheusLogger
from litellm.litellm_core_utils.litellm_logging import LOGGING_KEYS, Logging, _in_memory_loggers
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.integrations.prometheus_helpers.native_store import (
    NativeFamilyCollector,
    NativeTicket,
    PrometheusCompanion,
    Stopped,
)
from litellm.proxy._types import LiteLLM_TeamTable, LiteLLM_UserTable, Member, UserAPIKeyAuth
from litellm.proxy.common_utils.callback_utils import initialize_callbacks_on_proxy
from litellm.proxy.utils import ProxyLogging
from litellm.router_utils.cooldown_callbacks import get_prometheus_logger_from_callbacks
from litellm.rust_bridge import _native
from litellm.types.integrations.custom_logger_v2 import Outcome
from litellm.utils import add_configured_callback_to_specific_event

START: Final = datetime(2026, 11, 13, 9, 0, 0)  # noqa: DTZ001  # callback timestamps use the legacy naive contract
_CREATED: Final = re.compile(r"^(\S+_created(?:\{.*\})?) \S+$", re.MULTILINE)
_SETTINGS: Final = (
    "custom_prometheus_metadata_labels",
    "custom_prometheus_tags",
    "prometheus_emit_stream_label",
    "enable_end_user_cost_tracking_prometheus_only",
    "prometheus_end_user_metrics_max_series_per_metric",
    "prometheus_end_user_metrics_ttl_seconds",
    "prometheus_end_user_metrics_cleanup_interval_seconds",
    "prometheus_exclude_labels",
    "prometheus_exclude_metrics",
    "prometheus_metrics_config",
    "prometheus_metrics_max_series_per_metric",
    "prometheus_deployment_and_latency_caller_identity",
    "prometheus_emit_input_sequence_length_label",
    "prometheus_latency_buckets",
    "prometheus_emit_rate_limit_labels",
    "prometheus_user_budget_label_include_email_alias",
    "disable_end_user_cost_tracking",
    "prometheus_metrics_ttl_seconds",
    "prometheus_metrics_cleanup_interval_seconds",
)


import litellm.accounting.metrics  # noqa: E402,F401  # registers its collectors before the snapshot below

_PREEXISTING: Final = frozenset(REGISTRY._collector_to_names)
GOLDEN: Final = Path(__file__).parent / "prometheus_golden" / "python_outputs.json"


def _golden(case: str) -> object:
    return json.loads(GOLDEN.read_text())[case]


def _logger_collectors() -> list[Collector]:
    return [collector for collector in list(REGISTRY._collector_to_names) if collector not in _PREEXISTING]


def _clear_registry() -> None:
    for collector in _logger_collectors():
        REGISTRY.unregister(collector)


def _ours() -> CollectorRegistry:
    registry: Final = CollectorRegistry()
    for collector in _logger_collectors():
        registry.register(collector)
    return registry


@pytest.fixture(autouse=True)
def isolated() -> Generator[None]:
    saved: Final = {name: copy.deepcopy(getattr(litellm, name, None)) for name in _SETTINGS}
    _clear_registry()
    yield
    _clear_registry()
    for name, value in saved.items():
        setattr(litellm, name, value)


def _event(
    index: int,
    *,
    user: str = "u",
    end_user: str | None = None,
    stream: bool = False,
    cache_hit: object = False,
    usage_object: object = None,
    tags: list[str] | None = None,
    headers: Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
    zero_cost: Mapping[str, object] | None = None,
    queue: object = None,
    original_model_group: str | None = None,
    cost: float = 0.0015,
    exception: Exception | None = None,
    aware: bool = False,
    key_hash: str = "hashed",
    params: Mapping[str, object] | None = None,
    request_metadata: Mapping[str, object] | None = None,
    hidden: Mapping[str, object] | None = None,
    guardrails: object = None,
    tokens: Mapping[str, object] | None = None,
    response: object = None,
) -> dict[str, object]:
    base: Final = START.replace(tzinfo=timezone.utc) if aware else START
    start: Final = base + timedelta(seconds=index)
    api_start: Final = start + timedelta(milliseconds=120)
    first_token: Final = api_start + timedelta(milliseconds=333, microseconds=7)
    end: Final = api_start + timedelta(seconds=1, microseconds=123457 * (index + 1))
    payload: Final[dict[str, object]] = {
        "id": f"call-{index}",
        "call_type": "acompletion",
        "status": "success",
        "response_cost": cost,
        "total_tokens": 30 + index,
        "prompt_tokens": 20 + index,
        "completion_tokens": 10,
        "model": "gpt-4o-mini",
        "model_id": f"model-{index % 2}",
        "model_group": "gpt-group",
        "api_base": "https://api.example.com/v1",
        "custom_llm_provider": "openai",
        "request_tags": tags or [],
        "cache_hit": cache_hit,
        "stream": stream,
        "response": None,
        "model_parameters": {},
        "zero_cost_diagnostic": zero_cost,
        "metadata": {
            "user_api_key_hash": key_hash,
            "user_api_key_alias": 'alias "one"\nline',
            "user_api_key_team_id": "team-1",
            "user_api_key_team_alias": "Team One",
            "user_api_key_user_id": user,
            "user_api_key_user_email": f"{user}@example.com",
            "user_api_key_org_id": "org-1",
            "user_api_key_org_alias": "Org",
            "user_api_key_project_id": "p-1",
            "user_api_key_project_alias": None,
            "user_api_key_request_route": "/chat/completions",
            "requester_ip_address": "10.0.0.1",
            "user_agent": "curl/8",
            "user_api_key_end_user_id": end_user,
            "usage_object": usage_object,
            **(metadata or {}),
        },
        "hidden_params": {"litellm_overhead_time_ms": None, "additional_headers": headers, **(hidden or {})},
        "guardrail_information": guardrails,
        **(tokens or {}),
    }
    litellm_metadata: Final[dict[str, object]] = {"user_api_key_end_user_id": end_user}
    if queue is not None:
        litellm_metadata["queue_time_seconds"] = queue
    if original_model_group is not None:
        litellm_metadata["original_model_group"] = original_model_group
    litellm_metadata.update(request_metadata or {})
    kwargs: Final[dict[str, object]] = {
        "model": "gpt-4o-mini",
        "standard_logging_object": payload,
        "litellm_params": {"metadata": litellm_metadata, "user_api_key_end_user_id": end_user, **(params or {})},
        "start_time": start,
        "end_time": end,
        "api_call_start_time": api_start,
        "completion_start_time": first_token,
        "stream": stream,
        "log_event_type": "successful_api_call",
    }
    if exception is not None:
        kwargs["exception"] = exception
    if response is not None:
        kwargs["_response"] = response
    return kwargs


V3_HEADERS: Final = {
    "x-ratelimit-api_key-limit-requests": 100,
    "x-ratelimit-api_key-remaining-requests": 40,
    "x-ratelimit-team-limit-tokens": 1000,
    "x-ratelimit-team-remaining-tokens": 250,
    "x-ratelimit-model_per_project_itpm-limit-tokens": 500,
    "x-ratelimit-model_per_project_itpm-remaining-tokens": 499,
    "x-ratelimit-api_key-limit-tokens": True,
    "x-ratelimit-api_key-remaining-tokens": 1,
}

CORPUS: Final = (
    _event(0),
    _event(1, stream=True, cache_hit=True, tags=["prod", "User-Agent: curl/8.1"], queue=0.25),
    _event(
        2,
        user="v",
        end_user="customer-a",
        cache_hit=None,
        usage_object={
            "prompt_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 7, "audio_tokens": 2, "cache_creation_tokens": 3},
            "completion_tokens_details": {"reasoning_tokens": 5, "audio_tokens": 0},
            "duration_seconds": 4.5,
            "output_image_count": 2,
        },
        headers=V3_HEADERS,
        metadata={
            "requester_metadata": {"team_name": "core", "nested": {"tier": "gold"}},
            "spend_logs_metadata": {"app": "billing"},
            "mcp_tool_call_metadata": {"name": "search", "mcp_server_name": "docs"},
        },
        original_model_group="gpt-requested",
    ),
    _event(
        3,
        end_user="customer-b",
        usage_object={
            "cache_read_input_tokens": 11,
            "cache_creation_input_tokens": None,
            "prompt_tokens_details": {"cache_write_tokens": 4},
        },
        zero_cost={"reason": "missing_pricing_key"},
        cost=0.0,
        metadata={"mcp_tool_call_metadata": {"name": "free", "mcp_server_name": "docs"}},
    ),
    _event(4, end_user="customer-a", headers={"x-ratelimit-api_key-limit-requests": 100}, aware=True, queue=-1),
    _event(5, exception=Exception("Authentication Error: invalid api key")),
    _event(6, key_hash="sk-raw-key", stream=True, tags=["dev"], end_user="customer-c", queue=1),
    _event(7, user="w", cache_hit="unknown", end_user="customer-d"),
    _event(
        8,
        stream=True,
        params={"custom_llm_provider": "openai", "tpm": 9000},
        request_metadata={
            "model_group": "gpt-group",
            "model_info": {"id": "model-0", "rpm": 60, "tpm": 0},
            "litellm-key-remaining-requests-gpt-group": 5,
            "litellm-key-remaining-tokens-gpt-group": "250",
        },
        hidden={
            "litellm_overhead_time_ms": 12.5,
            "additional_headers": {"x_ratelimit_remaining_requests": " 77 ", "x_ratelimit_remaining_tokens": 1000},
        },
        guardrails=[
            {"guardrail_mode": "pre_call", "duration": 0.2},
            {"guardrail_mode": ["pre_call", "during_call"], "duration": 5},
            {"guardrail_mode": ["post_call"], "duration": "0.05"},
        ],
    ),
    _event(
        9,
        params={"custom_llm_provider": "azure"},
        hidden={
            "litellm_overhead_time_ms": 0,
            "additional_headers": {"x-ratelimit-model_per_key-remaining-tokens": 17},
        },
        guardrails={"guardrail_mode": "post_call", "duration": 0.4},
    ),
    _event(10, tokens={"prompt_tokens": 0}, response={"usage": {"prompt_tokens": 1500, "completion_tokens": 3}}),
    _event(11, tokens={"completion_tokens": "ten"}, metadata={"user_api_key_team_id": "team-raise"}),
    _event(12, hidden={"litellm_overhead_time_ms": "slow"}, metadata={"user_api_key_team_id": "team-overhead"}),
    _event(
        13, cost=float("nan"), tokens={"completion_tokens": float("inf")}, metadata={"user_api_key_team_id": 10**20}
    ),
    _event(14, queue="slow", metadata={"user_api_key_team_id": "team-queue"}),
    _event(15, tokens={"completion_tokens": [10]}, metadata={"user_api_key_team_id": "team-list"}),
)


Mode: TypeAlias = Literal["native", "fallback"]
MODES: Final[tuple[Mode, ...]] = ("native", "fallback")


def _line_of(logger: PrometheusLogger, mode: Mode) -> tuple[object, ...]:
    if mode == "native":
        assert isinstance(logger.v2_companion, PrometheusCompanion)
        return (logger, logger.v2_companion)
    return (logger,)


async def _drive(
    mode: Mode,
    monkeypatch: pytest.MonkeyPatch,
    run: Callable[[PrometheusLogger, tuple[object, ...]], Awaitable[str]] | None = None,
) -> str:
    monkeypatch.setattr("time.time", lambda: 1_763_000_000.25)
    monkeypatch.setattr("litellm.integrations.prometheus.datetime", _FixedDatetime)
    monkeypatch.setattr(litellm, "_async_success_callback", [])
    monkeypatch.setattr(litellm, "_async_failure_callback", [])
    logger: Final = PrometheusLogger()
    assert logger._native_store is not None
    lines: Final = _Lines()
    verbose_logger.addHandler(lines)
    try:
        text: Final = await (run or _driven)(logger, _line_of(logger, mode))
    finally:
        verbose_logger.removeHandler(lines)
    return text + "\n".join(lines.seen)


class _Lines(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.seen: Final[list[str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        message: Final = record.getMessage()
        if _NON_BLOCKING.match(message):
            self.seen.append(
                f"{record.levelname} {_NON_BLOCKING.pattern.replace(chr(92), '')} raised {message.strip().splitlines()[-1]}"
            )
        elif _LOGGED.match(message):
            self.seen.append(f"{record.levelname} {message}")


_LOGGED: Final = re.compile(
    r"(Prometheus Error|prometheus Layer Error|\[Non-Blocking\] Prometheus: Budget|Error recording)"
)
_NON_BLOCKING: Final = re.compile(r"LiteLLM\.LoggingError: \[Non-Blocking\] Exception occurred while success logging")
Event: TypeAlias = Literal["success", "failure"]


def _deliver(event: Event, response: object, callback: object, kwargs: object) -> Awaitable[None] | None:
    if not isinstance(callback, PrometheusLogger) or not isinstance(kwargs, dict):
        return None
    return _delivered(event, response, callback, kwargs)


async def _delivered(event: Event, response: object, logger: PrometheusLogger, kwargs: dict[str, object]) -> None:
    try:
        if event == "success":
            await logger.async_log_success_event(kwargs, response, kwargs["start_time"], kwargs["end_time"])
        else:
            await logger.async_log_failure_event(kwargs, None, kwargs["start_time"], kwargs["end_time"])
    except Exception:  # noqa: BLE001  # mirrors Logging._deliver_async_success_to
        verbose_logger.error(
            "LiteLLM.LoggingError: [Non-Blocking] Exception occurred while success logging %s", traceback.format_exc()
        )
        for callback in litellm.logging_callback_manager.get_all_callbacks():
            if isinstance(callback, PrometheusLogger):
                callback.increment_callback_logging_failure(callback_name=type(logger).__name__)
                break


def _owner(kwargs: dict[str, object]) -> Logging:
    owner: Final = Logging(
        model=str(kwargs["model"]),
        messages=[],
        stream=kwargs["stream"] is True,
        call_type="acompletion",
        start_time=kwargs["start_time"],  # pyright: ignore[reportArgumentType]  # every corpus event carries a datetime
        litellm_call_id=str(kwargs.get("litellm_call_id", "parity")),
        function_id="parity",
    )
    owner.model_call_details = kwargs
    return owner


async def _line(line: tuple[object, ...], kwargs: dict[str, object], event: Event) -> None:
    response: Final = kwargs.pop("_response", None)
    litellm._async_success_callback[:] = list(line) if event == "success" else []
    litellm._async_failure_callback[:] = list(line) if event == "failure" else []
    owner: Final = _owner(kwargs)
    args: Final = HookArgs(
        request=kwargs,
        result=response,
        start_time=kwargs["start_time"],
        end_time=kwargs["end_time"],
        outcome=Outcome.success if event == "success" else Outcome.failure,
        logging_obj=owner,
    )
    _ = await run_subscribers(
        hook_subscribers(Hook.attempt_ended, line, admit=Admit.every_entry),
        with_sources(KwargsFields(kwargs, LOGGING_KEYS), DERIVED_ATTEMPT_FIELDS, args),
        lambda callback, current: _deliver(event, response, callback, current),
        call=owner.v2_call_facts(kwargs["end_time"]),
    )


async def _driven(logger: PrometheusLogger, line: tuple[object, ...]) -> str:
    for kwargs in copy.deepcopy(CORPUS):
        await _line(line, kwargs, "success")
    outcomes: Final = await _direct_calls(logger, line)
    return generate_latest(_ours()).decode() + "\n".join(outcomes) + "\n"


async def _direct_calls(logger: PrometheusLogger, line: tuple[object, ...]) -> list[str]:
    failure: Final = _failure(
        10, litellm.RateLimitError(message="slow down", llm_provider="openai", model="gpt-4o-mini")
    )
    rejected: Final = _failure(11, litellm.BadRequestError(message="bad", model="gpt-4o-mini", llm_provider="openai"))
    rejected["standard_logging_object"]["model_id"] = None
    rejected["standard_logging_object"]["model_group"] = None
    rejected["litellm_params"]["metadata"]["model_group"] = "gpt-group"
    rejected["standard_logging_object"]["zero_cost_diagnostic"] = {"reason": "failed_call"}
    calls: Final = (
        lambda: logger.set_deployment_complete_outage("gpt-4o-mini", "model-0", "https://api.example.com/v1", "openai"),
        lambda: logger.increment_deployment_cooled_down(
            "gpt-4o-mini", "model-0", "https://api.example.com/v1", "openai", "429"
        ),
        lambda: logger._record_guardrail_metrics("pii", 0.42, "success", None, "pre_call"),
        lambda: logger._record_guardrail_metrics("pii", 1.5, "error", "Timeout", "pre_call"),
        lambda: logger.increment_callback_logging_failure("s3_v2"),
        lambda: logger.track_provider_remaining_budget("openai", 3.0, 10.0),
        lambda: logger.set_spend_capture_rate("openai", None),
        lambda: logger.set_spend_capture_rate("anthropic", 0.97),
        lambda: logger.set_dynamic_limiter_share("gpt-group", "team-1", "Team One", "high", 1200, 30),
        lambda: logger.increment_dynamic_limiter_throttled("gpt-group", "team-1", None, "high", "tokens", "team"),
        lambda: logger.record_managed_batch_created("gpt-4o-mini", "openai", "u", None, "alias"),
        lambda: logger.record_managed_file_created("gpt-4o-mini", None, "u", "u@example.com", None),
        lambda: logger.record_managed_file_size(2048, "batch", "jsonl", None, "openai", "u"),
        lambda: logger.record_managed_batch_duration(12.5, "gpt-4o-mini"),
        lambda: logger.record_managed_file_deleted("blocked"),
        lambda: logger.record_check_batch_cost_run(3, [("gpt-4o-mini", "openai"), (None, None)]),
        lambda: logger.record_check_batch_cost_error("timeout"),
        lambda: logger.set_deployment_healthy("gpt-4o-mini", "model-1", "https://api.example.com/v1", "openai"),
        lambda: logger._record_guardrail_metrics("pii", "slow", "success", None, "pre_call"),  # pyright: ignore[reportArgumentType]  # callers pass untyped values at runtime
        lambda: logger.record_managed_file_size("big", "batch", "jsonl", None, "openai", "u"),  # pyright: ignore[reportArgumentType]  # callers pass untyped values at runtime
        lambda: logger.record_managed_batch_duration("long", "gpt-4o-mini"),  # pyright: ignore[reportArgumentType]  # callers pass untyped values at runtime
        lambda: logger.record_check_batch_cost_run("many", []),  # pyright: ignore[reportArgumentType]  # callers pass untyped values at runtime
        lambda: logger._record_guardrail_metrics("nan-guard", float("nan"), "success", None, "pre_call"),
        lambda: logger.track_provider_remaining_budget("nan-provider", float("nan"), float("inf")),
        lambda: logger.record_managed_file_created("gpt-4o-mini", None, 10**20, None, None),  # pyright: ignore[reportArgumentType]  # callers pass untyped values at runtime
    )
    outcomes: Final = [_outcome(call) for call in calls]
    broken: Final = _failure(
        13,
        litellm.RateLimitError(message="slow", llm_provider="openai", model="gpt-4o-mini"),
        zero_cost={"note": "no reason"},
        metadata={"user_api_key_team_id": "team-broken"},
    )
    for event in (failure, rejected, broken):
        await _line(line, event, "failure")
    original: Final = litellm.RateLimitError(message="slow", llm_provider="openai", model="gpt-4o-mini")
    fallback_kwargs: Final = {
        "model": "gpt-4o",
        "metadata": {"user_api_key_hash": "hashed", "user_api_key_team_id": "team-1"},
        "tags": ["prod"],
    }
    await logger.log_success_fallback_event("gpt-group", fallback_kwargs, original)
    await logger.log_failure_fallback_event("gpt-group", fallback_kwargs, original)
    await logger.async_post_call_failure_hook(
        {"model": "gpt-group", "metadata": {}},
        litellm.RateLimitError(message="slow down", llm_provider="openai", model="gpt-4o-mini"),
        UserAPIKeyAuth(api_key="hashed", user_id="u", team_id="team-1"),
    )
    await logger.async_post_call_failure_hook(
        {"model": "gpt-group", "metadata": {"requester_ip_address": "10.0.0.2"}},
        litellm.AuthenticationError(message="bad key", llm_provider="openai", model="gpt-4o-mini"),
        UserAPIKeyAuth(api_key="hashed", user_id="u", user_email="u@example.com", key_alias="k", team_id="team-1"),
    )
    return outcomes + _budget_calls(logger) + [await _counts(logger)]


class _FixedClock(type):
    def __instancecheck__(cls, instance: object) -> bool:
        return isinstance(instance, datetime)


class _FixedDatetime(datetime, metaclass=_FixedClock):
    @classmethod
    def now(cls, tz: tzinfo | None = None) -> datetime:
        if tz is None:
            return START
        return datetime(2026, 11, 13, 9, 0, 0, tzinfo=timezone.utc).astimezone(tz)


def _budget_calls(logger: PrometheusLogger) -> list[str]:
    aware_reset: Final = datetime(2026, 11, 20, 0, 0, 0, tzinfo=timezone.utc)
    shifted_reset: Final = datetime(2026, 11, 14, 3, 30, 0, tzinfo=timezone(timedelta(hours=5)))
    naive_reset: Final = START + timedelta(hours=7, seconds=1)
    dst_reset: Final = datetime(2027, 3, 20, 0, 0, 0, tzinfo=ZoneInfo("America/New_York"))
    team: Final = LiteLLM_TeamTable(
        team_id="team-1",
        team_alias="Team One",
        spend=4.5,
        max_budget=10.0,
        budget_reset_at=aware_reset,
        members_with_roles=[Member(user_id="u", role="admin"), Member(user_id="v", role="user")],
    )
    calls: Final = (
        lambda: logger._set_team_budget_metrics(team),
        lambda: logger._set_team_budget_metrics(LiteLLM_TeamTable(team_id="team-2", spend=None)),
        lambda: logger.set_team_members_metric(team),
        lambda: logger._set_org_budget_metrics("org-1", "", 2.0, None, naive_reset),
        lambda: logger._set_org_budget_metrics("org-2", "Org", 1.0, 3.0, shifted_reset),
        lambda: logger._set_key_budget_metrics(
            UserAPIKeyAuth(token="hashed-1", key_alias="k", spend=1.25, max_budget=None, budget_reset_at=naive_reset)
        ),
        lambda: logger._set_key_budget_metrics(
            UserAPIKeyAuth(token="hashed-2", spend=0.5, max_budget=2.0, budget_reset_at=aware_reset)
        ),
        lambda: logger._set_user_budget_metrics(
            LiteLLM_UserTable(
                user_id="u", user_email="u@example.com", spend=3.0, max_budget=5.0, budget_reset_at=aware_reset
            )
        ),
        lambda: logger._set_user_budget_metrics(LiteLLM_UserTable(user_id="v", spend=0.0)),
        lambda: logger._set_customer_budget_metrics("cust-1", 1.0, 4.0, aware_reset),
        lambda: logger._set_customer_budget_metrics("cust-2", 2.0, None, None),
        lambda: logger._set_customer_budget_metrics("cust-3", 0.5, 1.0, naive_reset),
        lambda: logger._set_org_budget_metrics("org-dst", "Dst", 1.0, 2.0, dst_reset),
        lambda: logger._set_key_budget_metrics(
            UserAPIKeyAuth(token="hashed-dst", spend=0.0, max_budget=1.0, budget_reset_at=dst_reset)
        ),
    )
    return [_outcome(call) for call in calls]


class _Counted:
    def __init__(self, repository: object) -> None:
        self.repository = repository

    async def count(self) -> int:
        return 7 if isinstance(self.repository, _Users) else 3


class _Users:
    def __init__(self, _prisma: object) -> None:
        pass

    async def count_billable_users(self) -> int:
        return 5


class _Teams:
    def __init__(self, _prisma: object) -> None:
        pass


async def _counts(logger: PrometheusLogger) -> str:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("litellm.proxy.proxy_server.prisma_client", object())
        patch.setattr("litellm.integrations.prometheus._paginated_table", _Counted)
        patch.setattr("litellm.integrations.prometheus.UserRepository", _Users)
        patch.setattr("litellm.integrations.prometheus.TeamRepository", _Teams)
        await logger._initialize_user_and_team_count_metrics()
    return "counted"


def _failure(
    index: int,
    exception: Exception,
    zero_cost: Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    event: Final = _event(index, zero_cost=zero_cost, metadata=metadata)
    event["log_event_type"] = "failed_api_call"
    event["exception"] = exception
    event["standard_logging_object"]["status"] = "failure"
    return event


def _outcome(call: Callable[[], object]) -> str:
    try:
        call()
    except Exception as error:  # noqa: BLE001  # parity compares whatever each side raises
        return f"{type(error).__name__}: {error}"
    return "ok"


def _masked(text: str) -> str:
    return _CREATED.sub(lambda match: f"{match.group(1)} <created>", text)


CONFIGS: Final[Mapping[str, Mapping[str, object]]] = {
    "defaults": {},
    "custom_labels_tags_stream_end_user": {
        "custom_prometheus_metadata_labels": ["metadata.team_name", "nested.tier", "app", "user_api_key_team_alias"],
        "custom_prometheus_tags": ["prod", "User-Agent: curl/*", "absent"],
        "prometheus_emit_stream_label": True,
        "enable_end_user_cost_tracking_prometheus_only": True,
        "prometheus_end_user_metrics_max_series_per_metric": 2,
    },
    "end_user_ttl": {
        "enable_end_user_cost_tracking_prometheus_only": True,
        "prometheus_end_user_metrics_ttl_seconds": 0.000001,
        "prometheus_end_user_metrics_cleanup_interval_seconds": 0,
    },
    "excludes_include_labels_and_series_cap": {
        "prometheus_exclude_labels": ["user_agent", "client_ip"],
        "prometheus_exclude_metrics": ["litellm_cache_misses_metric"],
        "prometheus_metrics_config": [
            {
                "group": "tokens",
                "metrics": ["litellm_total_tokens_metric", "litellm_spend_metric"],
                "include_labels": ["team", "model_id", "end_user"],
            },
            {
                "group": "rest",
                "metrics": [
                    "litellm_requests_metric",
                    "litellm_proxy_total_requests_metric",
                    "litellm_api_key_rate_limit_allowed_metric",
                    "litellm_api_key_rate_limit_used_metric",
                    "litellm_llm_api_latency_metric",
                    "litellm_mcp_tool_calls_total",
                    "litellm_deployment_state",
                    "litellm_guardrail_requests_total",
                ],
            },
        ],
        "prometheus_metrics_max_series_per_metric": 2,
        "enable_end_user_cost_tracking_prometheus_only": True,
    },
    "caller_identity_both_and_buckets": {
        "prometheus_deployment_and_latency_caller_identity": "both",
        "prometheus_latency_buckets": [0.1, 1, 2.5, 600],
    },
    "input_sequence_length": {"prometheus_emit_input_sequence_length_label": True},
    "rate_limit_labels": {"prometheus_emit_rate_limit_labels": True},
    "user_budget_email_alias": {"prometheus_user_budget_label_include_email_alias": True},
    "caller_identity_user_email": {"prometheus_deployment_and_latency_caller_identity": "user_email"},
    "end_user_cost_tracking_disabled": {
        "enable_end_user_cost_tracking_prometheus_only": True,
        "disable_end_user_cost_tracking": True,
    },
    "series_cap_ttl": {
        "prometheus_metrics_max_series_per_metric": 3,
        "prometheus_metrics_ttl_seconds": 0.000001,
        "prometheus_metrics_cleanup_interval_seconds": 0,
    },
}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("config", list(CONFIGS))
async def test_the_native_callback_and_the_fallback_expose_exactly_what_the_python_logger_does(
    config: str, mode: Mode, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in CONFIGS[config].items():
        setattr(litellm, name, copy.deepcopy(value))
    python_text: Final = str(_golden(f"main/{config}"))
    _clear_registry()
    native_text: Final = await _drive(mode, monkeypatch)

    assert "litellm_spend_metric" in python_text or config.startswith("excludes")
    assert _masked(native_text) == _masked(python_text)
    assert ('input_sequence_length="1k-4k"' in python_text) is (config == "input_sequence_length")
    if config == "defaults":
        raised: Final = "ERROR LiteLLM.LoggingError: [Non-Blocking] Exception occurred while success logging raised"
        assert "ERROR prometheus Layer Error(): Exception occured - 'reason'" in python_text
        assert f"{raised} ValueError: could not convert string to float: 'ten'" in python_text
        assert f"{raised} TypeError: '>=' not supported between instances of 'str' and 'int'" in python_text
        assert "TypeError: float() argument must be a string or a real number, not 'list'" in python_text
        assert re.search(
            r'^litellm_callback_logging_failures_metric_total\{callback_name="PrometheusLogger"\} [1-9]',
            python_text,
            re.MULTILINE,
        )
        assert "ERROR Prometheus Error: set_llm_deployment_success_metrics. Exception occured" in python_text
        assert 'team="team-raise"' in python_text
        assert 'litellm_remaining_team_budget_metric{team="team-raise"' not in python_text
        assert 'org_id="org-dst"} 3044.0' in python_text


def _without(kwargs: dict[str, object], key: str) -> dict[str, object]:
    return {name: value for name, value in kwargs.items() if name != key}


TYPED_EDGES: Final = (
    _event(30, tokens={"response": {"service_tier": "priority"}, "model_parameters": {"service_tier": "flex"}}),
    _event(31, tokens={"model_parameters": {"service_tier": "flex"}}),
    _event(32, stream=True, tokens={"stream": None}),
    {**_event(33, stream=True), "completion_start_time": None},
    _without(_event(34, stream=True), "completion_start_time"),
    _event(
        35,
        params={"litellm_metadata": {"model_group": "routed-group"}},
        request_metadata={
            "model_group": "gpt-group",
            "litellm-key-remaining-requests-gpt-group": 9,
            "litellm-key-remaining-requests-routed-group": 4,
        },
    ),
    _event(
        36,
        metadata={
            "user_api_key_team_id": 123,
            "user_api_key_alias": 1.5,
            "user_api_key_user_id": True,
            "is_silent_experiment": True,
            "user_api_key_auth_metadata": {"team_name": "auth-team"},
        },
    ),
    _event(37, metadata={"user_api_key_team_id": ["a"], "user_api_key_alias": [], "user_api_key_user_id": {"k": 1}}),
    _event(38, metadata={"user_api_key_team_id": 10**40, "user_api_key_team_alias": ("t", 1)}),
    _event(39, tokens={"request_tags": "ab"}),
    _event(40, tokens={"request_tags": ("prod", "x")}),
    _event(41, tokens={"request_tags": {"prod": 1, "z": 2}}),
    _event(42, tokens={"request_tags": {"prod"}}),
    _event(43, tokens={"request_tags": ["prod-1", 7, None, ["y"]]}),
    _event(44, tokens={"request_tags": [7, "prod-1"]}),
    _event(45, tokens={"request_tags": [None]}),
    _event(
        46,
        guardrails=[
            {"guardrail_mode": [{"mode": "pre_call"}, "pre_call"], "duration": 0.3},
            {"guardrail_mode": ("pre_call",), "duration": 0.7},
            {"guardrail_mode": "post_call", "duration": [1]},
        ],
    ),
    _event(47, params={"metadata": ["group"]}),
    {**_event(48), "end_time": None},
    {**_event(49, aware=True), "end_time": None},
    _event(50, tokens={"completion_tokens": (10,)}),
)
EDGE_CONFIGS: Final[Mapping[str, Mapping[str, object]]] = {
    "defaults": {},
    "labels_and_stream": {
        "custom_prometheus_metadata_labels": ["metadata.team_name", "is_silent_experiment"],
        "prometheus_emit_stream_label": True,
    },
    "exact_tags": {"custom_prometheus_tags": ["prod", "a", "7", "x"]},
    "wildcard_tags": {"custom_prometheus_tags": ["absent", "prod*"]},
}


async def _edges(logger: PrometheusLogger, line: tuple[object, ...]) -> str:
    for kwargs in copy.deepcopy(TYPED_EDGES):
        await _line(line, kwargs, "success")
    return generate_latest(_ours()).decode()


@pytest.mark.parametrize("config", list(EDGE_CONFIGS))
async def test_typed_field_edge_cases_expose_what_the_python_logger_does(
    config: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in EDGE_CONFIGS[config].items():
        setattr(litellm, name, copy.deepcopy(value))
    python_text: Final = str(_golden(f"edges/{config}"))
    _clear_registry()
    native_text: Final = await _drive("native", monkeypatch, _edges)

    assert 'model="gpt-group"' in python_text
    assert "ERROR Prometheus Error: set_llm_deployment_success_metrics. Exception occured" in python_text
    assert ("expected string or bytes-like object, got 'NoneType'" in python_text) is (config == "wildcard_tags")
    assert "Exception occured - float() argument must be a string or a real number, not 'list'" in python_text
    assert "raised TypeError: float() argument must be a string or a real number, not 'tuple'" in python_text
    assert "raised AttributeError: 'list' object has no attribute 'get'" in python_text
    assert "raised TypeError: can't subtract offset-naive and offset-aware datetimes" in python_text
    assert _masked(native_text) == _masked(python_text)


async def _shared_call_id_lines(mode: Mode, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    monkeypatch.setattr(litellm, "_async_success_callback", [])
    monkeypatch.setattr(litellm, "_async_failure_callback", [])
    logger: Final = PrometheusLogger()
    line: Final = _line_of(logger, mode)
    shared: Final = [
        {
            **_event(20 + index, user=user, metadata={"user_api_key_team_id": f"team-{user}"}),
            "litellm_call_id": "shared",
        }
        for index, user in enumerate(("x", "y"))
    ]
    _ = await asyncio.gather(*(_line(line, kwargs, "success") for kwargs in shared))
    text: Final = _masked(generate_latest(_ours()).decode())
    _clear_registry()
    return sorted(text.splitlines())


async def test_concurrent_attempts_sharing_a_call_id_keep_their_own_series(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: Final = _golden("shared_call_id")
    assert isinstance(recorded, list)
    python_lines: Final = [str(line) for line in recorded]
    native_lines: Final = await _shared_call_id_lines("native", monkeypatch)

    assert any(
        'team="team-y"' in line and line.startswith("litellm_remaining_team_budget_metric") for line in python_lines
    )
    assert native_lines == python_lines


async def _openmetrics(mode: Mode, monkeypatch: pytest.MonkeyPatch) -> str:
    _ = await _drive(mode, monkeypatch)
    text: Final = _masked(generate_openmetrics(_ours()).decode())
    _clear_registry()
    return text


async def test_openmetrics_output_matches_too(monkeypatch: pytest.MonkeyPatch) -> None:
    python_text: Final = str(_golden("openmetrics"))
    assert await _openmetrics("native", monkeypatch) == python_text


def test_a_counter_handle_keeps_the_prometheus_client_errors() -> None:
    store: Final = _native.NativePrometheusStore()
    counter: Final = store.metric("counter", "c_total", "doc", ["a"], [], [], False, False, True)
    with pytest.raises(ValueError, match="Incorrect label names"):
        counter.labels(b="1")
    with pytest.raises(ValueError, match="non-negative"):
        counter.labels(a="1").inc(-1)
    with pytest.raises(ValueError, match="missing label values"):
        counter.inc()
    with pytest.raises(KeyError):
        counter.remove("absent")
    gauge: Final = store.metric("gauge", "g", "doc", ["a"], [], [], True, False, False)
    with pytest.raises(RuntimeError, match="mostrecent"):
        gauge.labels("x").inc()
    assert counter.collect()[5] == [(["1"], 0.0, 0.0, [], counter.collect()[5][0][4])]


def test_the_router_remaining_op_sets_each_gauge_it_is_given_under_the_request_labels() -> None:
    store: Final = _native.NativePrometheusStore()
    labelnames: Final = ["model_group", "api_provider"]
    tokens: Final = store.metric(
        "gauge", "litellm_remaining_tokens_metric", "doc", labelnames, [], [], False, False, False
    )
    requests: Final = store.metric(
        "gauge", "litellm_remaining_requests_metric", "doc", labelnames, [], [], False, False, False
    )

    store.op(
        "router_remaining",
        json.dumps({"values": {"model_group": "group", "api_provider": "bedrock"}, "tokens": 900, "requests": None}),
    )

    assert [(values, value) for values, value, *_ in tokens.collect()[5]] == [(["group", "bedrock"], 900.0)]
    assert requests.collect()[5] == []


_CALLBACK_LISTS: Final = (
    "callbacks",
    "input_callback",
    "success_callback",
    "failure_callback",
    "_async_input_callback",
    "_async_success_callback",
    "_async_failure_callback",
)
_TIMED: Final = re.compile(r"latency|seconds|duration|_time|_created|overhead")


Registration: TypeAlias = Callable[[], Mapping[str, object]]


def _instance_in(*names: str) -> Registration:
    def register() -> Mapping[str, object]:
        logger: Final = PrometheusLogger()
        for name in names:
            setattr(litellm, name, [logger])
        return {}

    return register


def _string_in(*names: str) -> Registration:
    def register() -> Mapping[str, object]:
        for name in names:
            setattr(litellm, name, ["prometheus"])
        return {}

    return register


def _configured_events() -> Mapping[str, object]:
    add_configured_callback_to_specific_event("prometheus", "success")
    add_configured_callback_to_specific_event("prometheus", "failure")
    return {}


def _proxy_startup() -> Mapping[str, object]:
    _ = _string_in("callbacks")()
    ProxyLogging(user_api_key_cache=DualCache())._init_litellm_callbacks()
    return {}


def _config_reload() -> Mapping[str, object]:
    initialize_callbacks_on_proxy(value=["prometheus"], premium_user=False, config_file_path="", litellm_settings={})
    return {}


def _dynamic(global_too: bool) -> Registration:
    def register() -> Mapping[str, object]:
        _ = _string_in("callbacks")() if global_too else None
        return {"success_callback": ["prometheus"], "failure_callback": ["prometheus"]}

    return register


_REGISTRATIONS: Final[Mapping[str, tuple[Registration, Literal["native", "fallback"]]]] = {
    "instance_in_callbacks": (_instance_in("callbacks"), "native"),
    "string_in_callbacks": (_string_in("callbacks"), "native"),
    "proxy_startup": (_proxy_startup, "native"),
    "config_reload": (_config_reload, "native"),
    "configured_success_and_failure_events": (_configured_events, "native"),
    "string_in_success_and_failure_callback": (_string_in("success_callback", "failure_callback"), "native"),
    "dynamic_and_global": (_dynamic(True), "native"),
    "instance_in_success_and_failure_callback": (_instance_in("success_callback", "failure_callback"), "native"),
    "instance_in_async_lists": (_instance_in("_async_success_callback", "_async_failure_callback"), "native"),
    "dynamic_only": (_dynamic(False), "native"),
}
_PYTHON_EMITS_NOTHING: Final = frozenset(
    {
        ("instance_in_success_and_failure_callback", "stream"),
        ("instance_in_success_and_failure_callback", "failure"),
    }
)
_SCENARIOS: Final[Mapping[str, Mapping[str, object]]] = {
    "success": {"mock_response": "hi there"},
    "stream": {"mock_response": "hi there", "stream": True},
    "failure": {"mock_response": "litellm.RateLimitError"},
    "callback_error": {"mock_response": "hi there", "queue_time_seconds": "slow"},
    "no_log": {"mock_response": "hi there", "no-log": True},
}


def _kept(line: str) -> bool:
    if line.startswith("#"):
        return False
    name: Final = line.split("{")[0].split(" ")[0]
    return _TIMED.search(name) is None or (name.endswith("_count") and not name.endswith("_created"))


def _fresh_loggers(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _CALLBACK_LISTS:
        monkeypatch.setattr(litellm, name, [])
    _in_memory_loggers[:] = [logger for logger in _in_memory_loggers if not isinstance(logger, PrometheusLogger)]


async def _completion_text(registration: str, scenario: str, monkeypatch: pytest.MonkeyPatch) -> str:
    _fresh_loggers(monkeypatch)
    dynamic: Final = _REGISTRATIONS[registration][0]()
    options: Final = dict(_SCENARIOS[scenario])
    queue: Final = options.pop("queue_time_seconds", None)
    metadata: Final[dict[str, object]] = {
        "user_api_key_hash": "hashed",
        "user_api_key_team_id": "team-1",
        "tags": ["prod"],
    }
    if queue is not None:
        metadata["queue_time_seconds"] = queue
    try:
        response: Final = await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
            metadata=metadata,
            **options,
            **dynamic,
        )
        if isinstance(response, CustomStreamWrapper):
            assert [chunk.id async for chunk in response]
    except litellm.RateLimitError:
        pass
    await asyncio.sleep(1)
    text: Final = generate_latest(_ours()).decode()
    return "\n".join(line for line in text.splitlines() if _kept(line))


@pytest.fixture
def native_recordings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    recorded: Final[list[str]] = []
    original: Final = PrometheusCompanion._recorded

    async def counted(companion: PrometheusCompanion, ticket: NativeTicket, ctx: CallContext[object]) -> Stopped | None:
        recorded.append(ticket.event)
        return await original(companion, ticket, ctx)

    monkeypatch.setattr(PrometheusCompanion, "_recorded", counted)
    return recorded


@pytest.fixture
def saved_in_memory_loggers() -> Generator[None]:
    saved: Final = tuple(_in_memory_loggers)
    yield
    _in_memory_loggers[:] = saved


@pytest.mark.parametrize("scenario", list(_SCENARIOS))
@pytest.mark.parametrize("registration", list(_REGISTRATIONS))
@pytest.mark.usefixtures("saved_in_memory_loggers")
async def test_every_registration_path_emits_each_request_exactly_once_as_python_does(
    registration: str, scenario: str, monkeypatch: pytest.MonkeyPatch, native_recordings: list[str]
) -> None:
    _ = await _completion_text(registration, scenario, monkeypatch)
    _clear_registry()
    native_recordings.clear()
    python_text: Final = str(_golden(f"registration/{registration}/{scenario}"))
    native_text: Final = await _completion_text(registration, scenario, monkeypatch)

    assert any(isinstance(collector, NativeFamilyCollector) for collector in _logger_collectors())
    assert native_text == python_text
    if scenario == "no_log":
        assert native_recordings == []
        assert "litellm_total_tokens_metric_total{" not in native_text
        return
    expected_event: Final = "failure" if scenario == "failure" else "success"
    if (registration, scenario) in _PYTHON_EMITS_NOTHING:
        assert native_recordings == []
        return
    assert native_recordings == ([expected_event] if _REGISTRATIONS[registration][1] == "native" else [])
    if scenario == "failure":
        assert "litellm_llm_api_failed_requests_metric_total{" in native_text
    else:
        assert re.search(r"^litellm_total_tokens_metric_total\{.*\} \d+\.0$", native_text, re.MULTILINE)
        assert "litellm_llm_api_latency_metric_count{" in native_text
    if scenario == "stream":
        assert "litellm_llm_api_time_to_first_token_metric_count{" in native_text
    failures: Final = re.search(
        r'^litellm_callback_logging_failures_metric_total\{callback_name="PrometheusLogger"\} (\S+)$',
        native_text,
        re.MULTILINE,
    )
    assert (failures is not None and failures.group(1) == "1.0") is (
        scenario == "callback_error" and registration != "dynamic_only"
    )


def _sync_completion_text(registration: str, monkeypatch: pytest.MonkeyPatch) -> str:
    _fresh_loggers(monkeypatch)
    _ = _REGISTRATIONS[registration][0]()
    _ = litellm.completion(model="gpt-4o-mini", messages=[{"role": "user", "content": "hello"}], mock_response="hi")
    return generate_latest(_ours()).decode()


@pytest.mark.parametrize("registration", ["instance_in_callbacks", "instance_in_success_and_failure_callback"])
@pytest.mark.usefixtures("saved_in_memory_loggers")
def test_a_sync_completion_emits_nothing_on_either_side(
    registration: str, monkeypatch: pytest.MonkeyPatch, native_recordings: list[str]
) -> None:
    python_text: Final = str(_golden(f"sync/{registration}"))
    native_text: Final = _sync_completion_text(registration, monkeypatch)

    assert native_recordings == []
    assert "litellm_total_tokens_metric_total{" not in python_text
    assert native_text == python_text


def _cooled_down_text(monkeypatch: pytest.MonkeyPatch) -> str:
    _fresh_loggers(monkeypatch)
    _ = _proxy_startup()
    logger: Final = get_prometheus_logger_from_callbacks()
    assert isinstance(logger, PrometheusLogger)
    assert logger.v2_companion is not None
    logger.increment_deployment_cooled_down("gpt-4o-mini", "id-1", "https://api.openai.com", "openai", "429")
    text: Final = generate_latest(_ours()).decode()
    return "\n".join(line for line in text.splitlines() if _kept(line))


@pytest.mark.usefixtures("saved_in_memory_loggers")
def test_the_router_cooldown_lookup_reaches_the_native_logger(monkeypatch: pytest.MonkeyPatch) -> None:
    python_text: Final = str(_golden("cooled_down"))
    native_text: Final = _cooled_down_text(monkeypatch)

    assert re.search(r"^litellm_deployment_cooled_down_total\{.*\} 1\.0$", native_text, re.MULTILINE)
    assert native_text == python_text


_PID: Final = re.compile(r'pid="\d+"')
MULTIPROCESS_CONFIGS: Final = (
    "defaults",
    "custom_labels_tags_stream_end_user",
    "excludes_include_labels_and_series_cap",
)


async def _merged(logger: PrometheusLogger, line: tuple[object, ...]) -> str:
    for kwargs in copy.deepcopy(CORPUS):
        await _line(line, kwargs, "success")
    outcomes: Final = await _direct_calls(logger, line)
    registry: Final = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return generate_latest(registry).decode() + "\n".join(outcomes) + "\n"


def multiprocess_main(mode: Mode, config: str, output: str) -> None:
    for name, value in CONFIGS[config].items():
        setattr(litellm, name, copy.deepcopy(value))
    with pytest.MonkeyPatch.context() as monkeypatch:
        text: Final = asyncio.run(_drive(mode, monkeypatch, _merged))
    _ = Path(output).write_text(text)


def _multiprocess_text(mode: Mode, config: str, tmp_path: Path) -> str:
    directory: Final = tmp_path / mode
    directory.mkdir()
    output: Final = tmp_path / f"{mode}.txt"
    environment: Final = dict(os.environ)
    environment["PROMETHEUS_MULTIPROC_DIR"] = str(directory)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(Path(litellm.__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[3]))
    )
    script: Final = (
        "from tests.unit.integrations.test_prometheus_native_parity import multiprocess_main; "
        f"multiprocess_main({mode!r}, {config!r}, {str(output)!r})"
    )
    finished: Final = subprocess.run(
        [sys.executable, "-P", "-c", script], env=environment, capture_output=True, text=True, timeout=600
    )
    assert finished.returncode == 0, finished.stderr[-4000:]
    return "\n".join(sorted(_PID.sub('pid="<pid>"', _masked(output.read_text())).splitlines()))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("config", MULTIPROCESS_CONFIGS)
def test_multiprocess_mode_writes_what_the_python_logger_writes_to_the_shared_directory(
    config: str, mode: Mode, tmp_path: Path
) -> None:
    python_text: Final = str(_golden(f"multiprocess/{config}"))
    native_text: Final = _multiprocess_text(mode, config, tmp_path)

    assert "litellm_spend_metric" in python_text or config.startswith("excludes")
    assert native_text == python_text


def _raised_line(error: BaseException) -> str:
    return traceback.format_exception_only(error)[-1].strip()


@pytest.mark.parametrize(
    "original",
    [
        AttributeError("'NoneType' object has no attribute 'get'"),
        KeyError("metadata"),
        LookupError("no usage"),
        TypeError("can't subtract offset-naive and offset-aware datetimes"),
    ],
    ids=["attribute", "key", "lookup", "type"],
)
def test_a_named_stop_is_logged_as_the_error_python_raised(original: Exception) -> None:
    ticket: Final = NativeTicket(
        companion=None,  # pyright: ignore[reportArgumentType]  # the stop decoding never reads the companion
        call_id="c",
        event="success",
        wire={},
        after=_never,
        budget=None,
    )
    stop: Final = (
        f'{{"step": "success_latency", "kind": "named", "message": {json.dumps(str(original))}, '
        f'"error_type": {json.dumps(type(original).__name__)}}}'
    )
    stopped: Final = ticket.stopped(RuntimeError(stop))
    assert stopped.step == "success_latency"
    assert isinstance(stopped.error, type(original))
    assert _raised_line(stopped.error) == _raised_line(original)


async def _never() -> None:
    raise AssertionError("not awaited")
