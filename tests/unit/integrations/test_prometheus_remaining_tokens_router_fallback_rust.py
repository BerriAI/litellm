"""
LIT-2719 — `litellm_remaining_tokens_metric` and
`litellm_remaining_requests_metric` only fired for providers that return
`x-ratelimit-remaining-*` response headers (OpenAI, Azure, Anthropic).

This guarded the gauges behind a provider-specific code path, so Bedrock and
Vertex deployments — which never populate those headers — silently produced no
data even when the proxy router had `tpm`/`rpm` configured.

The Rust Prometheus callback reads the proxy router through its `router_ref`
field after the tail step and emits the gauges with
`configured_limit - current_usage`, the remaining usage the router computes.

Tests cover:
- Bedrock fallback emits both gauges.
- Vertex AI fallback emits both gauges.
- Already-present headers short-circuit the router lookup entirely.
- Partial header coverage (only requests) still triggers the missing tokens
  gauge.
- llm_router unavailable / model_group missing / router raises → silent no-op.
"""

import time
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Final, cast
from unittest.mock import patch

import pytest
from prometheus_client import REGISTRY
from prometheus_client.samples import Sample

import litellm
from litellm.integrations.prometheus import PrometheusLogger
from litellm.redis.stores.router_usage import CounterKind, UsageCounter
from litellm.utils import get_utc_datetime
from tests.unit.integrations.prometheus_native_support import enable_native_prometheus, native_success, standard_payload

if TYPE_CHECKING:
    from litellm.rust_bridge._native import RouterRemainingReads

BEDROCK_MODEL: Final = "bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
VERTEX_MODEL: Final = "vertex_ai/gemini-1.5-pro"


@pytest.fixture(autouse=True)
def native_prometheus(monkeypatch: pytest.MonkeyPatch) -> None:
    enable_native_prometheus(monkeypatch)


@pytest.fixture(autouse=True)
def minute_pinned() -> None:
    seconds_left: Final = 60 - datetime.now(timezone.utc).second
    if seconds_left < 10:
        time.sleep(seconds_left + 0.1)


@pytest.fixture(scope="function")
def prometheus_logger():
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)
    return PrometheusLogger()


def _build_payload(
    model_group: str = "bedrock-claude-group",
    custom_llm_provider: str = "bedrock",
    additional_headers: Mapping[object, object] | None = None,
) -> dict[str, object]:
    return {
        "model_group": model_group,
        "custom_llm_provider": custom_llm_provider,
        "model": "anthropic.claude-3-sonnet-20240229-v1:0",
        "model_id": "deployment-id-1",
        "api_base": "https://bedrock-runtime.us-east-1.amazonaws.com",
        "hidden_params": {
            "additional_headers": additional_headers or {},
        },
        "metadata": {
            "user_api_key_hash": "test-key",
            "user_api_key_alias": None,
            "user_api_key_team_id": None,
            "user_api_key_team_alias": None,
        },
    }


def _router(
    model_group: str = "bedrock-claude-group",
    model: str = BEDROCK_MODEL,
    limits: dict[str, int] | None = None,
    used: dict[CounterKind, int] | None = None,
) -> litellm.Router:
    params: Final[dict[str, object]] = {"model": model, **(limits or {})}
    if model.startswith("vertex_ai/"):
        params.update({"vertex_project": "p", "vertex_location": "us-central1"})
    router: Final = litellm.Router(
        model_list=[{"model_name": model_group, "litellm_params": params, "model_info": {"id": "deployment-id-1"}}]
    )
    minute: Final = get_utc_datetime().strftime("%H-%M")
    usage: Final = router._core._usage  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]  # the store the router counts usage in
    for kind, value in (used or {}).items():
        usage.set_local(UsageCounter("global_router", "deployment-id-1", model, minute, kind), value)
    return router


class _ReadSpy:
    def __init__(self, router: litellm.Router, error: Exception | None = None) -> None:
        self.hosted: Final = router._core._hosted  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]  # the registry health().usage reads
        self.read: Final = self.hosted._core_remaining_usage_reads
        self.error: Final = error
        self.calls: list[str] = []

    async def __call__(self, model_group: str) -> "RouterRemainingReads":
        self.calls.append(model_group)
        if self.error is not None:
            raise self.error
        return await self.read(model_group)


def _spied(monkeypatch: pytest.MonkeyPatch, router: litellm.Router, error: Exception | None = None) -> _ReadSpy:
    spy: Final = _ReadSpy(router, error)
    monkeypatch.setattr(spy.hosted, "_core_remaining_usage_reads", spy)
    return spy


def _registry_samples() -> Iterator[Sample]:
    for metric in REGISTRY.collect():
        yield from metric.samples


def _sample_values(sample_name: str) -> tuple[float, ...]:
    return tuple(sample.value for sample in _registry_samples() if sample.name == sample_name)


def _request_kwargs(payload: dict[str, object]) -> dict[str, object]:
    return {
        "litellm_params": {"custom_llm_provider": payload["custom_llm_provider"], "metadata": {}},
        "model": payload["model"],
        "standard_logging_object": payload,
    }


class TestRouterFallbackEmitsForBedrock:
    @pytest.mark.asyncio
    async def test_should_emit_both_gauges_for_bedrock_when_router_has_limits(
        self, prometheus_logger: PrometheusLogger, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = _build_payload(custom_llm_provider="bedrock")
        router = _router(limits={"tpm": 100, "rpm": 10}, used={"tpm": 25, "rpm": 1})
        spy = _spied(monkeypatch, router)

        with patch("litellm.proxy.proxy_server.llm_router", router, create=True):
            await _log_success(prometheus_logger, _request_kwargs(payload))

        assert spy.calls == ["bedrock-claude-group"]
        assert _sample_values("litellm_remaining_tokens_metric") == (75.0,)
        assert _sample_values("litellm_remaining_requests_metric") == (9.0,)


class TestRouterFallbackEmitsForVertex:
    @pytest.mark.asyncio
    async def test_should_emit_both_gauges_for_vertex_when_router_has_limits(
        self, prometheus_logger: PrometheusLogger, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = _build_payload(
            model_group="vertex-gemini-group",
            custom_llm_provider="vertex_ai",
        )
        router = _router("vertex-gemini-group", VERTEX_MODEL, limits={"tpm": 12345, "rpm": 50})
        spy = _spied(monkeypatch, router)

        with patch("litellm.proxy.proxy_server.llm_router", router, create=True):
            await _log_success(prometheus_logger, _request_kwargs(payload))

        assert spy.calls == ["vertex-gemini-group"]
        assert _sample_values("litellm_remaining_tokens_metric") == (12345.0,)
        assert _sample_values("litellm_remaining_requests_metric") == (50.0,)


class TestExistingHeadersShortCircuit:
    @pytest.mark.asyncio
    async def test_should_skip_router_lookup_when_both_headers_already_present(
        self, prometheus_logger: PrometheusLogger, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = _build_payload(
            additional_headers={
                "x_ratelimit_remaining_tokens": 999,
                "x_ratelimit_remaining_requests": 99,
            }
        )
        router = _router(limits={"tpm": 100, "rpm": 10}, used={"tpm": 25, "rpm": 1})
        spy = _spied(monkeypatch, router)

        with patch("litellm.proxy.proxy_server.llm_router", router, create=True):
            await _log_success(prometheus_logger, _request_kwargs(payload))

        assert spy.calls == []
        assert _sample_values("litellm_remaining_tokens_metric") == (999.0,)
        assert _sample_values("litellm_remaining_requests_metric") == (99.0,)

    @pytest.mark.asyncio
    async def test_should_only_fill_missing_dimension_when_one_header_present(
        self, prometheus_logger: PrometheusLogger, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = _build_payload(
            additional_headers={
                "x_ratelimit_remaining_requests": 7,
            }
        )
        router = _router(limits={"tpm": 555, "rpm": 999})
        _spied(monkeypatch, router)

        with patch("litellm.proxy.proxy_server.llm_router", router, create=True):
            await _log_success(prometheus_logger, _request_kwargs(payload))

        assert _sample_values("litellm_remaining_tokens_metric") == (555.0,)
        assert _sample_values("litellm_remaining_requests_metric") == (7.0,)


class TestRouterFallbackDefensivePaths:
    @pytest.mark.asyncio
    async def test_should_noop_when_llm_router_is_none(self, prometheus_logger: PrometheusLogger) -> None:
        payload = _build_payload()

        with patch("litellm.proxy.proxy_server.llm_router", None, create=True):
            await _log_success(prometheus_logger, _request_kwargs(payload))

        assert _sample_values("litellm_remaining_tokens_metric") == ()
        assert _sample_values("litellm_remaining_requests_metric") == ()

    @pytest.mark.asyncio
    async def test_should_noop_when_model_group_missing(
        self, prometheus_logger: PrometheusLogger, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = _build_payload()
        payload["model_group"] = None
        router = _router(limits={"tpm": 100, "rpm": 10})
        spy = _spied(monkeypatch, router)

        with patch("litellm.proxy.proxy_server.llm_router", router, create=True):
            await _log_success(prometheus_logger, _request_kwargs(payload))

        assert spy.calls == []
        assert _sample_values("litellm_remaining_tokens_metric") == ()

    @pytest.mark.asyncio
    async def test_should_noop_when_router_returns_empty_dict(
        self, prometheus_logger: PrometheusLogger, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = _build_payload()
        router = _router()
        spy = _spied(monkeypatch, router)

        with patch("litellm.proxy.proxy_server.llm_router", router, create=True):
            await _log_success(prometheus_logger, _request_kwargs(payload))

        assert spy.calls == ["bedrock-claude-group"]
        assert await router.health().usage("bedrock-claude-group", kind="remaining") == {}
        assert _sample_values("litellm_remaining_tokens_metric") == ()
        assert _sample_values("litellm_remaining_requests_metric") == ()

    @pytest.mark.asyncio
    async def test_should_swallow_router_exception(
        self, prometheus_logger: PrometheusLogger, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = _build_payload()
        router = _router(limits={"tpm": 100, "rpm": 10})
        spy = _spied(monkeypatch, router, RuntimeError("router boom"))
        failures: Final[list[str]] = []
        monkeypatch.setattr(
            prometheus_logger,
            "increment_callback_logging_failure",
            lambda callback_name: failures.append(callback_name),
        )

        with patch("litellm.proxy.proxy_server.llm_router", router, create=True):
            # Must not raise.
            await _log_success(prometheus_logger, _request_kwargs(payload))

        assert spy.calls == ["bedrock-claude-group"]
        assert failures == []
        assert _sample_values("litellm_remaining_tokens_metric") == ()


async def _log_success(logger: PrometheusLogger, request_kwargs: dict[str, object]) -> None:
    payload: Final = cast("dict[str, object]", request_kwargs["standard_logging_object"])
    full = standard_payload(
        metadata=cast("dict[str, object]", payload["metadata"]),
        **{key: value for key, value in payload.items() if key != "metadata"},
    )
    await native_success(
        logger,
        {**request_kwargs, "standard_logging_object": full},
        None,
        datetime(2026, 1, 1),
        datetime(2026, 1, 1, 0, 0, 1),
    )


@pytest.mark.parametrize("remaining_tokens", ["856, abc", 10**400], ids=["garbled", "too-large-for-a-float"])
@pytest.mark.asyncio
async def test_unparseable_provider_ratelimit_header_skips_only_its_own_gauge(
    prometheus_logger: PrometheusLogger, remaining_tokens: str | int
):
    payload: Final = _build_payload(
        model_group="groq-group",
        custom_llm_provider="groq",
        additional_headers={
            "x_ratelimit_remaining_requests": 0,
            "x_ratelimit_remaining_tokens": remaining_tokens,
        },
    )
    request_kwargs: Final[dict[str, object]] = {
        "litellm_params": {"custom_llm_provider": "groq", "metadata": {}},
        "model": "llama-3.1-8b-instant",
        "standard_logging_object": payload,
    }

    await _log_success(prometheus_logger, request_kwargs)

    assert _sample_values("litellm_remaining_requests_metric") == (0.0,)
    assert _sample_values("litellm_remaining_tokens_metric") == ()
    assert _sample_values("litellm_deployment_success_responses_total") == (1.0,)


@pytest.mark.parametrize(
    ("additional_headers", "remaining_requests", "remaining_tokens"),
    [
        ({"x_ratelimit_remaining_requests": "12.5", "x_ratelimit_remaining_tokens": "856.5"}, 12.5, 856.5),
        (
            {1: "non-string key", "x_ratelimit_remaining_requests": "12.5", "x_ratelimit_remaining_tokens": "856.5"},
            12.5,
            856.5,
        ),
        (
            {"x_ratelimit_remaining_requests": "inf", "x_ratelimit_remaining_tokens": "\u0661\u0662.\u0665"},
            float("inf"),
            12.5,
        ),
    ],
    ids=["decimal", "decimal-with-a-non-string-key", "infinity-and-arabic-indic-digits"],
)
@pytest.mark.asyncio
async def test_provider_ratelimit_headers_set_both_gauges_to_the_value_float_reads(
    prometheus_logger: PrometheusLogger,
    additional_headers: dict[object, object],
    remaining_requests: float,
    remaining_tokens: float,
):
    payload: Final = _build_payload(
        model_group="groq-group",
        custom_llm_provider="groq",
        additional_headers=additional_headers,
    )

    await _log_success(
        prometheus_logger,
        {
            "litellm_params": {"custom_llm_provider": "groq", "metadata": {}},
            "model": "llama-3.1-8b-instant",
            "standard_logging_object": payload,
        },
    )

    assert _sample_values("litellm_remaining_requests_metric") == (remaining_requests,)
    assert _sample_values("litellm_remaining_tokens_metric") == (remaining_tokens,)
    assert _sample_values("litellm_deployment_success_responses_total") == (1.0,)
