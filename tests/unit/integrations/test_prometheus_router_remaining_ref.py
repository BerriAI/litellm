import time
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Final, cast
from unittest.mock import MagicMock, patch

import pytest
from prometheus_client import REGISTRY
from prometheus_client.samples import Sample

import litellm
from litellm.integrations import prometheus
from litellm.integrations.prometheus import PrometheusLogger
from litellm.redis.stores.router_usage import CounterKind, UsageCounter
from litellm.router_rust.core import remaining_usage_ref
from litellm.types.integrations.prometheus import UserAPIKeyLabelValues
from litellm.utils import get_utc_datetime
from tests.unit.integrations.prometheus_native_support import enable_native_prometheus, native_success, standard_payload

if TYPE_CHECKING:
    from litellm.rust_bridge._native import RouterRemainingReads

BEDROCK_MODEL: Final = "bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
VERTEX_MODEL: Final = "vertex_ai/gemini-1.5-pro"
READ_FAILED: Final = "Prometheus: get_remaining_model_group_usage failed for model_group=%s: %s"


@pytest.fixture(autouse=True)
def native_prometheus(monkeypatch: pytest.MonkeyPatch) -> None:
    enable_native_prometheus(monkeypatch)


@pytest.fixture(autouse=True)
def minute_pinned() -> None:
    seconds_left: Final = 60 - datetime.now(timezone.utc).second
    if seconds_left < 10:
        time.sleep(seconds_left + 0.1)


@pytest.fixture
def prometheus_logger() -> PrometheusLogger:
    for collector in list(REGISTRY._collector_to_names.keys()):  # pyright: ignore[reportPrivateUsage]
        REGISTRY.unregister(collector)
    return PrometheusLogger()


def _payload(
    model_group: object = "bedrock-claude-group",
    custom_llm_provider: str = "bedrock",
    additional_headers: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "model_group": model_group,
        "custom_llm_provider": custom_llm_provider,
        "model": "anthropic.claude-3-sonnet-20240229-v1:0",
        "model_id": "deployment-id-1",
        "api_base": "https://bedrock-runtime.us-east-1.amazonaws.com",
        "hidden_params": {"additional_headers": additional_headers or {}},
        "metadata": {
            "user_api_key_hash": "test-key",
            "user_api_key_alias": None,
            "user_api_key_team_id": None,
            "user_api_key_team_alias": None,
        },
    }


def _enum_values(model_group: str = "bedrock-claude-group") -> UserAPIKeyLabelValues:
    return UserAPIKeyLabelValues(
        end_user=None,
        hashed_api_key="test-key",
        api_key_alias=None,
        team=None,
        team_alias=None,
        requested_model=model_group,
        model_group=model_group,
        model_id="deployment-id-1",
        api_base="https://bedrock-runtime.us-east-1.amazonaws.com",
        api_provider="bedrock",
        litellm_model_name="anthropic.claude-3-sonnet-20240229-v1:0",
    )


def _attribute(value: object, name: str) -> object:
    return getattr(value, name)  # pyright: ignore[reportAny]


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
    set_local: Final = _attribute(_attribute(_attribute(router, "_core"), "_usage"), "set_local")
    assert callable(set_local)
    for kind, value in (used or {}).items():
        _ = set_local(UsageCounter("global_router", "deployment-id-1", model, minute, kind), value)
    return router


class _Spy:
    def __init__(self, read: object, error: Exception | None) -> None:
        self.read: Final = read
        self.error: Final = error
        self.calls: list[object] = []

    async def __call__(self, model_group: object) -> object:
        self.calls.append(model_group)
        if self.error is not None:
            raise self.error
        return await cast("Callable[[object], Awaitable[object]]", self.read)(model_group)


def _spied(monkeypatch: pytest.MonkeyPatch, router: litellm.Router, name: str, error: Exception | None = None) -> _Spy:
    hosted: Final = _attribute(_attribute(router, "_core"), "_hosted")
    spy: Final = _Spy(_attribute(hosted, name), error)
    monkeypatch.setattr(hosted, name, spy)
    return spy


def _sample_values(sample_name: str) -> tuple[float, ...]:
    samples: Final[Iterator[Sample]] = (sample for metric in REGISTRY.collect() for sample in metric.samples)
    return tuple(sample.value for sample in samples if sample.name == sample_name)


async def _set_router_remaining(logger: PrometheusLogger, payload: dict[str, object], group: str) -> None:
    await logger._async_set_router_remaining_metrics(  # pyright: ignore[reportPrivateUsage]
        standard_logging_payload=cast("StandardLoggingPayload", payload),
        enum_values=_enum_values(group),
    )


@pytest.mark.parametrize(
    ("group", "model", "limits", "used", "expected"),
    [
        ("bedrock-claude-group", BEDROCK_MODEL, {"tpm": 100, "rpm": 10}, {"tpm": 25, "rpm": 1}, ((75.0,), (9.0,))),
        ("vertex-gemini-group", VERTEX_MODEL, {"tpm": 12345, "rpm": 50}, {}, ((12345.0,), (50.0,))),
    ],
    ids=["bedrock", "vertex"],
)
@pytest.mark.asyncio
async def test_a_rust_router_is_read_through_its_router_reference(
    prometheus_logger: PrometheusLogger,
    monkeypatch: pytest.MonkeyPatch,
    group: str,
    model: str,
    limits: dict[str, int],
    used: dict[CounterKind, int],
    expected: tuple[tuple[float, ...], tuple[float, ...]],
) -> None:
    router: Final = _router(group, model, limits, used)
    reads: Final = _spied(monkeypatch, router, "_core_remaining_usage_reads")
    python_read: Final = _spied(monkeypatch, router, "_core_get_remaining_model_group_usage")

    with patch("litellm.proxy.proxy_server.llm_router", router, create=True):
        await _set_router_remaining(prometheus_logger, _payload(group), group)

    assert reads.calls == [group]
    assert python_read.calls == []
    assert (
        _sample_values("litellm_remaining_tokens_metric"),
        _sample_values("litellm_remaining_requests_metric"),
    ) == expected


@pytest.mark.asyncio
async def test_the_router_reference_answers_what_health_usage_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    router: Final = _router(limits={"tpm": 100, "rpm": 10}, used={"tpm": 25, "rpm": 1})
    reference: Final = remaining_usage_ref(router)

    assert reference is not None
    assert await reference.remaining("bedrock-claude-group") == await router.health().usage(
        "bedrock-claude-group", kind="remaining"
    )
    assert await reference.remaining("bedrock-claude-group") == {
        "x-ratelimit-remaining-tokens": 75,
        "x-ratelimit-limit-tokens": 100,
        "x-ratelimit-remaining-requests": 9,
        "x-ratelimit-limit-requests": 10,
    }


@pytest.mark.parametrize("model_group", [5, ("a",)], ids=["int", "tuple"])
@pytest.mark.asyncio
async def test_a_model_group_that_is_not_a_str_reads_the_router_as_python_does(
    prometheus_logger: PrometheusLogger, monkeypatch: pytest.MonkeyPatch, model_group: object
) -> None:
    router: Final = _router(limits={"tpm": 100, "rpm": 10})
    reads: Final = _spied(monkeypatch, router, "_core_remaining_usage_reads")
    python_read: Final = _spied(monkeypatch, router, "_core_get_remaining_model_group_usage")

    with patch("litellm.proxy.proxy_server.llm_router", router, create=True):
        await _set_router_remaining(prometheus_logger, _payload(model_group), "bedrock-claude-group")

    assert python_read.calls == [model_group]
    assert reads.calls == [model_group]


@pytest.mark.asyncio
async def test_a_failed_router_read_logs_the_raised_error_with_its_traceback(
    prometheus_logger: PrometheusLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    router: Final = _router(limits={"tpm": 100, "rpm": 10})
    error: Final = RuntimeError("router boom")
    reads: Final = _spied(monkeypatch, router, "_core_remaining_usage_reads", error)
    logged: Final = MagicMock()
    monkeypatch.setattr(prometheus.verbose_logger, "exception", logged)

    with patch("litellm.proxy.proxy_server.llm_router", router, create=True):
        await _set_router_remaining(prometheus_logger, _payload(), "bedrock-claude-group")

    assert reads.calls == ["bedrock-claude-group"]
    logged.assert_called_once_with(READ_FAILED, "bedrock-claude-group", error)
    assert _sample_values("litellm_remaining_tokens_metric") == ()


@pytest.mark.asyncio
async def test_a_limit_python_cannot_subtract_raises_its_type_error_through_the_reference() -> None:
    router: Final = _router(limits={"tpm": 100})
    hosted: Final = _attribute(_attribute(router, "_core"), "_hosted")

    async def reads(_model_group: str) -> "RouterRemainingReads":
        return (("100", None, None, None), None, (None, None), 0)  # pyright: ignore[reportReturnType]

    with patch.object(hosted, "_core_remaining_usage_reads", reads):
        reference: Final = remaining_usage_ref(router)
        assert reference is not None
        with pytest.raises(TypeError, match=r"unsupported operand type\(s\) for -: 'str' and 'int'"):
            _ = await reference.remaining("bedrock-claude-group")


async def _log_success(logger: PrometheusLogger, payload: dict[str, object]) -> None:
    metadata: Final = cast("dict[str, object]", payload["metadata"])
    full: Final = standard_payload(
        metadata=metadata, **{key: value for key, value in payload.items() if key != "metadata"}
    )
    await native_success(
        logger,
        {
            "litellm_params": {"custom_llm_provider": payload["custom_llm_provider"], "metadata": {}},
            "model": payload["model"],
            "standard_logging_object": full,
        },
        None,
        datetime(2026, 1, 1),
        datetime(2026, 1, 1, 0, 0, 1),
    )


@pytest.mark.parametrize("budget_raises", [False, True], ids=["budget-ran", "budget-raised"])
@pytest.mark.asyncio
async def test_the_router_is_read_after_the_budget_step_and_not_when_it_raises(
    prometheus_logger: PrometheusLogger, monkeypatch: pytest.MonkeyPatch, budget_raises: bool
) -> None:
    router: Final = _router(limits={"tpm": 100, "rpm": 10}, used={"tpm": 25, "rpm": 1})
    reads: Final = _spied(monkeypatch, router, "_core_remaining_usage_reads")
    budget: Final = _attribute(prometheus_logger, "_increment_remaining_budget_metrics")
    assert callable(budget)
    reads_before_budget: Final[list[int]] = []

    async def budget_step(**kwargs: object) -> None:
        reads_before_budget.append(len(reads.calls))
        if budget_raises:
            raise RuntimeError("budget boom")
        _ = await cast("Awaitable[object]", budget(**kwargs))

    monkeypatch.setattr(prometheus_logger, "_increment_remaining_budget_metrics", budget_step)

    with patch("litellm.proxy.proxy_server.llm_router", router, create=True):
        await _log_success(prometheus_logger, _payload())

    assert reads_before_budget == [0]
    assert reads.calls == ([] if budget_raises else ["bedrock-claude-group"])
    assert _sample_values("litellm_remaining_tokens_metric") == (() if budget_raises else (75.0,))


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from litellm.types.utils import StandardLoggingPayload
