import asyncio
import sys
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from types import ModuleType
from typing import Final, Literal

import pytest
from fastapi import HTTPException
from prometheus_client import REGISTRY

import litellm
from litellm.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.prometheus import PrometheusLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.logging_worker import LoggingWorker
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.common_utils.callback_utils import initialize_callbacks_on_proxy
from litellm.proxy.logging_endpoints.callback_logs_endpoints import ingest_callback_logs
from litellm.proxy.utils import ProxyLogging
from litellm.types.proxy.callback_logs_endpoints import CallbackLogRecord, CallbackLogsRequest
from litellm.types.utils import CallTypesLiteral, ModelResponse

_METRIC: Final = "litellm_callback_duration_seconds"
_TIMESTAMP: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
_Outcome = Literal["return", "raise", "cancel"]


class _Callback(CustomLogger):
    def __init__(self, outcome: _Outcome = "return") -> None:
        super().__init__()
        self.outcome = outcome
        self.started = asyncio.Event()
        self.blocker = asyncio.Event()

    async def _run(self) -> None:
        self.started.set()
        if self.outcome == "raise":
            raise HTTPException(status_code=403, detail="Callback rejected the request")
        if self.outcome == "cancel":
            await self.blocker.wait()

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: Mapping[str, object],
        call_type: CallTypesLiteral,
    ) -> dict[str, object]:
        await self._run()
        return {**data, "updated_by_callback": True}

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        await self._run()


@pytest.fixture
def _registry_scope(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    original: Final = frozenset(REGISTRY._collector_to_names)
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "_async_success_callback", [])
    monkeypatch.setattr(litellm, "prometheus_metrics_config", [{"group": "callback latency", "metrics": [_METRIC]}])
    yield
    for collector in frozenset(REGISTRY._collector_to_names) - original:
        REGISTRY.unregister(collector)


def _register(monkeypatch: pytest.MonkeyPatch, *, enabled: bool, callback: CustomLogger) -> None:
    module: Final = ModuleType("callback_latency_test_plugin")
    setattr(module, "first", callback)
    setattr(module, "second", _Callback())
    setattr(module, "noop", CustomLogger())
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(litellm, "prometheus_callback_latency", enabled, raising=False)
    monkeypatch.setattr(litellm, "callbacks", [PrometheusLogger()])
    initialize_callbacks_on_proxy(
        value=[f"{module.__name__}.{name}" for name in ("first", "second", "noop")],
        premium_user=False,
        config_file_path="",
        litellm_settings={},
    )


def _counts() -> dict[tuple[tuple[str, str], ...], float]:
    samples: Final = (
        sample for metric in REGISTRY.collect() for sample in metric.samples if sample.name == f"{_METRIC}_count"
    )
    return {tuple(sorted(sample.labels.items())): sample.value for sample in samples}


def _labels(name: str, hook: str, path: str) -> tuple[tuple[str, str], ...]:
    return (("callback_name", f"callback_latency_test_plugin.{name}"), ("execution_path", path), ("hook", hook))


async def _pre_call(*, stream: bool = False, call_type: CallTypesLiteral = "acompletion") -> Mapping[str, object]:
    return await ProxyLogging(user_api_key_cache=DualCache()).pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        data={"model": "chat", "messages": [{"role": "user", "content": "private input"}], "stream": stream},
        call_type=call_type,
    )


def _success_handler(call_type: str = "acompletion") -> Logging:
    return Logging(
        model="chat",
        messages=[],
        stream=False,
        call_type=call_type,
        start_time=_TIMESTAMP,
        litellm_call_id="private call id",
        function_id="test",
        dynamic_async_success_callbacks=[
            callback for callback in litellm.callbacks if not isinstance(callback, PrometheusLogger)
        ],
    )


@pytest.mark.usefixtures("_registry_scope")
@pytest.mark.parametrize(
    ("enabled", "stream", "call_type", "records"),
    [
        (True, False, "acompletion", True),
        (False, False, "acompletion", False),
        (True, True, "acompletion", False),
        (True, False, "aembedding", False),
    ],
)
async def test_pre_call_records_only_enabled_nonstreaming_chat_hooks(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, stream: bool, call_type: CallTypesLiteral, records: bool
) -> None:
    _register(monkeypatch, enabled=enabled, callback=_Callback())

    result: Final = await _pre_call(stream=stream, call_type=call_type)

    assert result["updated_by_callback"] is True
    assert _counts() == (
        {
            _labels("first", "async_pre_call_hook", "request_path"): 1,
            _labels("second", "async_pre_call_hook", "request_path"): 1,
        }
        if records
        else {}
    )
    if not enabled:
        assert _METRIC not in {family.name for family in REGISTRY.collect()}


@pytest.mark.usefixtures("_registry_scope")
@pytest.mark.parametrize("hook", ["async_pre_call_hook", "async_log_success_event"])
@pytest.mark.parametrize("outcome", ["raise", "cancel"])
async def test_failed_and_cancelled_hooks_record_once(
    monkeypatch: pytest.MonkeyPatch, hook: str, outcome: _Outcome
) -> None:
    callback: Final = _Callback(outcome)
    _register(monkeypatch, enabled=True, callback=callback)

    call: Final = (
        _pre_call()
        if hook == "async_pre_call_hook"
        else _success_handler().async_success_handler(result=ModelResponse(model="chat"), end_time=_TIMESTAMP)
    )
    if outcome == "cancel":
        task: Final = asyncio.ensure_future(call)
        await asyncio.wait_for(callback.started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    elif hook == "async_pre_call_hook":
        with pytest.raises(HTTPException, match="Callback rejected the request"):
            await call
    else:
        await call
        assert _counts()[_labels("second", hook, "request_path")] == 1

    assert _counts()[_labels("first", hook, "request_path")] == 1


@pytest.mark.usefixtures("_registry_scope")
@pytest.mark.parametrize(
    ("enabled", "call_type"), [(True, "acompletion"), (False, "acompletion"), (True, "aembedding")]
)
async def test_success_hooks_follow_the_logging_worker_execution_path(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, call_type: str
) -> None:
    callback: Final = _Callback()
    _register(monkeypatch, enabled=enabled, callback=callback)
    handler: Final = _success_handler(call_type)
    worker: Final = LoggingWorker()
    try:
        worker.ensure_initialized_and_enqueue(
            handler.async_success_handler(result=ModelResponse(model="chat"), end_time=_TIMESTAMP)
        )
        await asyncio.wait_for(worker.flush(), timeout=5)
    finally:
        await worker.stop(discard_pending=True)

    assert callback.started.is_set()
    assert _counts() == (
        {
            _labels("first", "async_log_success_event", "background"): 1,
            _labels("second", "async_log_success_event", "background"): 1,
        }
        if enabled and call_type == "acompletion"
        else {}
    )


@pytest.mark.usefixtures("_registry_scope")
async def test_metric_name_comes_from_registration_despite_callback_state_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback: Final = _Callback()
    monkeypatch.setattr(callback, "callback_registration_name", "callback-owned state", raising=False)
    _register(monkeypatch, enabled=True, callback=callback)

    await _pre_call()
    monkeypatch.setattr(callback, "callback_registration_name", "request-derived state")
    await _success_handler().async_success_handler(result=ModelResponse(model="chat"), end_time=_TIMESTAMP)

    assert _counts() == {
        _labels("first", "async_pre_call_hook", "request_path"): 1,
        _labels("second", "async_pre_call_hook", "request_path"): 1,
        _labels("first", "async_log_success_event", "request_path"): 1,
        _labels("second", "async_log_success_event", "request_path"): 1,
    }


@pytest.mark.usefixtures("_registry_scope")
@pytest.mark.parametrize(
    "call_fields",
    [
        {"call_type": "acompletion", "stream": True},
        {"call_type": "acompletion", "stream": False},
        {"stream": True},
    ],
)
async def test_replaying_finished_chat_delivers_success_without_observing_latency(
    monkeypatch: pytest.MonkeyPatch, call_fields: Mapping[str, object]
) -> None:
    callback: Final = _Callback()
    _register(monkeypatch, enabled=True, callback=callback)
    monkeypatch.setattr(litellm, "_async_success_callback", [callback])
    record: Final = CallbackLogRecord(
        status="success",
        standard_logging_payload={
            "id": "replayed chat",
            "model": "chat",
            "startTime": _TIMESTAMP.timestamp(),
            "endTime": _TIMESTAMP.timestamp(),
            **call_fields,
        },
    )

    result: Final = await ingest_callback_logs(
        CallbackLogsRequest(records=[record]),
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
    )

    assert (result.processed, result.failed) == (1, 0)
    assert callback.started.is_set()
    assert _counts() == {}
