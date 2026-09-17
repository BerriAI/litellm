import asyncio
import datetime
import inspect
from collections.abc import Iterator
from typing import Final
from unittest.mock import Mock

import pytest

from litellm.integrations.athina import AthinaLogger
from litellm.integrations.dynamodb import DyanmoDBLogger
from litellm.integrations.greenscale import GreenscaleLogger
from litellm.integrations.helicone import HeliconeLogger
from litellm.integrations.langfuse.langfuse import LangFuseLogger
from litellm.integrations.logfire_logger import LogfireLevel, LogfireLogger
from litellm.integrations.lunary import LunaryLogger
from litellm.integrations.openmeter import OpenMeterLogger
from litellm.integrations.prompt_layer import PromptLayerLogger
from litellm.integrations.s3 import S3Logger
from litellm.integrations.supabase import Supabase
from litellm.integrations.traceloop import TraceloopLogger
from litellm.integrations.weights_biases import WeightsBiasesLogger
from litellm.litellm_core_utils import litellm_logging
from litellm.rust_bridge import leaves

START: Final = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
END: Final = START + datetime.timedelta(seconds=1)
DETAILS: Final[dict[str, object]] = {
    "litellm_call_id": "call-id",
    "input": "document",
    "user": "user-1",
    "original_response": "raw",
    "litellm_params": {},
}


def _logger() -> Mock:
    return Mock(
        model="mistral/mistral-ocr-latest",
        messages="document",
        litellm_call_id="call-id",
        model_call_details=dict(DETAILS),
        standard_callback_dynamic_params={},
    )


def _stub(cls: type, method: str) -> tuple[object, Mock]:
    instance: Final[object] = object.__new__(cls)
    recorder: Final = Mock(name=f"{cls.__name__}.{method}")
    setattr(instance, method, recorder)
    return instance, recorder


def _assert_binds(cls: type, method: str, recorder: Mock, instance: object) -> None:
    recorder.assert_called_once()
    target: Final[object] = getattr(cls, method)
    assert callable(target)
    signature: Final = inspect.signature(target)
    bound: Final = signature.bind(instance, *recorder.call_args.args, **recorder.call_args.kwargs)
    required: Final = frozenset(
        name for name, parameter in signature.parameters.items() if parameter.default is inspect.Parameter.empty
    )
    assert required <= set(bound.arguments)


SUCCESS_CASES: Final[tuple[tuple[str, str, type, str], ...]] = (
    ("promptlayer", "promptLayerLogger", PromptLayerLogger, "log_event"),
    ("wandb", "weightsBiasesLogger", WeightsBiasesLogger, "log_event"),
    ("athina", "athinaLogger", AthinaLogger, "log_event"),
    ("logfire", "logfireLogger", LogfireLogger, "log_event"),
    ("greenscale", "greenscaleLogger", GreenscaleLogger, "log_event"),
    ("supabase", "supabaseClient", Supabase, "log_event"),
    ("lunary", "lunaryLogger", LunaryLogger, "log_event"),
    ("helicone", "heliconeLogger", HeliconeLogger, "log_success"),
    ("traceloop", "traceloopLogger", TraceloopLogger, "log_event"),
    ("s3", "s3Logger", S3Logger, "log_event"),
)

FAILURE_CASES: Final[tuple[tuple[str, str, type, str], ...]] = (
    ("lunary", "lunaryLogger", LunaryLogger, "log_event"),
    ("supabase", "supabaseClient", Supabase, "log_event"),
    ("traceloop", "traceloopLogger", TraceloopLogger, "log_event"),
    ("logfire", "logfireLogger", LogfireLogger, "log_event"),
)


def _install(monkeypatch: pytest.MonkeyPatch, global_name: str, value: object) -> None:
    monkeypatch.setattr(litellm_logging, global_name, value, raising=False)


@pytest.mark.parametrize(("name", "global_name", "cls", "method"), SUCCESS_CASES)
def test_named_success_calls_integration_with_its_real_signature(
    monkeypatch: pytest.MonkeyPatch, name: str, global_name: str, cls: type, method: str
) -> None:
    instance, recorder = _stub(cls, method)
    _install(monkeypatch, global_name, instance)
    response: Final = object()

    assert leaves.dispatch_named_success(_logger(), name, response, START, END) is None

    _assert_binds(cls, method, recorder, instance)
    kwargs: Final = recorder.call_args.kwargs
    assert kwargs["start_time"] is START and kwargs["end_time"] is END
    assert kwargs["response_obj"] is response
    assert callable(kwargs["print_verbose"])


@pytest.mark.parametrize(("name", "global_name", "cls", "method"), FAILURE_CASES)
def test_named_failure_calls_integration_with_its_real_signature(
    monkeypatch: pytest.MonkeyPatch, name: str, global_name: str, cls: type, method: str
) -> None:
    instance, recorder = _stub(cls, method)
    _install(monkeypatch, global_name, instance)
    error: Final = RuntimeError("boom")

    leaves.dispatch_named_failure(_logger(), name, error, "formatted", START, END)

    _assert_binds(cls, method, recorder, instance)
    kwargs: Final = recorder.call_args.kwargs
    assert kwargs["start_time"] is START and kwargs["end_time"] is END
    assert kwargs.get("response_obj") is None


def test_logfire_receives_enum_levels_and_exception_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    instance, recorder = _stub(LogfireLogger, "log_event")
    _install(monkeypatch, "logfireLogger", instance)
    error: Final = RuntimeError("boom")

    leaves.dispatch_named_success(_logger(), "logfire", object(), START, END)
    assert recorder.call_args.kwargs["level"] is LogfireLevel.INFO
    assert "original_response" not in recorder.call_args.kwargs["kwargs"]

    leaves.dispatch_named_failure(_logger(), "logfire", error, "formatted", START, END)
    assert recorder.call_args.kwargs["level"] is LogfireLevel.ERROR
    assert recorder.call_args.kwargs["kwargs"]["exception"] is error
    assert "original_response" not in recorder.call_args.kwargs["kwargs"]


def test_lunary_marks_success_and_error_events(monkeypatch: pytest.MonkeyPatch) -> None:
    instance, recorder = _stub(LunaryLogger, "log_event")
    _install(monkeypatch, "lunaryLogger", instance)

    leaves.dispatch_named_success(_logger(), "lunary", object(), START, END)
    assert recorder.call_args.kwargs["event"] == "end"
    assert recorder.call_args.kwargs["run_id"] == "call-id"

    leaves.dispatch_named_failure(_logger(), "lunary", RuntimeError("boom"), "formatted", START, END)
    assert recorder.call_args.kwargs["event"] == "error"
    assert recorder.call_args.kwargs["error"] == "formatted"


def test_supabase_request_hook_only_fires_pre_call(monkeypatch: pytest.MonkeyPatch) -> None:
    instance, recorder = _stub(Supabase, "input_log_event")
    _install(monkeypatch, "supabaseClient", instance)

    leaves.dispatch_named_request(_logger(), "supabase", "post_api_call")
    recorder.assert_not_called()

    leaves.dispatch_named_request(_logger(), "supabase", "pre_api_call")
    _assert_binds(Supabase, "input_log_event", recorder, instance)
    assert recorder.call_args.kwargs["end_user"] == "user-1"


def test_sentry_hooks_are_called_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    breadcrumb: Final = Mock()
    capture: Final = Mock()
    _install(monkeypatch, "add_breadcrumb", breadcrumb)
    _install(monkeypatch, "capture_exception", capture)
    error: Final = RuntimeError("boom")

    leaves.dispatch_named_request(_logger(), "sentry", "pre_api_call")
    assert breadcrumb.call_args.kwargs["category"] == "litellm.llm_call"

    leaves.dispatch_named_failure(_logger(), "sentry", error, "formatted", START, END)
    capture.assert_called_once_with(error)


@pytest.mark.parametrize(("name", "global_name"), [(case[0], case[1]) for case in SUCCESS_CASES])
def test_unconfigured_or_foreign_singleton_is_skipped(
    monkeypatch: pytest.MonkeyPatch, name: str, global_name: str
) -> None:
    _install(monkeypatch, global_name, None)
    assert leaves.dispatch_named_success(_logger(), name, object(), START, END) is None

    foreign: Final = Mock()
    _install(monkeypatch, global_name, foreign)
    assert leaves.dispatch_named_success(_logger(), name, object(), START, END) is None
    assert not foreign.method_calls


def _run(awaitable: object) -> None:
    async def consume() -> None:
        assert inspect.isawaitable(awaitable)
        _ = await awaitable

    asyncio.run(consume())


def test_openmeter_returns_the_async_coroutine(monkeypatch: pytest.MonkeyPatch) -> None:
    instance: Final = object.__new__(OpenMeterLogger)
    recorder: Final = Mock()

    async def async_log_success_event(**kwargs: object) -> None:
        recorder(**kwargs)

    monkeypatch.setattr(instance, "async_log_success_event", async_log_success_event)
    _install(monkeypatch, "openMeterLogger", instance)
    response: Final = object()

    _run(leaves.dispatch_named_success(_logger(), "openmeter", response, START, END))

    _assert_binds(OpenMeterLogger, "async_log_success_event", recorder, instance)
    assert recorder.call_args.kwargs["response_obj"] is response


def test_dynamodb_reuses_the_existing_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    instance: Final = object.__new__(DyanmoDBLogger)
    recorder: Final = Mock()

    async def _async_log_event(**kwargs: object) -> None:
        recorder(**kwargs)

    monkeypatch.setattr(instance, "_async_log_event", _async_log_event)
    _install(monkeypatch, "dynamoLogger", instance)

    _run(leaves.dispatch_named_success(_logger(), "dynamodb", object(), START, END))

    _assert_binds(DyanmoDBLogger, "_async_log_event", recorder, instance)
    assert litellm_logging.dynamoLogger is instance


@pytest.fixture
def langfuse_handler(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[LangFuseLogger, Mock, Mock]]:
    handler: Final = object.__new__(LangFuseLogger)
    recorder: Final = Mock(return_value={"trace_id": "trace-1"})
    monkeypatch.setattr(handler, "log_event_on_langfuse", recorder)
    _install(monkeypatch, "langFuseLogger", handler)
    cache: Final = Mock()
    _install(monkeypatch, "in_memory_trace_id_cache", cache)
    yield handler, recorder, cache


def test_langfuse_success_logs_and_caches_trace_id(langfuse_handler: tuple[LangFuseLogger, Mock, Mock]) -> None:
    handler, recorder, cache = langfuse_handler
    response: Final = object()

    leaves.dispatch_named_success(_logger(), "langfuse", response, START, END)

    _assert_binds(LangFuseLogger, "log_event_on_langfuse", recorder, handler)
    kwargs: Final = recorder.call_args.kwargs
    assert kwargs["response_obj"] is response
    assert kwargs["user_id"] == "user-1"
    assert kwargs["level"] == "DEFAULT" and kwargs["status_message"] is None
    assert "original_response" not in kwargs["kwargs"]
    cache.set_cache.assert_called_once_with(litellm_call_id="call-id", service_name="langfuse", trace_id="trace-1")


def test_langfuse_failure_logs_error_level(langfuse_handler: tuple[LangFuseLogger, Mock, Mock]) -> None:
    handler, recorder, _ = langfuse_handler

    leaves.dispatch_named_failure(_logger(), "langfuse", RuntimeError("boom"), "formatted", START, END)

    _assert_binds(LangFuseLogger, "log_event_on_langfuse", recorder, handler)
    kwargs: Final = recorder.call_args.kwargs
    assert kwargs["response_obj"] is None
    assert kwargs["level"] == "ERROR" and kwargs["status_message"] == "boom"


def test_langfuse_skips_cache_without_trace_id(langfuse_handler: tuple[LangFuseLogger, Mock, Mock]) -> None:
    _, recorder, cache = langfuse_handler
    recorder.return_value = {}

    leaves.dispatch_named_success(_logger(), "langfuse", object(), START, END)

    cache.set_cache.assert_not_called()


def test_signature_binding_rejects_drift() -> None:
    instance, recorder = _stub(Supabase, "log_event")
    recorder(model="m", end_userr="typo", response_obj=None, start_time=START, end_time=END)
    with pytest.raises(TypeError):
        _assert_binds(Supabase, "log_event", recorder, instance)
