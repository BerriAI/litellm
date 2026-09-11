import asyncio
import contextvars
import datetime
import weakref
from collections.abc import Callable, Coroutine
from concurrent.futures import Future, ThreadPoolExecutor
from importlib import import_module
from threading import get_ident
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
from litellm.litellm_core_utils import thread_pool_executor
from litellm.litellm_core_utils.call_completion import CallCompletion, PythonCompletion
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.rust_bridge import ocr as rust_ocr_bridge
from litellm.utils import client


class RecordingExecutor:
    def __init__(self) -> None:
        self.submissions: list[tuple[Callable[..., object], tuple[object, ...]]] = []

    def submit(self, function: Callable[..., object], *args: object) -> Future[object]:
        self.submissions.append((function, args))
        future: Final[Future[object]] = Future()
        future.set_result(function(*args))
        return future


class RecordingCompletion:
    def __init__(self) -> None:
        self.successes: list[object] = []
        self.failures: list[Exception] = []

    def success(
        self,
        result: object,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        self.successes.append(result)

    def failure(
        self,
        exception: Exception,
        traceback_exception: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        self.failures.append(exception)

    async def async_failure(
        self,
        exception: Exception,
        traceback_exception: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        self.failures.append(exception)


class RecordingLogging:
    def __init__(self, marker: contextvars.ContextVar[str], observed: list[tuple[object, str]]) -> None:
        self._marker = marker
        self._observed = observed
        self._defer_async_logging = False
        self._enqueue_deferred_logging: Callable[[], None] | None = None

    def success_handler(
        self,
        result: object,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        self._observed.append((result, self._marker.get()))

    def failure_handler(
        self,
        exception: Exception,
        traceback_exception: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None: ...

    async def async_success_handler(
        self,
        result: object,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None: ...

    async def async_failure_handler(
        self,
        exception: Exception,
        traceback_exception: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None: ...

    def handle_sync_success_callbacks_for_async_calls(
        self,
        result: object,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None: ...


def test_python_completion_preserves_sync_context_and_response_identity() -> None:
    marker: Final[contextvars.ContextVar[str]] = contextvars.ContextVar("marker")
    marker.set("request-context")
    executor: Final = RecordingExecutor()
    response: Final = object()
    observed: Final[list[tuple[object, str]]] = []
    logging_obj: Final = RecordingLogging(marker, observed)
    completion: Final = PythonCompletion(
        logging_obj,
        executor,
        async_call=False,
        internal_call=False,
        completion_with_fallbacks=False,
    )
    now: Final = datetime.datetime.now(datetime.timezone.utc)

    completion.success(response, now, now)

    assert observed == [(response, "request-context")]
    assert len(executor.submissions) == 1


def test_sync_wrapper_dispatches_with_logging_executor_and_caller_context(monkeypatch: pytest.MonkeyPatch) -> None:
    marker: Final[contextvars.ContextVar[str]] = contextvars.ContextVar("wrapper-context", default="missing")
    marker.set("request-context")
    caller_thread: Final = get_ident()
    response: Final = object()
    observed: Final[list[tuple[object, str, int]]] = []
    logging_obj: Final = MagicMock()

    def record_success(result: object, start_time: datetime.datetime, end_time: datetime.datetime) -> None:
        observed.append((result, marker.get(), get_ident()))

    def ocr(**kwargs: object) -> object:
        return response

    logging_obj.success_handler.side_effect = record_success
    monkeypatch.setattr("litellm.utils.function_setup", MagicMock(return_value=(logging_obj, {})))
    monkeypatch.setattr("litellm.utils.load_credentials_from_list", MagicMock())
    wrapped: Final = client(ocr)
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(thread_pool_executor, "executor", executor)
        result: Final = wrapped()

    assert result is response
    assert len(observed) == 1
    assert observed[0][0] is response
    assert observed[0][1] == "request-context"
    assert observed[0][2] != caller_thread


@pytest.mark.asyncio
async def test_call_completion_attaches_once_and_forwards_final_objects() -> None:
    python_completion: Final = RecordingCompletion()
    native_completion: Final = RecordingCompletion()
    ignored_completion: Final = RecordingCompletion()
    completion: Final = CallCompletion(python_completion)
    response: Final = object()
    error: Final = ValueError("mapped failure")
    now: Final = datetime.datetime.now(datetime.timezone.utc)

    assert completion.python_implementation is python_completion
    assert completion.attach(native_completion)
    assert not completion.attach(ignored_completion)

    completion.success(response, now, now)
    completion.failure(error, "traceback", now, now)
    await completion.async_failure(error, "traceback", now, now)

    assert native_completion.successes == [response]
    assert native_completion.failures == [error, error]
    assert python_completion.successes == []
    assert ignored_completion.successes == []


@pytest.mark.asyncio
async def test_python_completion_retains_deferred_success_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    response: Final = object()
    logging_obj: Final = MagicMock()
    logging_obj._defer_async_logging = True
    logging_obj.async_success_handler = AsyncMock()
    completion: Final = CallCompletion(
        PythonCompletion(
            logging_obj,
            RecordingExecutor(),
            async_call=True,
            internal_call=False,
            completion_with_fallbacks=False,
        )
    )
    worker: Final = MagicMock()
    monkeypatch.setattr("litellm.litellm_core_utils.logging_worker.GLOBAL_LOGGING_WORKER", worker)
    scheduled: Final[list[Coroutine[object, object, None]]] = []
    monkeypatch.setattr("asyncio.create_task", scheduled.append)
    now: Final = datetime.datetime.now(datetime.timezone.utc)

    completion.success(response, now, now)
    completion.release()

    logging_obj._enqueue_deferred_logging()
    assert len(scheduled) == 1
    await scheduled[0]
    await worker.ensure_initialized_and_enqueue.call_args.kwargs["async_coroutine"]
    logging_obj.async_success_handler.assert_awaited_once_with(result=response, start_time=now, end_time=now)


@pytest.mark.asyncio
async def test_async_ocr_wrapper_injects_completion_after_fresh_deployment_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_completion: Final = RecordingCompletion()
    original_response: Final = object()
    replacement_response: Final = object()
    shared_metadata: Final = {"request": "shared"}
    replacement_kwargs: Final[dict[str, object]] = {"metadata": shared_metadata}
    hook_input: dict[str, object] | None = None

    async def fresh_kwargs(kwargs: dict[str, object], call_type: str) -> dict[str, object]:
        nonlocal hook_input
        hook_input = kwargs
        return replacement_kwargs

    async def aocr(**kwargs: object) -> object:
        completion = kwargs.get("_litellm_call_completion")
        assert isinstance(completion, CallCompletion)
        assert completion.attach(native_completion)
        assert kwargs["metadata"] is shared_metadata
        return original_response

    async def replace_response(request_data: dict[str, object], response: object, call_type: object) -> object:
        assert response is original_response
        return replacement_response

    monkeypatch.setattr("litellm.utils.async_pre_call_deployment_hook", fresh_kwargs)
    monkeypatch.setattr("litellm.utils.async_post_call_success_deployment_hook", replace_response)
    monkeypatch.setattr("litellm.utils.function_setup", MagicMock(return_value=(MagicMock(), {})))
    monkeypatch.setattr("litellm.utils.load_credentials_from_list", MagicMock())
    wrapped: Final = client(aocr)

    result: Final = await wrapped()

    assert result is replacement_response
    assert native_completion.successes == [replacement_response]
    assert hook_input is not None
    assert "_litellm_call_completion" not in hook_input
    assert "_litellm_call_completion" not in replacement_kwargs
    assert replacement_kwargs["metadata"] is shared_metadata


@pytest.mark.asyncio
async def test_async_ocr_wrapper_sends_final_failure_to_attached_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_completion: Final = RecordingCompletion()
    mapped_error: Final = ValueError("mapped OCR failure")

    async def aocr(**kwargs: object) -> object:
        completion = kwargs.get("_litellm_call_completion")
        assert isinstance(completion, CallCompletion)
        assert completion.attach(native_completion)
        raise mapped_error

    monkeypatch.setattr("litellm.utils.function_setup", MagicMock(return_value=(MagicMock(), {})))
    monkeypatch.setattr("litellm.utils.load_credentials_from_list", MagicMock())
    wrapped: Final = client(aocr)

    with pytest.raises(ValueError, match="mapped OCR failure") as caught:
        await wrapped()

    assert caught.value is mapped_error
    assert native_completion.failures == [mapped_error, mapped_error]


@pytest.mark.asyncio
async def test_async_ocr_wrapper_reports_metadata_failure_without_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_completion: Final = RecordingCompletion()
    response: Final = object()
    metadata_error: Final = ValueError("metadata failure")

    async def aocr(**kwargs: object) -> object:
        completion = kwargs.get("_litellm_call_completion")
        assert isinstance(completion, CallCompletion)
        assert completion.attach(native_completion)
        return response

    def fail_metadata(**kwargs: object) -> None:
        raise metadata_error

    monkeypatch.setattr("litellm.utils.function_setup", MagicMock(return_value=(MagicMock(), {})))
    monkeypatch.setattr("litellm.utils.load_credentials_from_list", MagicMock())
    monkeypatch.setattr("litellm.utils.update_response_metadata", fail_metadata)
    wrapped: Final = client(aocr)

    with pytest.raises(ValueError, match="metadata failure") as caught:
        await wrapped()

    assert caught.value is metadata_error
    assert native_completion.successes == []
    assert native_completion.failures == [metadata_error, metadata_error]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_ocr_completion_stays_separate_from_marshaled_provider_options(
    monkeypatch: pytest.MonkeyPatch, asynchronous: bool
) -> None:
    native_completion: Final = RecordingCompletion()
    response: Final = OCRResponse(model="mistral-ocr-latest", pages=[])
    metadata: Final = {"request": "shared"}
    pages: Final = [0, 2]

    def run(
        request: rust_ocr_bridge.LiteLLMOcrRequest,
        resolve_secret: Callable[[str], str | None],
        convert_file_document: Callable[[dict[str, object]], dict[str, str]],
    ) -> OCRResponse:
        assert request.kwargs["metadata"] is metadata
        assert "_litellm_call_completion" not in request.kwargs
        marshalled: Final = rust_ocr_bridge._marshal(request, resolve_secret, convert_file_document)
        assert "_litellm_call_completion" not in marshalled.kwargs
        assert marshalled.kwargs["pages"] is pages
        assert marshalled.call_completion is request.call_completion
        assert marshalled.call_completion is not None
        assert marshalled.call_completion.attach(native_completion)
        return response

    async def arun(
        request: rust_ocr_bridge.LiteLLMOcrRequest,
        resolve_secret: Callable[[str], str | None],
        convert_file_document: Callable[[dict[str, object]], dict[str, str]],
    ) -> OCRResponse:
        return run(request, resolve_secret, convert_file_document)

    def run_bridge(
        request: rust_ocr_bridge.LiteLLMOcrRequest,
        resolve_api_key: Callable[[str], str | None],
    ) -> OCRResponse:
        return run(request, resolve_api_key, lambda document: {})

    async def arun_bridge(
        request: rust_ocr_bridge.LiteLLMOcrRequest,
        resolve_api_key: Callable[[str], str | None],
    ) -> OCRResponse:
        return await arun(request, resolve_api_key, lambda document: {})

    ocr_main: Final = import_module("litellm.ocr.main")
    monkeypatch.setattr(ocr_main, "rust_enabled", lambda: True)
    monkeypatch.setattr(ocr_main, "_rust_ocr_supported", lambda request: True)
    monkeypatch.setattr(ocr_main, "_run_rust_ocr", run_bridge)
    monkeypatch.setattr(ocr_main, "_run_rust_aocr", arun_bridge)
    arguments: Final = {
        "model": "mistral/mistral-ocr-latest",
        "document": {"type": "document_url", "document_url": "https://example.com/doc.pdf"},
        "api_key": "test-key",
        "metadata": metadata,
        "pages": pages,
    }

    result: Final = await litellm.aocr(**arguments) if asynchronous else litellm.ocr(**arguments)

    assert result is response
    assert native_completion.successes == [response]
    assert native_completion.failures == []
    assert arguments["metadata"] is metadata
    assert "_litellm_call_completion" not in arguments


@pytest.mark.parametrize(
    ("asynchronous", "exit_path"),
    [(False, "success"), (True, "success"), (False, "callback_error"), (True, "callback_error"), (True, "cancelled")],
)
@pytest.mark.asyncio
async def test_wrapper_releases_completion_resources_on_every_exit(
    monkeypatch: pytest.MonkeyPatch, asynchronous: bool, exit_path: str
) -> None:
    retained: Final[list[tuple[CallCompletion, weakref.ReferenceType[object]]]] = []
    callback_error: Final = RuntimeError("failure callback failed")
    implementation: Final = MagicMock(spec=RecordingCompletion)
    implementation.failure.side_effect = callback_error
    implementation.async_failure = AsyncMock()
    response: Final = object()

    def ocr(*, _litellm_call_completion: CallCompletion, **kwargs: object) -> object:
        retained.append((_litellm_call_completion, weakref.ref(_litellm_call_completion.python_implementation)))
        assert _litellm_call_completion.attach(implementation)
        if exit_path == "cancelled":
            raise asyncio.CancelledError
        if exit_path == "callback_error":
            raise ValueError("provider failed")
        return response

    async def aocr(*, _litellm_call_completion: CallCompletion, **kwargs: object) -> object:
        return ocr(_litellm_call_completion=_litellm_call_completion, **kwargs)

    monkeypatch.setattr("litellm.utils.function_setup", MagicMock(return_value=(MagicMock(), {})))
    monkeypatch.setattr("litellm.utils.load_credentials_from_list", MagicMock())
    wrapped: Final = client(aocr if asynchronous else ocr)

    if exit_path == "success":
        result: Final = await wrapped() if asynchronous else wrapped()
        assert result is response
        assert implementation.success.call_args.args[0] is response
    elif exit_path == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            await wrapped()
        implementation.success.assert_not_called()
        implementation.failure.assert_not_called()
    else:
        with pytest.raises(RuntimeError, match="failure callback failed") as caught:
            await wrapped() if asynchronous else wrapped()
        assert caught.value is callback_error
        implementation.async_failure.assert_not_called()

    assert len(retained) == 1
    assert retained[0][1]() is None
