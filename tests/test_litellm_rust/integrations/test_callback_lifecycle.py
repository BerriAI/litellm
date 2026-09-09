import asyncio
import gc
import json
import threading
import weakref
from collections.abc import Mapping
from datetime import datetime
from typing import Final

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.types.utils import CallTypes
from tests.test_litellm_rust.conftest import Backend, isolated_backend
from tests.test_litellm_rust.integrations import MESSAGES_ROUTE, MESSAGES_STREAM, OCR_ASYNC, Route, provider_response
from tests.test_litellm_rust.support.callback_recorder import (
    LiveReferenceLogger,
    RecordingLogger,
    SecondaryLiveReferenceLogger,
    drain_logging,
)
from tests.test_litellm_rust.support.provenance import has_rust_response_marker
from tests.test_litellm_rust.support.recording_server import ResponseSpec, recording_service
from tests.test_litellm_rust.support.requests import MESSAGES, MESSAGES_EVENTS

pytestmark = pytest.mark.requires_rust_extension


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("python", "rust"))
async def test_terminal_callbacks_receive_the_same_live_kwargs_and_response(backend: Backend) -> None:
    async with isolated_backend(backend):
        with recording_service() as provider:
            provider.default_response = provider_response(MESSAGES_ROUTE)
            first: Final = LiveReferenceLogger()
            second: Final = SecondaryLiveReferenceLogger()
            response: Final = await MESSAGES_ROUTE.invoke(provider, callbacks=[first, second])
            first_event: Final = (await first.wait_for_async())[0]
            second_event: Final = (await second.wait_for_async())[0]

            assert first_event.kwargs is second_event.kwargs
            assert first_event.response is second_event.response
            assert has_rust_response_marker(response) is (backend == "rust")
            first.release()
            second.release()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("python", "rust"))
async def test_redacted_and_plain_loggers_receive_separate_payloads_with_shared_metadata(backend: Backend) -> None:
    class Capture(CustomLogger):
        def __init__(self, redact: bool) -> None:
            super().__init__(turn_off_message_logging=redact)
            self.kwargs: object = None

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            self.kwargs = kwargs

    async with isolated_backend(backend):
        with recording_service() as provider:
            provider.default_response = provider_response(MESSAGES_ROUTE)
            redacted: Final = Capture(True)
            plain: Final = Capture(False)
            response: Final = await MESSAGES_ROUTE.invoke(provider, callbacks=[redacted, plain])
            await drain_logging()

            redacted_kwargs: Final = redacted.kwargs
            plain_kwargs: Final = plain.kwargs
            assert isinstance(redacted_kwargs, dict)
            assert isinstance(plain_kwargs, dict)
            redacted_payload: Final = redacted_kwargs["standard_logging_object"]
            plain_payload: Final = plain_kwargs["standard_logging_object"]
            assert redacted_kwargs is not plain_kwargs
            assert redacted_payload is not plain_payload
            assert redacted_payload["metadata"] is plain_payload["metadata"]
            assert "redacted-by-litellm" in json.dumps(redacted_payload["messages"])
            assert MESSAGES_ROUTE.expected_text in json.dumps(plain_payload["response"])
            assert "redacted-by-litellm" not in json.dumps(plain_payload)
            assert has_rust_response_marker(response) is (backend == "rust")


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("python", "rust"))
@pytest.mark.parametrize("accepted", (True, False), ids=("accepted", "rejected"))
async def test_deferred_terminal_callback_runs_only_after_proxy_completion_decision(
    backend: Backend, accepted: bool
) -> None:
    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

    async with isolated_backend(backend):
        with recording_service() as provider:
            provider.default_response = provider_response(MESSAGES_ROUTE)
            recorder: Final = RecordingLogger()
            logger: Final = Logging(
                model=MESSAGES_ROUTE.provider_model,
                messages=MESSAGES,
                stream=False,
                call_type=MESSAGES_ROUTE.call_type,
                start_time=datetime.now(),
                litellm_call_id="deferred-completion",
                function_id="deferred-completion",
                dynamic_async_success_callbacks=[recorder],
                dynamic_async_failure_callbacks=[recorder],
            )
            logger._defer_async_logging = True
            response: Final = await MESSAGES_ROUTE.invoke(provider, litellm_logging_obj=logger)
            await asyncio.sleep(0)
            assert "async_log_success_event" not in recorder.names

            ProxyBaseLLMRequestProcessing._flush_deferred_async_logging(
                logging_obj=logger,
                exception_raised=not accepted,
            )
            if accepted:
                accepted_events: Final = await recorder.wait_for_async("async_log_success_event")
                assert len(accepted_events) == 1
                assert "async_log_failure_event" not in recorder.names
            else:
                error: Final = RuntimeError("post-call guardrail rejected response")
                await logger.async_failure_handler(error, str(error))
                rejected_events: Final = await recorder.wait_for_async("async_log_failure_event")
                assert len(rejected_events) == 1
                assert "async_log_success_event" not in recorder.names
            assert has_rust_response_marker(response) is (backend == "rust")


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("python", "rust"))
async def test_suspended_terminal_callback_retains_metadata_until_completion(backend: Backend) -> None:
    class Root:
        pass

    class Suspended(CustomLogger):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.saw_retained_root = False

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            self.saw_retained_root = kwargs["litellm_params"]["metadata"]["retained"] is not None
            self.started.set()
            await self.release.wait()

    async def invoke() -> weakref.ReferenceType[Root]:
        async with isolated_backend(backend):
            with recording_service() as provider:
                provider.default_response = provider_response(MESSAGES_ROUTE)
                callback: Final = Suspended()
                root = Root()
                reference: Final = weakref.ref(root)
                response: Final = await MESSAGES_ROUTE.invoke(
                    provider,
                    callbacks=[callback],
                    metadata={"retained": root},
                )
                del root
                await asyncio.wait_for(callback.started.wait(), timeout=2)
                assert callback.saw_retained_root is True
                gc.collect()
                assert reference() is not None
                assert has_rust_response_marker(response) is (backend == "rust")
                callback.release.set()
                await drain_logging()
                del response
                return reference

    reference: Final = await invoke()
    await asyncio.sleep(0)
    gc.collect()
    assert reference() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("python", "rust"))
@pytest.mark.parametrize(
    ("route", "phase"),
    (
        (OCR_ASYNC, "pre"),
        (MESSAGES_ROUTE, "pre"),
        (MESSAGES_STREAM, "pre"),
        (OCR_ASYNC, "success"),
        (MESSAGES_ROUTE, "success"),
    ),
    ids=("ocr-pre", "messages-pre", "messages-stream-pre", "ocr-success", "messages-success"),
)
async def test_deployment_rejection_logs_terminal_failure_without_calling_deployment_failure_hook(
    backend: Backend, route: Route, phase: str
) -> None:
    class Reject(CustomLogger):
        async def async_pre_call_deployment_hook(
            self, kwargs: dict[str, object], call_type: CallTypes | None
        ) -> dict[str, object]:
            if phase == "pre":
                raise litellm.BadRequestError("deployment rejected", "test-provider", route.provider_model)
            return kwargs

        async def async_post_call_success_deployment_hook(
            self, request_data: Mapping[str, object], response: object, call_type: CallTypes | None
        ) -> object:
            raise litellm.BadRequestError("deployment rejected", "test-provider", route.provider_model)

    async with isolated_backend(backend):
        with recording_service() as provider:
            provider.expected_requests = 0 if phase == "pre" else 1
            provider.default_response = provider_response(route)
            recorder: Final = RecordingLogger()
            litellm.callbacks.extend((Reject(), recorder))

            with pytest.raises(litellm.BadRequestError, match="deployment rejected"):
                await route.invoke(provider)
            await recorder.wait_for_async("async_log_failure_event")

            assert len(provider.requests) == provider.expected_requests
            assert "async_log_success_event" not in recorder.names
            assert recorder.names.count("async_post_call_failure_deployment_hook") == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ("provider_error", "truncated", "close"))
async def test_established_rust_stream_finishes_once_without_deployment_failure_hook(ending: str) -> None:
    with recording_service() as provider:
        provider.default_response = ResponseSpec(
            body=None,
            events=(
                (
                    *MESSAGES_EVENTS[:1],
                    (
                        "error",
                        {"type": "error", "error": {"type": "overloaded_error", "message": "upstream overloaded"}},
                    ),
                )
                if ending == "provider_error"
                else MESSAGES_EVENTS[:-1]
                if ending == "truncated"
                else MESSAGES_EVENTS
            ),
        )
        release: Final = threading.Event()
        if ending == "close":
            provider.enqueue(
                ResponseSpec(
                    body=None,
                    chunks=tuple(
                        f"event: {event}\ndata: {json.dumps(data)}\n\n".encode() for event, data in MESSAGES_EVENTS
                    ),
                    release=release,
                )
            )
        recorder: Final = RecordingLogger()
        litellm.callbacks.append(recorder)
        try:
            stream: Final = await MESSAGES_STREAM.open_stream(provider)
            assert has_rust_response_marker(stream)
            assert recorder.names.count("async_pre_call_deployment_hook") == 1
            assert "async_post_call_failure_deployment_hook" not in recorder.names

            if ending == "close":
                assert await anext(stream)
                await asyncio.wait_for(stream.aclose(), timeout=1)
                await stream.aclose()
                assert "async_log_success_event" not in recorder.names
                assert "async_log_failure_event" not in recorder.names
                release.set()
                await recorder.wait_for_async("async_log_success_event")
            else:
                try:
                    async for _ in stream:
                        pass
                except litellm.APIError:
                    pass
            await drain_logging()
        finally:
            release.set()

        assert len(provider.requests) == 1
        assert recorder.names.count("async_log_failure_event") == (0 if ending == "close" else 1), recorder.names
        assert recorder.names.count("async_log_success_event") == (1 if ending == "close" else 0), recorder.names
        assert recorder.names.count("async_post_call_failure_deployment_hook") == 0
