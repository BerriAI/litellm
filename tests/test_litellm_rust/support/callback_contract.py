import asyncio
import threading
from collections.abc import Awaitable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Protocol

import pytest
from pydantic import TypeAdapter

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.rust_bridge import catalog
from litellm.rust_bridge.catalog import Route, RouteRule
from litellm.rust_bridge.configuration import Rollout
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.support.isolation import rebound
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.requests import request_body, request_headers

OBJECT: Final = TypeAdapter(dict[str, object])
CONTEXT: Final = ContextVar("callback-contract", default="unset")


class SyncCall(Protocol):
    def __call__(self, **kwargs: object) -> object: ...


class AsyncCall(Protocol):
    def __call__(self, **kwargs: object) -> Awaitable[object]: ...


class ResponseText(Protocol):
    def __call__(self, response: object) -> str: ...


class ReplaceResponse(Protocol):
    def __call__(self, response: object, text: str) -> object: ...


@dataclass(frozen=True, slots=True)
class CallbackRoute:
    route: Route
    arguments: Mapping[str, object]
    response: Mapping[str, object]
    call_types: tuple[str, str]
    sync: SyncCall
    asynchronous: AsyncCall
    text: ResponseText
    replace: ReplaceResponse
    expected_text: str

    async def invoke(self, server: RecordingServer, asynchronous: bool = True, **options: object) -> object:
        arguments: Final = {**self.arguments, "api_key": "test-key", "api_base": server.base_url, **options}
        with rebound(catalog, "RULES", (RouteRule(self.route, Rollout.RUST_REQUIRED), *catalog.RULES)):
            if asynchronous:
                return await self.asynchronous(**arguments)
            return await asyncio.to_thread(self.sync, **arguments)

    def logger(self, recorder: RecordingLogger, asynchronous: bool = True) -> Logging:
        model: Final = self.arguments["model"]
        assert isinstance(model, str)
        return Logging(
            model=model,
            messages=[],
            stream=False,
            call_type=self.call_types[int(asynchronous)],
            start_time=datetime.now(),
            litellm_call_id="contract-call",
            function_id="contract-function",
            dynamic_async_success_callbacks=[recorder],
            dynamic_async_failure_callbacks=[recorder],
        )


@pytest.fixture
def callback_server(callback_route: CallbackRoute, recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=dict(callback_route.response))
    return recording_server


class TestCallbackContract:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("asynchronous", (False, True), ids=("sync", "async"))
    @pytest.mark.parametrize("registration", ("global", "request"))
    async def test_callback_success_contract(
        self, callback_route: CallbackRoute, callback_server: RecordingServer, asynchronous: bool, registration: str
    ) -> None:
        recorder: Final = RecordingLogger()
        options: Final = {"callbacks": [recorder]} if registration == "request" else {}
        if registration == "global":
            litellm.callbacks.append(recorder)
        response: Final = await callback_route.invoke(
            callback_server,
            asynchronous,
            litellm_call_id="contract-success",
            metadata={"contract": "metadata"},
            **options,
        )
        event_name: Final = "async_log_success_event" if asynchronous else "log_success_event"
        success: Final = await recorder.wait_for_async(event_name)
        pre: Final = recorder.wait_for("log_pre_api_call")
        details: Final = OBJECT.validate_python(success[0].kwargs)
        params: Final = OBJECT.validate_python(details["litellm_params"])
        assert len(callback_server.requests) == len(pre) == len(success) == 1
        assert callback_route.text(response) == callback_route.expected_text
        assert callback_route.text(success[0].response) == callback_route.expected_text
        assert success[0].call_type == callback_route.call_types[int(asynchronous)]
        assert details["litellm_call_id"] == "contract-success"
        assert OBJECT.validate_python(params["metadata"])["contract"] == "metadata"
        assert request_body(OBJECT.validate_python(pre[0].kwargs)) == callback_server.requests[0].body
        assert recorder.names.count("log_post_api_call") == 1
        assert recorder.names.index("log_pre_api_call") < recorder.names.index("log_post_api_call")
        assert recorder.names.index("log_post_api_call") < recorder.names.index(event_name)
        assert not any("failure" in name for name in recorder.names)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("asynchronous", (False, True), ids=("sync", "async"))
    @pytest.mark.parametrize("registration", ("global", "request"))
    async def test_callback_failure_contract(
        self, callback_route: CallbackRoute, callback_server: RecordingServer, asynchronous: bool, registration: str
    ) -> None:
        callback_server.enqueue(ResponseSpec(body={"error": {"message": "contract rejection"}}, status=400))
        recorder: Final = RecordingLogger()
        options: Final = {"callbacks": [recorder]} if registration == "request" else {}
        if registration == "global":
            litellm.callbacks.append(recorder)
        with pytest.raises(litellm.BadRequestError) as caught:
            await callback_route.invoke(callback_server, asynchronous, **options)
        await drain_logging()
        failures: Final = tuple(event for event in recorder.events if "failure" in event.name)
        expected: Final = ("log_failure_event", "async_log_failure_event") if asynchronous else ("log_failure_event",)
        assert tuple(event.name for event in failures) == expected
        assert all(OBJECT.validate_python(event.kwargs)["exception"] is caught.value for event in failures)
        assert all(event.call_type == callback_route.call_types[int(asynchronous)] for event in failures)
        assert all(event.response is None for event in failures)
        assert len(callback_server.requests) == 1
        assert not any("success" in name for name in recorder.names)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("raise_after_edit", (False, True), ids=("returns", "raises"))
    async def test_callback_edits_survive_later_observers(
        self, callback_route: CallbackRoute, callback_server: RecordingServer, raise_after_edit: bool
    ) -> None:
        retained: Final[list[dict[str, object]]] = []
        recorder: Final = RecordingLogger()

        class Retain(CustomLogger):
            def log_pre_api_call(self, model: object, messages: object, kwargs: dict[str, object]) -> None:
                retained.append(request_body(kwargs))

        class Edit(CustomLogger):
            def log_pre_api_call(self, model: object, messages: object, kwargs: dict[str, object]) -> None:
                request_body(kwargs)["callback_extension"] = {"nested": ["edited"]}
                request_headers(kwargs)["x-callback-contract"] = "edited"
                kwargs["callback_state"] = "retained"
                if raise_after_edit:
                    raise RuntimeError("observer failed after editing")

        await callback_route.invoke(callback_server, callbacks=[Retain(), Edit(), recorder])
        success: Final = await recorder.wait_for_async("async_log_success_event")
        pre: Final = OBJECT.validate_python(recorder.wait_for("log_pre_api_call")[0].kwargs)
        sent: Final = callback_server.requests[0]
        assert retained[0] == request_body(pre) == sent.body
        assert retained[0]["callback_extension"] == {"nested": ["edited"]}
        assert sent.headers["x-callback-contract"] == "edited"
        assert OBJECT.validate_python(success[0].kwargs)["callback_state"] == "retained"
        assert len(callback_server.requests) == len(success) == 1

    @pytest.mark.asyncio
    async def test_deployment_hook_receives_logger_before_provider_io(
        self, callback_route: CallbackRoute, callback_server: RecordingServer
    ) -> None:
        observed: Final[list[object]] = []
        recorder: Final = RecordingLogger()

        class RequireLogger(CustomLogger):
            async def async_pre_call_deployment_hook(self, kwargs: dict[str, object], call_type: object) -> object:
                logger: Final = kwargs["litellm_logging_obj"]
                observed.append(logger)
                return {**kwargs, "extra_headers": {"x-deployment-hook": "ran"}}

        litellm.callbacks.append(RequireLogger())
        await callback_route.invoke(callback_server, callbacks=[recorder])
        success: Final = await recorder.wait_for_async("async_log_success_event")
        assert len(observed) == len(success) == len(callback_server.requests) == 1
        assert isinstance(observed[0], Logging)
        assert callback_server.requests[0].headers["x-deployment-hook"] == "ran"

    @pytest.mark.asyncio
    async def test_deployment_replacement_precedes_success_logging(
        self, callback_route: CallbackRoute, callback_server: RecordingServer
    ) -> None:
        phases: Final[list[tuple[str, object, str]]] = []
        caller: Final = asyncio.current_task()
        recorder: Final = RecordingLogger()

        class Replace(CustomLogger):
            async def async_pre_call_deployment_hook(self, kwargs: dict[str, object], call_type: object) -> object:
                CONTEXT.set("deployment")
                phases.append(("pre", asyncio.current_task(), CONTEXT.get()))
                return kwargs

            async def async_post_call_success_deployment_hook(
                self, request_data: dict[str, object], response: object, call_type: object
            ) -> object:
                phases.append(("post", asyncio.current_task(), CONTEXT.get()))
                return callback_route.replace(response, "reviewed by callback")

        litellm.callbacks.append(Replace())
        response: Final = await callback_route.invoke(callback_server, callbacks=[recorder])
        success: Final = await recorder.wait_for_async("async_log_success_event")
        assert phases == [("pre", caller, "deployment"), ("post", caller, "deployment")]
        assert callback_route.text(response) == "reviewed by callback"
        assert callback_route.text(success[0].response) == "reviewed by callback"
        assert len(callback_server.requests) == len(success) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("phase", ("pre", "post"))
    async def test_blocking_hook_selects_failure_only(
        self, callback_route: CallbackRoute, callback_server: RecordingServer, phase: str
    ) -> None:
        rejected: Final = RuntimeError("contract guardrail rejected")
        recorder: Final = RecordingLogger()
        callback_server.expected_requests = int(phase == "post")

        class Block(CustomGuardrail):
            async def async_pre_call_deployment_hook(self, kwargs: dict[str, object], call_type: object) -> object:
                if phase == "pre":
                    raise rejected
                return kwargs

            async def async_post_call_success_deployment_hook(
                self, request_data: dict[str, object], response: object, call_type: object
            ) -> object:
                raise rejected

        litellm.callbacks.append(Block())
        with pytest.raises(RuntimeError, match="contract guardrail rejected") as caught:
            await callback_route.invoke(callback_server, callbacks=[recorder])
        await drain_logging()
        failures: Final = tuple(event for event in recorder.events if "failure" in event.name)
        assert caught.value is rejected
        assert tuple(event.name for event in failures) == ("log_failure_event", "async_log_failure_event")
        assert all(OBJECT.validate_python(event.kwargs)["exception"] is caught.value for event in failures)
        assert len(callback_server.requests) == int(phase == "post")
        assert not any("success" in name for name in recorder.names)

    @pytest.mark.asyncio
    async def test_failing_failure_observer_preserves_error_and_remaining_observers(
        self, callback_route: CallbackRoute, callback_server: RecordingServer
    ) -> None:
        callback_server.enqueue(ResponseSpec(body={"error": {"message": "contract rejection"}}, status=400))
        recorder: Final = RecordingLogger()
        seen: Final[list[tuple[str, object]]] = []

        class Broken(CustomLogger):
            def log_failure_event(
                self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
            ) -> None:
                seen.append(("sync", kwargs["exception"]))
                raise RuntimeError("sync failure observer")

            async def async_log_failure_event(
                self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
            ) -> None:
                seen.append(("async", kwargs["exception"]))
                raise RuntimeError("async failure observer")

        with pytest.raises(litellm.BadRequestError) as caught:
            await callback_route.invoke(callback_server, callbacks=[Broken(), recorder])
        assert seen == [("sync", caught.value), ("async", caught.value)]
        assert tuple(name for name in recorder.names if "failure" in name) == (
            "log_failure_event",
            "async_log_failure_event",
        )
        assert len(callback_server.requests) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("asynchronous", (False, True), ids=("sync", "async"))
    async def test_callback_registration_is_deduplicated(
        self, callback_route: CallbackRoute, callback_server: RecordingServer, asynchronous: bool
    ) -> None:
        recorder: Final = RecordingLogger()
        await callback_route.invoke(
            callback_server, asynchronous, callbacks=[recorder, recorder], success_callback=[recorder]
        )
        name: Final = "async_log_success_event" if asynchronous else "log_success_event"
        events: Final = await recorder.wait_for_async(name)
        assert len(events) == 1
        assert recorder.names.count("log_pre_api_call") == 1
        assert not any("failure" in name for name in recorder.names)
        assert len(callback_server.requests) == 1

    @pytest.mark.asyncio
    async def test_pre_api_callback_runs_in_caller_context(
        self, callback_route: CallbackRoute, callback_server: RecordingServer
    ) -> None:
        recorder: Final = RecordingLogger()
        await callback_route.invoke(callback_server, callbacks=[recorder])
        pre: Final = recorder.wait_for("log_pre_api_call")
        assert len(pre) == 1
        assert pre[0].thread is threading.current_thread()
        assert pre[0].loop is asyncio.get_running_loop()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("asynchronous", (False, True), ids=("sync", "async"))
    async def test_success_observer_failure_does_not_replace_response(
        self, callback_route: CallbackRoute, callback_server: RecordingServer, asynchronous: bool
    ) -> None:
        recorder: Final = RecordingLogger()

        class Broken(CustomLogger):
            def log_success_event(
                self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
            ) -> None:
                raise RuntimeError("success observer failed")

            async def async_log_success_event(
                self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
            ) -> None:
                raise RuntimeError("async success observer failed")

        response: Final = await callback_route.invoke(callback_server, asynchronous, callbacks=[Broken(), recorder])
        event_name: Final = "async_log_success_event" if asynchronous else "log_success_event"
        events: Final = await recorder.wait_for_async(event_name)
        assert callback_route.text(response) == callback_route.expected_text
        assert len(events) == len(callback_server.requests) == 1
        assert not any("failure" in name for name in recorder.names)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("blocked", (False, True), ids=("released", "discarded"))
    async def test_deferred_success_is_released_at_most_once(
        self, callback_route: CallbackRoute, callback_server: RecordingServer, blocked: bool
    ) -> None:
        recorder: Final = RecordingLogger()
        logger: Final = callback_route.logger(recorder)
        logger._defer_async_logging = True
        await callback_route.invoke(callback_server, litellm_logging_obj=logger)
        await drain_logging()
        assert "async_log_success_event" not in recorder.names
        ProxyBaseLLMRequestProcessing._flush_deferred_async_logging(logger, blocked)
        ProxyBaseLLMRequestProcessing._flush_deferred_async_logging(logger, blocked)
        await drain_logging()
        assert recorder.names.count("async_log_success_event") == int(not blocked)
        assert not any("failure" in name for name in recorder.names)
        assert len(callback_server.requests) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("phase", ("pre", "http"))
    async def test_cancelled_call_has_no_terminal_callbacks(
        self, callback_route: CallbackRoute, callback_server: RecordingServer, phase: str
    ) -> None:
        entered: Final = asyncio.Event()
        recorder: Final = RecordingLogger()

        class Pause(CustomLogger):
            async def async_pre_call_deployment_hook(self, kwargs: dict[str, object], call_type: object) -> object:
                if phase == "pre":
                    entered.set()
                    await asyncio.Event().wait()
                return kwargs

        litellm.callbacks.append(Pause())
        callback_server.expected_requests = int(phase == "http")
        if phase == "http":
            callback_server.enqueue(ResponseSpec(body=dict(callback_route.response), delay=0.2))
        task: Final = asyncio.create_task(callback_route.invoke(callback_server, callbacks=[recorder]))
        if phase == "pre":
            await asyncio.wait_for(entered.wait(), 5)
        else:
            await callback_server.wait_for_requests(1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await drain_logging()
        assert not any("success" in name or "failure" in name for name in recorder.names)
        assert len(callback_server.requests) == int(phase == "http")
