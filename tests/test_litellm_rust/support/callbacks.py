import asyncio
import copy
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Final

import pytest
from pydantic import BaseModel

from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.rust_bridge.catalog import Route
from litellm.types.integrations.custom_logger import AgenticLoopPlan, AgenticLoopRequestPatch
from litellm.types.llms.openai import ResponsesAPIResponse
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger
from tests.test_litellm_rust.support.recording_server import RecordingServer
from tests.test_litellm_rust.support.routing import native_route

CALL_ID: Final = "callback-contract"


@dataclass(frozen=True, slots=True)
class Surface:
    route: Route
    request: Mapping[str, object]
    response: Mapping[str, object]
    events: tuple[tuple[str, object], ...]
    bad_request: Mapping[str, object]
    rate_limited: Mapping[str, object]
    create: Callable[..., object]
    acreate: Callable[..., Awaitable[object]]

    async def call(self, server: RecordingServer, *, asynchronous: bool = True, **kwargs: object) -> object:
        arguments: Final = {
            **copy.deepcopy(dict(self.request)),
            "api_key": "test-key",
            "api_base": server.base_url,
            "litellm_call_id": CALL_ID,
            **kwargs,
        }
        with native_route(self.route):
            if asynchronous:
                return await self.acreate(**arguments)
            return await asyncio.to_thread(self.create, **arguments)


def gap(reason: str, raises: type[Exception] = AssertionError) -> pytest.MarkDecorator:
    return pytest.mark.xfail(reason=reason, raises=raises, strict=True)


def response_id(response: object) -> object:
    if isinstance(response, ResponsesAPIResponse):
        return ResponsesAPIRequestUtils.decode_responses_api_response_id(response.id).get("response_id") or response.id
    return response.get("id") if isinstance(response, dict) else None


def with_id(response: object, new_id: str) -> object:
    if isinstance(response, BaseModel):
        return response.model_copy(update={"id": new_id})
    assert isinstance(response, dict)
    return {**response, "id": new_id}


async def consume(stream: object) -> tuple[object, ...]:
    if isinstance(stream, AsyncIterator):
        return tuple([chunk async for chunk in stream])
    assert isinstance(stream, Iterator)
    return tuple(stream)


class HookRecorder(RecordingLogger):
    def log_post_api_call(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self._record("log_post_api_call", kwargs, response_obj)

    async def async_logging_hook(
        self, kwargs: dict[str, object], result: object, call_type: str
    ) -> tuple[dict[str, object], object]:
        self._record("async_logging_hook", kwargs, result)
        return kwargs, result

    def redact_standard_logging_payload_from_model_call_details(
        self, model_call_details: dict[str, object]
    ) -> dict[str, object]:
        self._record("redact_standard_logging_payload_from_model_call_details", model_call_details)
        return model_call_details

    async def async_pre_call_deployment_hook(self, kwargs: dict[str, object], call_type: object) -> dict[str, object]:
        self._record("async_pre_call_deployment_hook", kwargs)
        return kwargs

    async def async_post_call_success_deployment_hook(
        self, request_data: dict[str, object], response: object, call_type: object
    ) -> object:
        self._record("async_post_call_success_deployment_hook", request_data, response)
        return response

    async def async_post_call_failure_deployment_hook(
        self,
        request_data: dict[str, object],
        original_exception: Exception,
        call_type: object,
        *,
        fallback_depth: int | None = None,
    ) -> None:
        self._record("async_post_call_failure_deployment_hook", request_data, original_exception)

    async def log_model_group_rate_limit_error(
        self, exception: Exception, original_model_group: str | None, kwargs: dict[str, object]
    ) -> None:
        self._record("log_model_group_rate_limit_error", kwargs, exception)


def agentic_hooks(recorder: RecordingLogger) -> tuple[str, ...]:
    return tuple(name for name in recorder.names if "agentic" in name)


class PlannedAgenticLoop(RecordingLogger):
    async def async_should_run_agentic_loop(self, response: object, **kwargs: object) -> tuple[bool, dict[str, object]]:
        self._record("async_should_run_agentic_loop", response=response)
        return response_id(response) != "followup", {}

    async def async_build_agentic_loop_plan(self, response: object, **kwargs: object) -> AgenticLoopPlan:
        self._record("async_build_agentic_loop_plan", response=response)
        return AgenticLoopPlan(
            run_agentic_loop=True,
            request_patch=AgenticLoopRequestPatch.model_validate(
                {"messages": [{"role": "user", "content": "followup"}]}
            ),
        )

    async def async_post_agentic_loop_response_hook(self, response: object, **kwargs: object) -> object:
        self._record("async_post_agentic_loop_response_hook", response=response)
        return with_id(response, "agentic-replaced")

    async def async_agentic_loop_cleanup_hook(self, **kwargs: object) -> None:
        self._record("async_agentic_loop_cleanup_hook")


class LegacyAgenticLoop(RecordingLogger):
    async def async_should_run_agentic_loop(self, response: object, **kwargs: object) -> tuple[bool, dict[str, object]]:
        self._record("async_should_run_agentic_loop", response=response)
        return True, {}

    async def async_run_agentic_loop(self, response: object, **kwargs: object) -> object:
        self._record("async_run_agentic_loop", response=response)
        return with_id(response, "agentic-replaced")


def relaxed(server: RecordingServer) -> Iterator[RecordingServer]:
    server.expected_requests = None
    yield server
    server.responses.clear()
