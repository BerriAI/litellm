import asyncio
import copy
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Final

import pytest
from pydantic import BaseModel

import litellm
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.rust_bridge import catalog
from litellm.rust_bridge.catalog import Route, RouteRule
from litellm.rust_bridge.configuration import Rollout
from litellm.types.integrations.custom_logger import AgenticLoopPlan, AgenticLoopRequestPatch
from litellm.types.llms.openai import ResponsesAPIResponse
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger
from tests.test_litellm_rust.support.isolation import rebound
from tests.test_litellm_rust.support.recording_server import RecordingServer
from tests.test_litellm_rust.support.requests import (
    MESSAGES,
    MESSAGES_EVENTS,
    MESSAGES_MODEL,
    MESSAGES_RESPONSE,
    RESPONSES_EVENTS,
    RESPONSES_MODEL,
    RESPONSES_RESPONSE,
)

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
        rule: Final = RouteRule(self.route, Rollout.RUST_REQUIRED)
        with rebound(catalog, "RULES", (rule, *catalog.RULES)):
            if asynchronous:
                return await self.acreate(**arguments)
            return await asyncio.to_thread(self.create, **arguments)


def gap(reason: str, raises: type[Exception] = AssertionError) -> pytest.MarkDecorator:
    return pytest.mark.xfail(reason=reason, raises=raises, strict=True)


def assert_transport(server: RecordingServer, count: int = 1) -> None:
    assert len(server.requests) == count
    assert not any(request.headers.get("user-agent", "").startswith("python-httpx") for request in server.requests)


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


def _anthropic_error(kind: str) -> Mapping[str, object]:
    return {"type": "error", "error": {"type": kind, "message": kind}}


def _openai_error(kind: str) -> Mapping[str, object]:
    return {"error": {"type": kind, "message": kind, "param": None, "code": None}}


MESSAGES_SURFACE: Final = Surface(
    route=Route.MESSAGES,
    request={"model": MESSAGES_MODEL, "messages": list(MESSAGES), "max_tokens": 64},
    response=MESSAGES_RESPONSE,
    events=MESSAGES_EVENTS,
    bad_request=_anthropic_error("invalid_request_error"),
    rate_limited=_anthropic_error("rate_limit_error"),
    create=litellm.anthropic.messages.create,
    acreate=litellm.anthropic.messages.acreate,
)
RESPONSES_SURFACE: Final = Surface(
    route=Route.RESPONSES,
    request={"model": RESPONSES_MODEL, "input": "Hello"},
    response=RESPONSES_RESPONSE,
    events=RESPONSES_EVENTS,
    bad_request=_openai_error("invalid_request_error"),
    rate_limited=_openai_error("rate_limit_exceeded"),
    create=litellm.responses,
    acreate=litellm.aresponses,
)
