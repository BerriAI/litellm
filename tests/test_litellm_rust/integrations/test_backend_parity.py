from typing import Final

import pytest

import litellm
from tests.test_litellm_rust.conftest import Backend, isolated_backend
from tests.test_litellm_rust.integrations import (
    MESSAGES_ROUTE,
    OCR_ASYNC,
    OCR_SYNC,
    Route,
    RunObservation,
    provider_response,
    route_id,
    wait_for_callback,
)
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.support.provenance import has_rust_response_marker
from tests.test_litellm_rust.support.recording_server import ResponseSpec, recording_service
from tests.test_litellm_rust.support.requests import CHAT_MESSAGES, CHAT_MODEL, CHAT_RESPONSE

pytestmark = pytest.mark.requires_rust_extension


async def observe_route(backend: Backend, route: Route) -> RunObservation:
    async with isolated_backend(backend):
        with recording_service() as provider:
            provider.default_response = provider_response(route)
            recorder: Final = RecordingLogger()
            response: Final = await route.invoke(provider, callbacks=[recorder])
            event: Final = (await wait_for_callback(route, recorder))[0]
            payload: Final = event.kwargs["standard_logging_object"]
            provider_body: Final = provider.requests[0].body
            if not isinstance(provider_body, dict):
                raise TypeError(f"Expected provider object body, got {type(provider_body).__name__}")
            return RunObservation(
                call_type=payload["call_type"],
                model=payload["model"],
                response_cost=payload["response_cost"],
                response_text=route.response_text(response),
                provider_body=provider_body,
                rust_dispatch=has_rust_response_marker(response),
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("route", (OCR_SYNC, OCR_ASYNC, MESSAGES_ROUTE), ids=route_id)
async def test_python_and_rust_match_public_route_outputs_and_logging(route: Route) -> None:
    python: Final = await observe_route("python", route)
    rust: Final = await observe_route("rust", route)
    python_body: Final = {**python.provider_body, "stream": python.provider_body.get("stream", False)}
    rust_body: Final = {**rust.provider_body, "stream": rust.provider_body.get("stream", False)}

    assert python.call_type == rust.call_type == route.call_type
    assert python.model == rust.model == route.provider_model
    assert python.response_cost == pytest.approx(rust.response_cost)
    assert python.response_text == rust.response_text == route.expected_text
    assert python_body == rust_body
    assert python.rust_dispatch is False
    assert rust.rust_dispatch is True


async def observe_retry(backend: Backend) -> tuple[tuple[str, ...], bool]:
    async with isolated_backend(backend):
        with recording_service() as provider:
            provider.expected_requests = 2
            provider.enqueue(ResponseSpec(body={"message": "retry this attempt"}, status=500))
            provider.default_response = ResponseSpec(body=CHAT_RESPONSE)
            recorder: Final = RecordingLogger()
            response: Final = await litellm.acompletion(
                model=CHAT_MODEL,
                messages=CHAT_MESSAGES,
                api_key="test-key",
                api_base=provider.base_url,
                callbacks=[recorder],
                num_retries=1,
            )
            await drain_logging()
            return recorder.names, has_rust_response_marker(response)


@pytest.mark.asyncio
async def test_python_and_rust_emit_the_same_retry_callback_sequence() -> None:
    python_names, python_used_rust = await observe_retry("python")
    rust_names, rust_used_rust = await observe_retry("rust")

    assert rust_names == python_names
    assert rust_names.count("log_pre_api_call") == 2
    assert rust_names.count("async_log_failure_event") == 1
    assert rust_names.count("async_log_success_event") == 0
    assert python_used_rust is False
    assert rust_used_rust is True
