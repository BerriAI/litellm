from typing import Final

import pytest
from pydantic import BaseModel

import litellm
from tests.test_litellm_rust.messages.callback_contracts import (
    AGENTIC_HOOKS,
    BACKENDS,
    CALLABLE_REGISTRATIONS,
    FAILURE_HOOKS,
    RESPONSES_RESPONSE,
    SUCCESS_HOOKS,
    assert_agentic_contract,
    assert_transport,
    call,
    assert_callable_contract,
    consume,
    assert_dataset_contract,
    assert_deployment_contract,
    assert_failure_contract,
    assert_logger_fn_contract,
    assert_success_contract,
)
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger
from tests.test_litellm_rust.support.isolation import rebound
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec

pytestmark = pytest.mark.requires_rust_extension


@pytest.mark.asyncio
@pytest.mark.parametrize("native", BACKENDS)
@pytest.mark.parametrize(
    "hook,asynchronous",
    (
        *(pytest.param(hook, True, id=f"async-{hook}") for hook in SUCCESS_HOOKS),
        *(pytest.param(hook, False, id=f"sync-{hook}") for hook in SUCCESS_HOOKS[:4]),
    ),
)
async def test_inference_success_callback_contract(
    recording_server: RecordingServer, native: bool, hook: str, asynchronous: bool
) -> None:
    await assert_success_contract(recording_server, "responses", native, hook, asynchronous)


@pytest.mark.asyncio
@pytest.mark.parametrize("native", BACKENDS)
@pytest.mark.parametrize("hook", FAILURE_HOOKS)
async def test_inference_failure_callback_contract(recording_server: RecordingServer, native: bool, hook: str) -> None:
    await assert_failure_contract(recording_server, "responses", native, hook)


@pytest.mark.asyncio
@pytest.mark.parametrize("native", BACKENDS)
@pytest.mark.parametrize("hook", ("async_pre_call_deployment_hook", "async_post_call_success_deployment_hook"))
async def test_deployment_hooks_change_the_wire_request_and_public_response(
    recording_server: RecordingServer, native: bool, hook: str
) -> None:
    await assert_deployment_contract(recording_server, "responses", native, hook)


@pytest.mark.asyncio
@pytest.mark.parametrize("native", BACKENDS)
@pytest.mark.parametrize("event,asynchronous,dynamic", CALLABLE_REGISTRATIONS)
async def test_callable_inference_callback_contract(
    recording_server: RecordingServer, native: bool, event: str, asynchronous: bool, dynamic: bool
) -> None:
    await assert_callable_contract(recording_server, "responses", native, event, asynchronous, dynamic)


@pytest.mark.asyncio
@pytest.mark.parametrize("native", BACKENDS)
@pytest.mark.parametrize("hook", AGENTIC_HOOKS)
async def test_body_agentic_callback_contract(recording_server: RecordingServer, native: bool, hook: str) -> None:
    await assert_agentic_contract(recording_server, "responses", native, hook)


@pytest.mark.asyncio
@pytest.mark.parametrize("native", BACKENDS)
async def test_body_logging_setup_clears_a_supplied_logger_fn(recording_server: RecordingServer, native: bool) -> None:
    await assert_logger_fn_contract(recording_server, "responses", native)


@pytest.mark.asyncio
@pytest.mark.parametrize("native", BACKENDS)
async def test_body_streaming_deployment_hook_rewrites_chunks_before_delivery(
    recording_server: RecordingServer, native: bool
) -> None:
    recorder: Final = RecordingLogger()

    class Rewrite(RecordingLogger):
        async def async_post_call_streaming_deployment_hook(
            self, request_data: dict[str, object], response_chunk: BaseModel, call_type: object
        ) -> BaseModel:
            recorder._record("async_post_call_streaming_deployment_hook", request_data, response_chunk)
            if getattr(response_chunk, "type", None) == "response.output_text.delta":
                return response_chunk.model_copy(update={"delta": "rewritten"})
            return response_chunk

    recording_server.expected_requests = None
    recording_server.default_response = ResponseSpec(
        body=None,
        events=(
            (
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 1,
                    "item_id": "msg_callbacks",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "original",
                },
            ),
            (
                "response.completed",
                {"type": "response.completed", "sequence_number": 2, "response": RESPONSES_RESPONSE},
            ),
        ),
    )
    with rebound(litellm, "callbacks", [Rewrite()]):
        stream: Final = await call(recording_server, "responses", native, stream=True)
        chunks: Final = await consume(stream)

    assert_transport(recording_server, native)
    assert getattr(chunks[0], "delta", None) == "rewritten"
    assert len(recorder.events) == len(chunks)
    assert getattr(recorder.events[0].response, "delta", None) == "original"


@pytest.mark.asyncio
@pytest.mark.parametrize("native", BACKENDS)
async def test_dataset_hook_filters_the_async_success_sink_before_delivery(
    recording_server: RecordingServer, native: bool
) -> None:
    await assert_dataset_contract(recording_server, "responses", native)
