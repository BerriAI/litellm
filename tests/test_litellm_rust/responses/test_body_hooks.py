from collections.abc import Iterator
from datetime import datetime
from typing import Final

import pytest
from pydantic import BaseModel

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from tests.test_litellm_rust.responses.data import (
    BODY_HOOKS,
    RESPONSES_EVENTS,
    RESPONSES_RESPONSE,
    RESPONSES_SURFACE,
    STREAMING,
)
from tests.test_litellm_rust.support.callback_recorder import drain_logging
from tests.test_litellm_rust.support.callbacks import (
    CALL_ID,
    LegacyAgenticLoop,
    PlannedAgenticLoop,
    agentic_hooks,
    consume,
    relaxed,
    response_id,
)
from tests.test_litellm_rust.support.isolation import rebound
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.routing import assert_served_natively

pytestmark = pytest.mark.requires_rust_extension

SURFACE: Final = RESPONSES_SURFACE
STREAM: Final = ResponseSpec(body=None, events=RESPONSES_EVENTS)


@pytest.fixture
def server(server: RecordingServer) -> Iterator[RecordingServer]:
    yield from relaxed(server)


@pytest.mark.asyncio
@STREAMING
async def test_body_streaming_deployment_hook_rewrites_chunks_before_delivery(server: RecordingServer) -> None:
    seen: Final[list[str]] = []

    class Rewrite(CustomLogger):
        async def async_post_call_streaming_deployment_hook(
            self, request_data: dict[str, object], response_chunk: BaseModel, call_type: object
        ) -> BaseModel:
            seen.append(getattr(response_chunk, "type", ""))
            if getattr(response_chunk, "type", None) != "response.output_text.delta":
                return response_chunk
            return response_chunk.model_copy(update={"delta": "rewritten"})

    server.enqueue(STREAM)
    with rebound(litellm, "callbacks", [Rewrite()]):
        stream: Final = await SURFACE.call(server, stream=True)
        chunks: Final = await consume(stream)

    assert_served_natively(server)
    assert seen == [event for event, _ in RESPONSES_EVENTS]
    assert [getattr(chunk, "delta", None) for chunk in chunks] == [None, "rewritten", None]


@pytest.mark.asyncio
@BODY_HOOKS
async def test_body_planned_agentic_loop_sends_the_followup_and_replaces_the_response(server: RecordingServer) -> None:
    loop: Final = PlannedAgenticLoop()
    server.enqueue(ResponseSpec(body=RESPONSES_RESPONSE))
    server.default_response = ResponseSpec(body={**RESPONSES_RESPONSE, "id": "followup"})

    with rebound(litellm, "callbacks", [loop]):
        response: Final = await SURFACE.call(server)
        await drain_logging()

    assert_served_natively(server, count=2)
    assert isinstance(server.requests[1].body, dict)
    assert server.requests[1].body["input"] == [{"role": "user", "content": "followup"}]
    assert response_id(response) == "agentic-replaced"
    assert agentic_hooks(loop) == (
        "async_should_run_agentic_loop",
        "async_build_agentic_loop_plan",
        "async_should_run_agentic_loop",
        "async_post_agentic_loop_response_hook",
        "async_agentic_loop_cleanup_hook",
    )


@pytest.mark.asyncio
@BODY_HOOKS
async def test_body_legacy_agentic_loop_replaces_the_response(server: RecordingServer) -> None:
    loop: Final = LegacyAgenticLoop()

    with rebound(litellm, "callbacks", [loop]):
        response: Final = await SURFACE.call(server)

    assert_served_natively(server)
    assert response_id(response) == "agentic-replaced"
    assert agentic_hooks(loop) == ("async_should_run_agentic_loop", "async_run_agentic_loop")


@pytest.mark.asyncio
async def test_body_logging_setup_drops_a_logger_fn_from_a_supplied_logging_object(server: RecordingServer) -> None:
    observed: Final[list[object]] = []
    logging_obj: Final = Logging(
        model=RESPONSES_SURFACE.request["model"],
        messages=[],
        stream=False,
        start_time=datetime.now(),
        call_type="aresponses",
        litellm_call_id=CALL_ID,
        function_id=CALL_ID,
    )
    logging_obj.update_environment_variables(litellm_params={"logger_fn": observed.append}, optional_params={})

    await SURFACE.call(server, litellm_logging_obj=logging_obj)

    assert_served_natively(server)
    assert observed == []
