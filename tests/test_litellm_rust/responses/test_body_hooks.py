from datetime import datetime
import json
from typing import Final

import pytest
from pydantic import BaseModel

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from tests.test_litellm_rust.responses.data import (
    BODY_HOOKS,
    REQUEST,
    RESPONSES_EVENTS,
    RESPONSES_RESPONSE,
    STREAMING,
    call,
    responses_reply,
)
from tests.test_litellm_rust.support.body_hooks import LegacyAgenticLoop, PlannedAgenticLoop, agentic_hooks
from tests.test_litellm_rust.support.callback_recorder import drain_logging
from tests.test_litellm_rust.support.isolation import rebound
from tests.test_litellm_rust.support.results import consume, response_id
from tests.test_litellm_rust.support.routing import CALL_ID, assert_served_natively
from tests._support.provider import SharedProvider

pytestmark = pytest.mark.requires_rust_extension


@pytest.mark.asyncio
@STREAMING
async def test_body_streaming_deployment_hook_rewrites_chunks_before_delivery(server: SharedProvider) -> None:
    seen: Final[list[str]] = []

    class Rewrite(CustomLogger):
        async def async_post_call_streaming_deployment_hook(
            self, request_data: dict[str, object], response_chunk: BaseModel, call_type: object
        ) -> BaseModel:
            seen.append(getattr(response_chunk, "type", ""))
            if getattr(response_chunk, "type", None) != "response.output_text.delta":
                return response_chunk
            return response_chunk.model_copy(update={"delta": "rewritten"})

    with rebound(litellm, "callbacks", [Rewrite()]):
        stream: Final = await call(server, stream=True)
        chunks: Final = await consume(stream)

    assert_served_natively(server)
    assert seen == [event for event, _ in RESPONSES_EVENTS]
    assert [getattr(chunk, "delta", None) for chunk in chunks] == [None, "rewritten", None]


@pytest.mark.asyncio
@BODY_HOOKS
async def test_body_planned_agentic_loop_sends_the_followup_and_replaces_the_response(server: SharedProvider) -> None:
    loop: Final = PlannedAgenticLoop()
    server.expect(responses_reply(), responses_reply({**RESPONSES_RESPONSE, "id": "followup"}))

    with rebound(litellm, "callbacks", [loop]):
        response: Final = await call(server)
        await drain_logging()

    requests: Final = assert_served_natively(server, count=2)
    second_body: Final = json.loads(requests[1].body)
    assert isinstance(second_body, dict)
    assert second_body["input"] == [{"role": "user", "content": "followup"}]
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
async def test_body_legacy_agentic_loop_replaces_the_response(server: SharedProvider) -> None:
    loop: Final = LegacyAgenticLoop()

    server.expect(responses_reply())
    with rebound(litellm, "callbacks", [loop]):
        response: Final = await call(server)

    assert_served_natively(server, response)
    assert response_id(response) == "agentic-replaced"
    assert agentic_hooks(loop) == ("async_should_run_agentic_loop", "async_run_agentic_loop")


@pytest.mark.asyncio
async def test_body_logging_setup_drops_a_logger_fn_from_a_supplied_logging_object(server: SharedProvider) -> None:
    observed: Final[list[object]] = []
    logging_obj: Final = Logging(
        model=REQUEST["model"],
        messages=[],
        stream=False,
        start_time=datetime.now(),
        call_type="aresponses",
        litellm_call_id=CALL_ID,
        function_id=CALL_ID,
    )
    logging_obj.update_environment_variables(litellm_params={"logger_fn": observed.append}, optional_params={})

    server.expect(responses_reply())
    await call(server, litellm_logging_obj=logging_obj)

    assert_served_natively(server)
    assert observed == []
