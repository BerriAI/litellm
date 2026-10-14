from collections.abc import Iterator
from datetime import datetime
from typing import Final

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.websearch_interception.handler import WebSearchInterceptionLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.anthropic.pass_through.messages import interceptors
from litellm.llms.anthropic.pass_through.messages.interceptors.base import MessagesInterceptor
from tests.test_litellm_rust.messages.data import BODY_HOOKS, MESSAGES_RESPONSE, MESSAGES_SURFACE
from tests.test_litellm_rust.support.callback_recorder import drain_logging
from tests.test_litellm_rust.support.callbacks import (
    CALL_ID,
    LegacyAgenticLoop,
    PlannedAgenticLoop,
    agentic_hooks,
    relaxed,
    response_id,
)
from tests.test_litellm_rust.support.isolation import rebound
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.routing import assert_served_natively

pytestmark = pytest.mark.requires_rust_extension

SURFACE: Final = MESSAGES_SURFACE


@pytest.fixture
def server(server: RecordingServer) -> Iterator[RecordingServer]:
    yield from relaxed(server)


@pytest.mark.asyncio
@BODY_HOOKS
async def test_body_pre_request_hook_rewrites_the_provider_request(server: RecordingServer) -> None:
    class Rewrite(CustomLogger):
        async def async_pre_request_hook(
            self, model: str, messages: list[dict[str, object]], kwargs: dict[str, object]
        ) -> dict[str, object]:
            return {**kwargs, "system": "body hook instruction"}

    with rebound(litellm, "callbacks", [Rewrite()]):
        await SURFACE.call(server)

    assert_served_natively(server)
    assert isinstance(server.requests[0].body, dict)
    assert server.requests[0].body.get("system") == "body hook instruction"


@pytest.mark.asyncio
@BODY_HOOKS
async def test_body_websearch_short_circuit_answers_without_the_provider(server: RecordingServer) -> None:
    class ShortCircuit(WebSearchInterceptionLogger):
        async def try_short_circuit_search(
            self,
            model: str,
            messages: list[dict[str, object]],
            tools: list[dict[str, object]] | None,
            custom_llm_provider: str | None,
            kwargs: dict[str, object] | None = None,
        ) -> object:
            return {**MESSAGES_RESPONSE, "id": "short-circuited"}

    with rebound(litellm, "callbacks", [ShortCircuit()]):
        response: Final = await SURFACE.call(server)

    assert response_id(response) == "short-circuited"
    assert server.requests == []


@pytest.mark.asyncio
@BODY_HOOKS
async def test_body_interceptor_replaces_the_provider_call(server: RecordingServer) -> None:
    class Replace(MessagesInterceptor):
        def can_handle(self, tools: list[dict[str, object]] | None, custom_llm_provider: str | None) -> bool:
            return True

        async def handle(self, **kwargs: object) -> object:
            return {**MESSAGES_RESPONSE, "id": "intercepted"}

    with rebound(interceptors, "_interceptors", [Replace()]):
        response: Final = await SURFACE.call(server)

    assert response_id(response) == "intercepted"
    assert server.requests == []


@pytest.mark.asyncio
@BODY_HOOKS
async def test_body_planned_agentic_loop_sends_the_followup_and_replaces_the_response(server: RecordingServer) -> None:
    loop: Final = PlannedAgenticLoop()
    server.enqueue(ResponseSpec(body=MESSAGES_RESPONSE))
    server.default_response = ResponseSpec(body={**MESSAGES_RESPONSE, "id": "followup"})

    with rebound(litellm, "callbacks", [loop]):
        response: Final = await SURFACE.call(server)
        await drain_logging()

    assert_served_natively(server, count=2)
    assert isinstance(server.requests[1].body, dict)
    assert server.requests[1].body["messages"] == [{"role": "user", "content": "followup"}]
    assert response_id(response) == "agentic-replaced"
    assert agentic_hooks(loop) == (
        "async_should_run_agentic_loop",
        "async_build_agentic_loop_plan",
        "async_should_run_agentic_loop",
        "async_post_agentic_loop_response_hook",
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
        model=MESSAGES_SURFACE.request["model"],
        messages=[],
        stream=False,
        start_time=datetime.now(),
        call_type="anthropic_messages",
        litellm_call_id=CALL_ID,
        function_id=CALL_ID,
    )
    logging_obj.update_environment_variables(litellm_params={"logger_fn": observed.append}, optional_params={})

    await SURFACE.call(server, litellm_logging_obj=logging_obj)

    assert_served_natively(server)
    assert observed == []
