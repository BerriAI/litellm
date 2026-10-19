from datetime import datetime
import json
from typing import Final

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.websearch_interception.handler import WebSearchInterceptionLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.anthropic.pass_through.messages import interceptors
from litellm.llms.anthropic.pass_through.messages.interceptors.base import MessagesInterceptor
from tests.test_litellm_rust.messages.data import BODY_HOOKS, MESSAGES_RESPONSE, REQUEST, call, messages_reply
from tests.test_litellm_rust.support.callback_recorder import drain_logging
from tests.test_litellm_rust.support.body_hooks import LegacyAgenticLoop, PlannedAgenticLoop, agentic_hooks
from tests.test_litellm_rust.support.isolation import rebound
from tests._support.provider import SharedProvider
from tests._support.wire import Reply
from tests.test_litellm_rust.support.results import response_id
from tests.test_litellm_rust.support.routing import CALL_ID, assert_served_natively

pytestmark = pytest.mark.requires_rust_extension


@pytest.mark.asyncio
@BODY_HOOKS
async def test_body_pre_request_hook_rewrites_the_provider_request(server: SharedProvider) -> None:
    class Rewrite(CustomLogger):
        async def async_pre_request_hook(
            self, model: str, messages: list[dict[str, object]], kwargs: dict[str, object]
        ) -> dict[str, object]:
            return {**kwargs, "system": "body hook instruction"}

    server.expect(messages_reply())
    with rebound(litellm, "callbacks", [Rewrite()]):
        await call(server)

    requests: Final = assert_served_natively(server)
    body: Final = json.loads(requests[0].body)
    assert isinstance(body, dict)
    assert body.get("system") == "body hook instruction"


@pytest.mark.asyncio
@BODY_HOOKS
async def test_body_websearch_short_circuit_answers_without_the_provider(server: SharedProvider) -> None:
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
        response: Final = await call(server)

    assert response_id(response) == "short-circuited"
    requests: Final = server.received()
    assert requests == ()


@pytest.mark.asyncio
@BODY_HOOKS
async def test_body_interceptor_replaces_the_provider_call(server: SharedProvider) -> None:
    class Replace(MessagesInterceptor):
        def can_handle(self, tools: list[dict[str, object]] | None, custom_llm_provider: str | None) -> bool:
            return True

        async def handle(self, **kwargs: object) -> object:
            return {**MESSAGES_RESPONSE, "id": "intercepted"}

    with rebound(interceptors, "_interceptors", [Replace()]):
        response: Final = await call(server)

    assert response_id(response) == "intercepted"
    requests: Final = server.received()
    assert requests == ()


@pytest.mark.asyncio
@BODY_HOOKS
async def test_body_planned_agentic_loop_sends_the_followup_and_replaces_the_response(server: SharedProvider) -> None:
    loop: Final = PlannedAgenticLoop()
    server.expect(messages_reply(), messages_reply({**MESSAGES_RESPONSE, "id": "followup"}))

    with rebound(litellm, "callbacks", [loop]):
        response: Final = await call(server)
        await drain_logging()

    requests: Final = assert_served_natively(server, count=2)
    second_body: Final = json.loads(requests[1].body)
    assert isinstance(second_body, dict)
    assert second_body["messages"] == [{"role": "user", "content": "followup"}]
    assert response_id(response) == "agentic-replaced"
    assert agentic_hooks(loop) == (
        "async_should_run_agentic_loop",
        "async_build_agentic_loop_plan",
        "async_should_run_agentic_loop",
        "async_post_agentic_loop_response_hook",
    )


@pytest.mark.asyncio
@BODY_HOOKS
async def test_body_legacy_agentic_loop_replaces_the_response(server: SharedProvider) -> None:
    loop: Final = LegacyAgenticLoop()

    server.expect(messages_reply())
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
        call_type="anthropic_messages",
        litellm_call_id=CALL_ID,
        function_id=CALL_ID,
    )
    logging_obj.update_environment_variables(litellm_params={"logger_fn": observed.append}, optional_params={})

    server.expect(messages_reply())
    await call(server, litellm_logging_obj=logging_obj)

    assert_served_natively(server)
    assert observed == []
