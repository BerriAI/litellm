from collections.abc import Iterator
from typing import Final

import pytest
from fastapi import HTTPException

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import ContentFilterGuardrail
from litellm.rust_bridge import catalog
from litellm.rust_bridge.catalog import Route, RouteRule
from litellm.rust_bridge.configuration import Rollout
from litellm.types.guardrails import BlockedWord, ContentFilterAction, GuardrailEventHooks
from litellm.types.utils import CallTypes
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger
from tests.test_litellm_rust.support.isolation import rebound
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.requests import MESSAGES, MESSAGES_MODEL, MESSAGES_RESPONSE, request_body

pytestmark = pytest.mark.requires_rust_extension


@pytest.fixture(autouse=True)
def opt_messages_into_rust() -> Iterator[None]:
    with rebound(catalog, "RULES", (RouteRule(Route.MESSAGES, Rollout.RUST_OPT_IN), *catalog.RULES)):
        yield


@pytest.fixture
def messages_server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)
    return recording_server


def arguments(server: RecordingServer, **kwargs: object) -> dict[str, object]:
    return {
        "model": MESSAGES_MODEL,
        "messages": [dict(message) for message in MESSAGES],
        "max_tokens": 64,
        "api_key": "test-key",
        "api_base": server.base_url,
        **kwargs,
    }


@pytest.mark.asyncio
async def test_native_messages_pre_call_guardrail_rewrites_the_request_seen_by_provider_and_logger(
    messages_server: RecordingServer,
) -> None:
    seen: Final = []

    class Rewrite(CustomGuardrail):
        def __init__(self) -> None:
            super().__init__(
                guardrail_name="rewrite-messages", event_hook=GuardrailEventHooks.pre_call, default_on=True
            )

        async def async_pre_call_deployment_hook(
            self, kwargs: dict[str, object], call_type: CallTypes | None
        ) -> dict[str, object]:
            seen.append(call_type)
            return {**kwargs, "messages": [{"role": "user", "content": "Reviewed prompt"}]}

    guardrail: Final = Rewrite()
    recorder: Final = RecordingLogger()
    litellm.callbacks.append(guardrail)

    response: Final = await litellm.anthropic.messages.acreate(
        **arguments(messages_server, callbacks=[recorder], guardrails=[guardrail.guardrail_name])
    )
    assert response["content"] == MESSAGES_RESPONSE["content"]

    assert seen == [CallTypes.anthropic_messages]
    assert len(messages_server.requests) == 1
    assert not messages_server.requests[0].headers.get("user-agent", "").startswith("python-httpx")
    assert messages_server.requests[0].body["messages"] == [{"role": "user", "content": "Reviewed prompt"}]
    assert "guardrails" not in messages_server.requests[0].body
    assert request_body(recorder.wait_for("log_pre_api_call")[0].kwargs) == messages_server.requests[0].body


@pytest.mark.asyncio
async def test_native_messages_post_call_guardrail_replaces_the_response_seen_by_caller_and_callback(
    messages_server: RecordingServer,
) -> None:
    seen: Final = []

    class Replace(CustomGuardrail):
        def __init__(self) -> None:
            super().__init__(
                guardrail_name="replace-messages", event_hook=GuardrailEventHooks.post_call, default_on=True
            )

        async def async_post_call_success_deployment_hook(
            self, request_data: dict[str, object], response: dict[str, object], call_type: CallTypes | None
        ) -> dict[str, object]:
            seen.append(call_type)
            return {**response, "content": [{"type": "text", "text": "Reviewed response"}]}

    guardrail: Final = Replace()
    recorder: Final = RecordingLogger()
    litellm.callbacks.append(guardrail)

    response: Final = await litellm.anthropic.messages.acreate(
        **arguments(messages_server, callbacks=[recorder], guardrails=[guardrail.guardrail_name])
    )
    success: Final = await recorder.wait_for_async("async_log_success_event")

    assert seen == [CallTypes.anthropic_messages]
    assert response["content"] == [{"type": "text", "text": "Reviewed response"}]
    assert len(success) == 1
    assert success[0].response.choices[0].message.content == "Reviewed response"


@pytest.mark.asyncio
async def test_native_messages_post_call_content_filter_blocks_provider_text(
    messages_server: RecordingServer,
) -> None:
    guardrail: Final = ContentFilterGuardrail(
        guardrail_name="block-messages-response",
        event_hook=GuardrailEventHooks.post_call,
        blocked_words=[BlockedWord(keyword="Hello from native Messages", action=ContentFilterAction.BLOCK)],
    )
    recorder: Final = RecordingLogger()
    litellm.callbacks.append(guardrail)

    with pytest.raises(HTTPException, match="Content blocked") as blocked:
        await litellm.anthropic.messages.acreate(
            **arguments(
                messages_server,
                callbacks=[recorder],
                guardrails=[guardrail.guardrail_name],
                user_api_key_request_route="/v1/messages",
            )
        )

    assert blocked.value.status_code == 400
    assert len(messages_server.requests) == 1
    assert "async_log_success_event" not in recorder.names
    failure: Final = await recorder.wait_for_async("async_log_failure_event")
    assert len(failure) == 1
    assert failure[0].kwargs["exception"] is blocked.value


@pytest.mark.asyncio
async def test_native_messages_pre_call_guardrail_can_block_before_provider_request(
    messages_server: RecordingServer,
) -> None:
    blocked: Final = ValueError("blocked prompt")

    class Block(CustomGuardrail):
        def __init__(self) -> None:
            super().__init__(guardrail_name="block-messages", event_hook=GuardrailEventHooks.pre_call, default_on=True)

        async def async_pre_call_deployment_hook(
            self, kwargs: dict[str, object], call_type: CallTypes | None
        ) -> dict[str, object]:
            raise blocked

    guardrail: Final = Block()
    recorder: Final = RecordingLogger()
    litellm.callbacks.append(guardrail)
    messages_server.expected_requests = 0

    with pytest.raises(ValueError, match="blocked prompt") as raised:
        await litellm.anthropic.messages.acreate(
            **arguments(messages_server, callbacks=[recorder], guardrails=[guardrail.guardrail_name])
        )

    assert raised.value is blocked
    assert messages_server.requests == []
    assert "log_pre_api_call" not in recorder.names
    assert "async_log_success_event" not in recorder.names
    failure: Final = await recorder.wait_for_async("async_log_failure_event")
    assert len(failure) == 1
    assert failure[0].kwargs["exception"] is blocked
