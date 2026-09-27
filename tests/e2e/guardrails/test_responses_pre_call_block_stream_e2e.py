from __future__ import annotations

import time
from collections.abc import Callable
from typing import Final

import pytest
from e2e_config import POLL_INTERVAL, POLL_TIMEOUT, unique_marker
from e2e_http import StreamingResponse
from guardrails_client import CustomCodeParamsBody, GuardrailsClient
from lifecycle import ResourceManager
from pydantic import BaseModel, TypeAdapter

pytestmark = pytest.mark.e2e

DENIAL: Final = "This model is not currently available. Please contact support if you think this is a mistake."

CUSTOM_CODE: Final = f'''
def apply_guardrail(inputs, request_data, input_type):
    return block("{DENIAL}")
'''


class _ContentPart(BaseModel):
    type: str
    text: str | None = None


class _OutputItem(BaseModel):
    type: str | None = None
    id: str | None = None
    role: str | None = None
    status: str | None = None
    content: list[_ContentPart] = []


class _Usage(BaseModel):
    total_tokens: int = 0


class _ResponseBody(BaseModel):
    output: list[_OutputItem] = []
    usage: _Usage | None = None


class _EventHead(BaseModel):
    type: str


class _CompletedEvent(BaseModel):
    type: str
    response: _ResponseBody


_EVENT_HEAD: Final = TypeAdapter(_EventHead)


def _denial_delivered(result: StreamingResponse) -> bool:
    if not result.ok:
        return False
    if DENIAL in result.body:
        return True
    return any(DENIAL in event for event in result.stream_events)


def _poll_terminal(result: StreamingResponse) -> bool:
    if _denial_delivered(result):
        return True
    if result.ok:
        return False
    return "Guardrail not found" not in result.body and result.status_code not in (-1, 401, 429)


def _poll_attempt(call: Callable[[], StreamingResponse], deadline: float) -> StreamingResponse:
    result: Final = call()
    if _poll_terminal(result) or time.monotonic() >= deadline:
        return result
    time.sleep(POLL_INTERVAL)
    return _poll_attempt(call, deadline)


def _poll_for_block(call: Callable[[], StreamingResponse]) -> StreamingResponse:
    return _poll_attempt(call, time.monotonic() + POLL_TIMEOUT)


def _assert_blocked_response(response: _ResponseBody) -> None:
    item = next(iter(response.output), None)
    assert item is not None, f"blocked response carried no output item: {response.output!r}"
    assert item.type == "message", f"output[0] must be a message item, got {item.type!r}: {item!r}"
    assert item.role == "assistant", f"output[0] role must be assistant, got {item.role!r}"
    assert item.status == "completed", f"output[0] status must be completed, got {item.status!r}"
    part = next(iter(item.content), None)
    assert part is not None, f"output[0] carried no content part: {item!r}"
    assert part.type == "output_text", f"content[0] must be output_text, got {part.type!r}"
    assert part.text == DENIAL, f"content[0] text must be the denial, got {part.text!r}"
    assert response.usage is not None and response.usage.total_tokens == 0, (
        f"a blocked response never reached a provider, usage must be zero: {response.usage!r}"
    )


class TestResponsesPreCallBlock:
    def _register_block(self, client: GuardrailsClient, resources: ResourceManager) -> str:
        name: Final = f"e2e-custom-code-responses-block-{unique_marker()}"
        guardrail_id: Final = client.register(
            name,
            CustomCodeParamsBody(mode="pre_call", default_on=False, custom_code=CUSTOM_CODE),
        )
        resources.defer(lambda: client.delete_guardrail(guardrail_id))
        return name

    @pytest.mark.covers("guardrail.custom_code.pre_call.blocks", exercised_on=["responses"])
    def test_stream_block_is_sse_with_completed_assistant_message(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        name: Final = self._register_block(client, resources)
        model: Final = client.create_backend_model(
            resources, prefix="e2e-responses-block", backend="openai/gpt-4.1-mini", api_key="os.environ/OPENAI_API_KEY"
        )

        result: Final = _poll_for_block(
            lambda: client.responses_stream_raw(scoped_key, model, "say hi", guardrails=[name])
        )

        assert result.status_code == 200, f"a pre_call block answers 200, got {result.status_code}: {result.body[:400]}"
        assert (result.content_type or "").startswith("text/event-stream"), (
            f"stream=true must answer SSE, got content-type {result.content_type!r}: {result.body[:400]}"
        )
        events: Final = tuple(_EVENT_HEAD.validate_json(payload).type for payload in result.stream_events)
        completed: Final = tuple(
            _CompletedEvent.model_validate_json(payload)
            for payload, event_type in zip(result.stream_events, events)
            if event_type == "response.completed"
        )
        assert len(completed) == 1, (
            f"the denial stream must end in exactly one response.completed event, got events {events!r}"
        )
        _assert_blocked_response(completed[0].response)

    @pytest.mark.covers("guardrail.custom_code.pre_call.blocks", exercised_on=["responses"])
    def test_non_stream_block_is_schema_valid_json(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        name: Final = self._register_block(client, resources)
        model: Final = client.create_backend_model(
            resources, prefix="e2e-responses-block", backend="openai/gpt-4.1-mini", api_key="os.environ/OPENAI_API_KEY"
        )

        result: Final = _poll_for_block(lambda: client.responses(scoped_key, model, "say hi", guardrails=[name]))

        assert result.status_code == 200, f"a pre_call block answers 200, got {result.status_code}: {result.body[:400]}"
        assert (result.content_type or "").startswith("application/json"), (
            f"a non-streaming block answers JSON, got content-type {result.content_type!r}"
        )
        _assert_blocked_response(_ResponseBody.model_validate_json(result.body))
