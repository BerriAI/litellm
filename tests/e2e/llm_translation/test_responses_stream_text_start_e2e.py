"""Live e2e: /v1/responses streaming emits a text-start before every text delta.

github issue #27671: on multi-step tool conversations, the text the model streams
after a tool round arrived as `response.output_text.delta` events whose item had
never been announced - no `response.output_item.added` / `response.content_part.added`
carried that item id first - so OpenAI-SDK parsers died with "text part
chatcmpl-<id> not found". The customer hit it through OpenCode against bedrock
anthropic; the defect lives in the shared completion-to-responses streaming
iterator every non-native-responses provider goes through, which is what this
test drives (gemini here, same translation layer).

The request replays a completed tool round (function_call + function_call_output
in the input) so the streamed text is exactly the post-tool "step 2" text the
issue describes.
"""

from __future__ import annotations

from typing import Final, Literal

import pytest
from pydantic import BaseModel

from e2e_config import unique_marker
from e2e_http import StreamingResponse
from endpoints_client import FunctionParameterProperty, FunctionParameters, ResponsesFunctionTool
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient

pytestmark = pytest.mark.e2e

BACKEND_MODEL = "gemini/gemini-2.5-flash"

_WEATHER_TOOL = ResponsesFunctionTool(
    name="get_weather",
    description="Get the current weather for a location",
    parameters=FunctionParameters(
        properties={"location": FunctionParameterProperty(type="string")},
        required=["location"],
    ),
)


class _UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str


class _FunctionCall(BaseModel):
    type: Literal["function_call"] = "function_call"
    call_id: str
    name: str
    arguments: str


class _FunctionCallOutput(BaseModel):
    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str


class _ToolRoundRequest(BaseModel):
    model: str
    stream: bool = True
    tools: list[ResponsesFunctionTool]
    input: list[_UserMessage | _FunctionCall | _FunctionCallOutput]


class _EventItem(BaseModel):
    id: str | None = None


class _StreamEvent(BaseModel):
    type: str
    item_id: str | None = None
    item: _EventItem | None = None

    def announced_item_id(self) -> str | None:
        """The item id this event announces or targets, wherever the event
        carries it (delta events use item_id; added/done events nest it)."""
        if self.item_id is not None:
            return self.item_id
        return self.item.id if self.item else None


_ANNOUNCE_TYPES: Final = ("response.output_item.added", "response.content_part.added")


def _announced_before(events: list[_StreamEvent], index: int) -> frozenset[str]:
    """Item ids announced by an output_item.added / content_part.added event
    strictly before `index`: the ids a text delta at `index` is allowed to target."""
    return frozenset(
        item_id
        for event in events[:index]
        if event.type in _ANNOUNCE_TYPES and (item_id := event.announced_item_id()) is not None
    )


def _stream_post_tool_text(proxy: ProxyClient, key: str, model: str) -> StreamingResponse:
    call_id = f"call_{unique_marker()}"
    return proxy.transport.stream(
        "/v1/responses",
        headers=proxy.transport.bearer(key),
        json=_ToolRoundRequest(
            model=model,
            tools=[_WEATHER_TOOL],
            input=[
                _UserMessage(content="What is the weather in Paris?"),
                _FunctionCall(call_id=call_id, name="get_weather", arguments='{"location": "Paris"}'),
                _FunctionCallOutput(call_id=call_id, output="22C and sunny"),
            ],
        ),
    )


class TestResponsesStreamTextStart:
    @pytest.mark.covers(
        "llm.responses.gemini.tool_use.stream.text_start_precedes_deltas",
        exercised_on=["responses"],
    )
    def test_post_tool_text_deltas_follow_a_text_start_with_same_item_id(
        self, proxy: ProxyClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        model = f"e2e-text-start-{unique_marker()}"
        model_id = proxy.create_model(
            model, LiteLLMParamsBody(model=BACKEND_MODEL, api_key="os.environ/GEMINI_API_KEY")
        )
        resources.defer(lambda: proxy.delete_model(model_id))

        result = _stream_post_tool_text(proxy, scoped_key, model)
        assert result.ok and result.is_streaming, f"responses stream was not established: {result}"
        assert result.stream_error is None, f"stream carried an error event: {result.stream_error}"

        events = [_StreamEvent.model_validate_json(payload) for payload in result.stream_events]
        deltas = [event for event in events if event.type == "response.output_text.delta"]
        assert deltas, f"the post-tool round streamed no text deltas: {[e.type for e in events]}"

        for index, event in enumerate(events):
            if event.type != "response.output_text.delta":
                continue
            item_id = event.announced_item_id()
            assert item_id in _announced_before(events, index), (
                f"text delta for item {item_id!r} arrived before any output_item.added/"
                f"content_part.added announced it; OpenAI-SDK parsers fail with "
                f"'text part ... not found' (#27671). Events so far: "
                f"{[e.type for e in events[: index + 1]]}"
            )
        assert any(event.type == "response.completed" for event in events), (
            f"stream never completed: {[e.type for e in events][-5:]}"
        )
