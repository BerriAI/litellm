from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import pytest
from e2e_config import unique_marker
from e2e_http import Result, UnknownApiError, unwrap
from e2e_metadata import Capability, Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import (
    ChatAssistantTurn,
    ChatAssistantTurnWithCot,
    ChatBody,
    ChatMessage,
    ChatMessageWithMetadata,
    ChatResponse,
    ChatTool,
    ChatToolFunction,
    ChatToolResultTurn,
    ChatTurn,
    LiteLLMParamsBody,
    ToolCall,
    ToolCallFunction,
)
from passthrough_client import PassthroughClient

pytestmark = pytest.mark.e2e

BACKEND: Final = "mistral/ministral-3b-latest"
TOOL_CALL_ID: Final = "D681PevKs"
WEATHER_TOOL: Final = ChatTool(
    function=ChatToolFunction(
        name="get_weather",
        description="Get the current weather for a city",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    )
)
SUBJECT: Final = Subject(
    domain=Domain.LLM_TRANSLATION,
    route=Route.CHAT_COMPLETIONS,
    providers=(Provider.MISTRAL,),
    models=(BACKEND,),
    mode=Mode.NONSTREAM,
)
TOOL_SUBJECT: Final = Subject(
    domain=Domain.LLM_TRANSLATION,
    route=Route.CHAT_COMPLETIONS,
    providers=(Provider.MISTRAL,),
    models=(BACKEND,),
    capabilities=(Capability.FUNCTION_CALLING,),
    mode=Mode.NONSTREAM,
)


def _register(client: PassthroughClient, resources: ResourceManager, *, drop_params: bool | None) -> tuple[str, str]:
    model: Final = f"e2e-mistral-message-fields-{unique_marker()}"
    model_id: Final = client.proxy.create_model(
        model, LiteLLMParamsBody(model=BACKEND, api_key="os.environ/MISTRAL_API_KEY", drop_params=drop_params)
    )
    resources.defer(lambda: client.proxy.delete_model(model_id))
    return model, resources.key()


def _copilot_conversation() -> tuple[ChatTurn, ...]:
    return (
        ChatMessage(role="user", content=f"Say hi. {unique_marker()}"),
        ChatAssistantTurnWithCot(content="Hi!", cot_id="cot-1f6b2c", cot_summary="Greet the user briefly."),
        ChatMessage(role="user", content="Now say bye in one word."),
    )


def _metadata_conversation() -> tuple[ChatTurn, ...]:
    return (
        ChatMessageWithMetadata(role="user", content=f"Say hi. {unique_marker()}", metadata={"source": "chat"}),
        ChatAssistantTurn(content="Hi!"),
        ChatMessageWithMetadata(role="user", content="Now say bye in one word.", metadata={"source": "chat"}),
    )


def _tool_conversation() -> tuple[ChatTurn, ...]:
    return (
        ChatMessage(role="user", content=f"What is the weather in Paris? {unique_marker()}"),
        ChatAssistantTurn(
            content="",
            tool_calls=[
                ToolCall(
                    id=TOOL_CALL_ID,
                    type="function",
                    function=ToolCallFunction(name="get_weather", arguments='{"city": "Paris"}'),
                )
            ],
        ),
        ChatToolResultTurn(tool_call_id=TOOL_CALL_ID, name="get_weather", content="22 degrees C and sunny"),
    )


def _chat(
    client: PassthroughClient,
    key: str,
    model: str,
    messages: Sequence[ChatTurn],
    *,
    drop_params: bool | None = None,
    tools: list[ChatTool] | None = None,
) -> Result[ChatResponse]:
    return client.proxy.chat(
        key, ChatBody(model=model, messages=messages, max_tokens=32, drop_params=drop_params, tools=tools)
    )


def _answer(result: Result[ChatResponse]) -> str:
    response: Final = unwrap(result)
    message: Final = response.choices[0].message if response.choices else None
    assert message is not None and message.content, f"Mistral returned no answer: {response}"
    return message.content


def _assert_rejected_before_mistral(result: Result[ChatResponse], rejected_fields: Sequence[str]) -> None:
    assert isinstance(result, UnknownApiError), f"expected the proxy to reject the request, got {result}"
    assert result.status_code == 400, f"expected 400 from the proxy, got {result.status_code}: {result.body}"
    missing: Final = tuple(field for field in (*rejected_fields, "drop_params") if field not in result.body)
    assert not missing, f"the error should name {missing}: {result.body}"


class TestMistralClientOnlyMessageFields:
    @meta(SUBJECT)
    def test_copilot_cot_fields_are_rejected_with_a_drop_params_hint_when_drop_params_is_off(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model, key = _register(client, resources, drop_params=None)
        _assert_rejected_before_mistral(_chat(client, key, model, _copilot_conversation()), ("cot_id", "cot_summary"))

    @meta(SUBJECT)
    def test_user_message_metadata_is_rejected_with_a_drop_params_hint_when_drop_params_is_off(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model, key = _register(client, resources, drop_params=None)
        _assert_rejected_before_mistral(_chat(client, key, model, _metadata_conversation()), ("metadata",))

    @meta(SUBJECT)
    def test_deployment_drop_params_lets_a_copilot_conversation_reach_mistral(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model, key = _register(client, resources, drop_params=True)
        assert _answer(_chat(client, key, model, _copilot_conversation()))

    @meta(SUBJECT)
    def test_request_drop_params_lets_user_message_metadata_reach_mistral(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model, key = _register(client, resources, drop_params=None)
        assert _answer(_chat(client, key, model, _metadata_conversation(), drop_params=True))

    @meta(TOOL_SUBJECT)
    def test_tool_round_trip_with_tool_name_still_reaches_mistral_when_drop_params_is_off(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model, key = _register(client, resources, drop_params=None)
        answer: Final = _answer(_chat(client, key, model, _tool_conversation(), tools=[WEATHER_TOOL]))
        assert "22" in answer, f"the answer ignored the tool result: {answer!r}"

    @meta(SUBJECT)
    def test_assistant_prefix_turn_still_reaches_mistral_when_drop_params_is_off(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model, key = _register(client, resources, drop_params=None)
        conversation: Final[tuple[ChatTurn, ...]] = (
            ChatMessage(role="user", content=f"Name the capital of France. {unique_marker()}"),
            ChatAssistantTurn(content="The capital of France is", prefix=True),
        )
        assert _answer(_chat(client, key, model, conversation))
