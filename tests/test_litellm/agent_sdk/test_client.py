from typing import Final

import pytest

from litellm.agent_sdk import (
    AgentClient,
    AgentOptions,
    AssistantMessage,
    CompletionProvider,
    ResultMessage,
    TextBlock,
    TurnContext,
    query,
)
from litellm.agent_sdk.types import CompletionFailure, CompletionOutcome, CompletionSuccess, ConversationMessage


class RecordingRouter:
    def __init__(self) -> None:
        self.contexts: tuple[TurnContext, ...] = ()

    async def route(self, context: TurnContext) -> str:
        self.contexts = (*self.contexts, context)
        return "openai/gpt-5.4-mini" if context.turn_number == 1 else "anthropic/claude-opus-4-8"


class RecordingProvider(CompletionProvider):
    def __init__(self, outcomes: tuple[CompletionOutcome, ...]) -> None:
        self.outcomes: Final = outcomes
        self.calls: tuple[tuple[str, tuple[ConversationMessage, ...]], ...] = ()

    async def complete(
        self,
        *,
        model: str,
        system_prompt: str | None,
        messages: tuple[ConversationMessage, ...],
        max_tokens: int | None,
    ) -> CompletionOutcome:
        call_index: Final = len(self.calls)
        self.calls = (*self.calls, (model, messages))
        return self.outcomes[call_index]


@pytest.mark.asyncio
async def test_client_routes_each_turn_and_preserves_conversation() -> None:
    router: Final = RecordingRouter()
    provider: Final = RecordingProvider(
        (CompletionSuccess(content="First answer"), CompletionSuccess(content="Second answer"))
    )
    client: Final = AgentClient(
        AgentOptions(model="fallback/model", model_router=router, system_prompt="Be useful"),
        completion_provider=provider,
    )

    first_messages: Final = tuple([message async for message in client.query("First question")])
    second_messages: Final = tuple([message async for message in client.query("Second question")])

    assert first_messages == (
        AssistantMessage(content=(TextBlock(text="First answer"),), model="openai/gpt-5.4-mini"),
        ResultMessage(result="First answer", is_error=False, num_turns=1, model="openai/gpt-5.4-mini"),
    )
    assert isinstance(second_messages[0], AssistantMessage)
    assert second_messages[0].model == "anthropic/claude-opus-4-8"
    assert tuple(context.turn_number for context in router.contexts) == (1, 2)
    assert provider.calls[1][1] == (
        ConversationMessage(role="user", content="First question"),
        ConversationMessage(role="assistant", content="First answer"),
        ConversationMessage(role="user", content="Second question"),
    )


@pytest.mark.asyncio
async def test_top_level_query_returns_failure_as_result_message() -> None:
    provider: Final = RecordingProvider((CompletionFailure(error="provider unavailable"),))

    messages: Final = tuple(
        [
            message
            async for message in query(
                prompt="Review this",
                options=AgentOptions(model="gemini/gemini-2.5-flash"),
                completion_provider=provider,
            )
        ]
    )

    assert messages == (
        ResultMessage(
            result="provider unavailable",
            is_error=True,
            num_turns=1,
            model="gemini/gemini-2.5-flash",
        ),
    )
