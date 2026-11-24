"""Live /v1/chat/completions conversations on OpenAI models the proxy bridges to the
Responses API, where the caller changes reasoning effort and adds a tool mid-conversation.

The caller resends the full history every turn and appends each assistant message exactly
as the OpenAI SDK returned it. On GPT-6 that change must reach OpenAI as a mid-conversation
input item instead of a rewritten top-level setting, so the turn that changes settings still
reads the conversation so far from OpenAI's prompt cache. GPT-5 has no such item, so the
same conversation there must keep working with the new setting applied directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias, cast

import pytest
from e2e_config import unique_marker
from e2e_metadata import Capability, Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from llm_translation.sdk_clients import NO_PROXY_CACHE, SdkClients
from models import LiteLLMParamsBody
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam, ChatCompletionToolParam
from proxy_client import ProxyClient

pytestmark = [pytest.mark.e2e, pytest.mark.provider_live]

GPT6_MODEL: Final = "openai/gpt-6.2-sol"
GPT5_MODEL: Final = "openai/gpt-5.5"
PRIMING_ATTEMPTS: Final = 5
TOOL_ROUNDS: Final = 3
TOOL_RESULT: Final = "Sunny, 21C, local time 14:05."

Effort: TypeAlias = Literal["low", "high"]


def _function_tool(name: str, description: str) -> ChatCompletionToolParam:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }


WEATHER_TOOL: Final = _function_tool("get_weather", "Get the current weather for a city.")
TIME_TOOL: Final = _function_tool("get_local_time", "Get the current local time in a city.")


@dataclass(frozen=True, slots=True)
class Turn:
    question: str
    effort: Effort
    tools: tuple[ChatCompletionToolParam, ...]


@dataclass(frozen=True, slots=True)
class Reply:
    history: tuple[ChatCompletionMessageParam, ...]
    prompt_tokens: int
    cached_tokens: int


@dataclass(frozen=True, slots=True)
class Conversation:
    client: OpenAI
    model: str
    cache_key: str

    def complete(self, history: Sequence[ChatCompletionMessageParam], turn: Turn) -> ChatCompletion:
        return self.client.chat.completions.create(
            model=self.model,
            messages=list(history),
            tools=list(turn.tools),
            reasoning_effort=turn.effort,
            extra_body={**NO_PROXY_CACHE, "prompt_cache_key": self.cache_key},
        )

    def settle(
        self, history: tuple[ChatCompletionMessageParam, ...], turn: Turn, first: ChatCompletion | None, rounds: int
    ) -> Reply:
        completion: Final = self.complete(history, turn)
        measured: Final = first or completion
        message: Final = completion.choices[0].message
        answered: Final = (
            *history,
            cast(ChatCompletionMessageParam, message.model_dump()),  # cast-ok: the SDK user appends model_dump()
        )
        if not message.tool_calls:
            assert message.content, f"{self.model}: turn {turn.question!r} returned no answer: {completion!r}"
            usage: Final = measured.usage
            assert usage is not None, f"{self.model}: turn {turn.question!r} reported no usage: {measured!r}"
            details: Final = usage.prompt_tokens_details
            cached: Final = details.cached_tokens if details is not None and details.cached_tokens else 0
            return Reply(history=answered, prompt_tokens=usage.prompt_tokens, cached_tokens=cached)
        assert rounds < TOOL_ROUNDS, f"{self.model}: turn {turn.question!r} kept calling tools: {completion!r}"
        results: Final = tuple(
            cast(ChatCompletionMessageParam, {"role": "tool", "tool_call_id": call.id, "content": TOOL_RESULT})
            for call in message.tool_calls
        )
        return self.settle((*answered, *results), turn, measured, rounds + 1)

    def ask(self, history: tuple[ChatCompletionMessageParam, ...], turn: Turn) -> Reply:
        return self.settle((*history, {"role": "user", "content": turn.question}), turn, None, 0)


def _system_prompt(marker: str) -> str:
    clauses: Final = " ".join(
        f"Clause {index} of the travel desk handbook for session {marker}: confirm the traveller's city, "
        f"keep answers to one sentence, and never invent bookings or prices."
        for index in range(110)
    )
    return f"You are a terse travel desk assistant.\n{clauses}"


def _turns() -> tuple[Turn, ...]:
    return (
        Turn("In one word, which ocean borders Lisbon?", "low", (WEATHER_TOOL,)),
        Turn("In one word, which river runs through Paris?", "low", (WEATHER_TOOL,)),
        Turn("In one word, which country is Kyoto in?", "high", (WEATHER_TOOL, TIME_TOOL)),
        Turn("In one word, which continent is Peru on?", "high", (WEATHER_TOOL, TIME_TOOL)),
    )


def _primed_first_turn(conversation: Conversation, system: ChatCompletionMessageParam, turn: Turn) -> Reply:
    replies: Final = (conversation.ask((system,), turn) for _ in range(PRIMING_ATTEMPTS))
    warm: Final = next((reply for reply in replies if reply.cached_tokens > 0), None)
    assert warm is not None, f"{conversation.model}: the system prompt never read back from the prompt cache"
    return warm


def _converse(conversation: Conversation, turns: Sequence[Turn], first: Reply) -> tuple[Reply, ...]:
    if not turns:
        return (first,)
    return (first, *_converse(conversation, turns[1:], conversation.ask(first.history, turns[0])))


def _register(proxy: ProxyClient, resources: ResourceManager, backend: str) -> str:
    deployment: Final = f"e2e-chat-config-update-{unique_marker()}"
    deployment_id: Final = proxy.create_model(
        deployment, LiteLLMParamsBody(model=backend, api_key="os.environ/OPENAI_API_KEY"), provider_live=True
    )
    resources.defer(lambda: proxy.delete_model(deployment_id))
    return deployment


def _conversation(proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients, backend: str) -> Conversation:
    deployment: Final = _register(proxy, resources, backend)
    return Conversation(sdk.openai(resources.key(models=[deployment])), deployment, unique_marker())


def _run(conversation: Conversation, *, primed: bool) -> tuple[Reply, ...]:
    system: Final[ChatCompletionMessageParam] = {"role": "system", "content": _system_prompt(unique_marker())}
    turns: Final = _turns()
    first: Final = (
        _primed_first_turn(conversation, system, turns[0]) if primed else conversation.ask((system,), turns[0])
    )
    return _converse(conversation, turns[1:], first)


def _token_counts(replies: Sequence[Reply]) -> list[tuple[int, int]]:
    return [(reply.prompt_tokens, reply.cached_tokens) for reply in replies]


class TestChatResponsesBridgeConfigurationUpdate:
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.OPENAI,),
            models=(GPT6_MODEL,),
            capabilities=(Capability.REASONING, Capability.PROMPT_CACHING),
            mode=Mode.NONSTREAM,
        )
    )
    def test_gpt6_effort_change_and_added_tool_keep_reading_the_conversation_from_the_prompt_cache(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        replies: Final = _run(_conversation(proxy, resources, sdk, GPT6_MODEL), primed=True)
        _, second, changed, after = replies

        assert second.cached_tokens > 0, f"turn 2 read nothing from the prompt cache: {_token_counts(replies)}"
        assert changed.cached_tokens >= second.cached_tokens, (
            f"turn 3 changed reasoning effort and added a tool, and read {changed.cached_tokens} cached tokens "
            f"of its {changed.prompt_tokens}-token prompt, fewer than the {second.cached_tokens} turn 2 read "
            f"from its {second.prompt_tokens}-token prompt. The change rewrote the cached prefix. "
            f"(prompt, cached) per turn: {_token_counts(replies)}"
        )
        assert after.cached_tokens >= changed.cached_tokens, (
            f"turn 4 read {after.cached_tokens} cached tokens, fewer than the {changed.cached_tokens} turn 3 read, "
            f"so turn 3's prefix was not replayed as sent: {_token_counts(replies)}"
        )

    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.OPENAI,),
            models=(GPT5_MODEL,),
            capabilities=(Capability.REASONING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_gpt5_effort_change_and_added_tool_mid_conversation_still_answer(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        replies: Final = _run(_conversation(proxy, resources, sdk, GPT5_MODEL), primed=False)

        assert len(replies) == len(_turns()), _token_counts(replies)
