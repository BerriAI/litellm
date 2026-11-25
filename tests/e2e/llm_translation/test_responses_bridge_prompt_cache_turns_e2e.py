from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

import pytest
from _pytest.mark.structures import ParameterSet
from anthropic import Anthropic
from anthropic.types import Message, MessageParam, ToolParam
from e2e_config import PROMPT_CACHE_OPENAI_MODEL, unique_marker
from e2e_metadata import Capability, Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from openai import OpenAI, omit
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from openai.types.completion_usage import CompletionUsage
from openai.types.responses import ResponseInputParam
from openai.types.responses import ToolParam as ResponsesToolParam
from proxy_client import ProxyClient
from sdk_clients import NO_PROXY_CACHE, SdkClients

pytestmark = [pytest.mark.e2e, pytest.mark.provider_live]

CONVERSATION_ATTEMPTS: Final = 3
SHARE_OF_DIRECT_CACHE: Final = 0.9
SHARE_OF_FIRST_PROMPT: Final = 0.9
RULES: Final = 160
QUESTIONS: Final = (
    "In one short sentence, what does rule 3 ask for?",
    "In one short sentence, does rule 7 ask for the same thing?",
    "In one short sentence, how many rules are there?",
)
STORES: Final = (
    pytest.param(None, id="store_unset"),
    pytest.param(True, id="store_true"),
    pytest.param(False, id="store_false"),
)
TOOL_NAME: Final = "get_weather"
TOOL_DESCRIPTION: Final = "Get the current weather for a city."
TOOL_PARAMETERS: Final[dict[str, object]] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}

Surface: TypeAlias = Literal["chat_completions", "messages"]


@dataclass(frozen=True, slots=True)
class Backend:
    id: str
    registry_route: str
    provider: Provider
    litellm_params: LiteLLMParamsBody
    passthrough_model: str | None = None
    skip_reason: str | None = None


BACKENDS: Final = (
    Backend(
        "openai",
        "openai",
        Provider.OPENAI,
        LiteLLMParamsBody(model=f"openai/{PROMPT_CACHE_OPENAI_MODEL}", api_key="os.environ/OPENAI_API_KEY"),
        passthrough_model=PROMPT_CACHE_OPENAI_MODEL,
    ),
    Backend(
        "bedrock_mantle",
        "bedrock_mantle",
        Provider.BEDROCK_MANTLE,
        LiteLLMParamsBody(model="bedrock_mantle/openai.gpt-6.2-sol", aws_region_name="us-east-1"),
        skip_reason="Stage IAM lacks aws-marketplace:Subscribe for Mantle openai.gpt-6.2-sol",
    ),
)


def _backends(surface: Surface, mode: Mode) -> tuple[ParameterSet, ...]:
    return tuple(
        pytest.param(
            backend,
            id=backend.id,
            marks=(
                pytest.mark.covers(
                    f"llm.{surface}.{backend.registry_route}.prompt_cache_5m."
                    f"{mode.value}.system_prompt_cached_every_turn",
                    exercised_on=[surface],
                ),
                meta(
                    Subject(
                        domain=Domain.LLM_TRANSLATION,
                        route=Route(surface),
                        providers=(backend.provider,),
                        models=(backend.litellm_params.model,),
                        capabilities=(Capability.PROMPT_CACHING,),
                        mode=mode,
                    )
                ),
                *((pytest.mark.skip(reason=backend.skip_reason),) if backend.skip_reason else ()),
            ),
        )
        for backend in BACKENDS
    )


@dataclass(frozen=True, slots=True)
class Turn:
    text: str
    input_tokens: int
    cached_tokens: int


TurnSender: TypeAlias = Callable[[str, tuple[str, ...]], Turn]


def _system_prompt(marker: str) -> str:
    rules: Final = " ".join(
        f"Rule {index} is to answer in one short sentence without calling a tool." for index in range(RULES)
    )
    return f"Session {marker}. {rules}"


def _history(answers: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    return tuple(zip(QUESTIONS, answers, strict=False))


def _chat_messages(prompt: str, answers: tuple[str, ...]) -> list[ChatCompletionMessageParam]:
    earlier: Final = itertools.chain.from_iterable(
        _chat_pair(question, answer) for question, answer in _history(answers)
    )
    return [{"role": "system", "content": prompt}, *earlier, {"role": "user", "content": QUESTIONS[len(answers)]}]


def _chat_pair(question: str, answer: str) -> tuple[ChatCompletionMessageParam, ChatCompletionMessageParam]:
    return {"role": "user", "content": question}, {"role": "assistant", "content": answer}


def _messages_history(answers: tuple[str, ...]) -> list[MessageParam]:
    earlier: Final = itertools.chain.from_iterable(
        _messages_pair(question, answer) for question, answer in _history(answers)
    )
    return [*earlier, {"role": "user", "content": QUESTIONS[len(answers)]}]


def _messages_pair(question: str, answer: str) -> tuple[MessageParam, MessageParam]:
    return {"role": "user", "content": question}, {"role": "assistant", "content": answer}


def _responses_pair(question: str, answer: str) -> ResponseInputParam:
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


def _responses_input(prompt: str, answers: tuple[str, ...]) -> ResponseInputParam:
    earlier: Final = itertools.chain.from_iterable(
        _responses_pair(question, answer) for question, answer in _history(answers)
    )
    return [
        {"role": "developer", "content": prompt},
        *earlier,
        {"role": "user", "content": QUESTIONS[len(answers)]},
    ]


def _chat_tools() -> list[ChatCompletionToolParam]:
    return [
        {
            "type": "function",
            "function": {"name": TOOL_NAME, "description": TOOL_DESCRIPTION, "parameters": TOOL_PARAMETERS},
        }
    ]


def _messages_tools() -> list[ToolParam]:
    return [{"name": TOOL_NAME, "description": TOOL_DESCRIPTION, "input_schema": TOOL_PARAMETERS}]


def _responses_tools() -> list[ResponsesToolParam]:
    return [
        {
            "type": "function",
            "name": TOOL_NAME,
            "description": TOOL_DESCRIPTION,
            "parameters": TOOL_PARAMETERS,
            "strict": False,
        }
    ]


def _chat_cached_tokens(usage: CompletionUsage | None) -> int:
    details: Final = usage.prompt_tokens_details if usage is not None else None
    return (details.cached_tokens or 0) if details is not None else 0


def _chat_turn(client: OpenAI, model: str, store: bool | None, messages: list[ChatCompletionMessageParam]) -> Turn:
    completion: Final = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=_chat_tools(),
        tool_choice="none",
        reasoning_effort="low",
        store=omit if store is None else store,
        extra_body=dict(NO_PROXY_CACHE),
    )
    return Turn(
        text=completion.choices[0].message.content or "",
        input_tokens=completion.usage.prompt_tokens if completion.usage is not None else 0,
        cached_tokens=_chat_cached_tokens(completion.usage),
    )


def _streamed_chat_turn(
    client: OpenAI, model: str, store: bool | None, messages: list[ChatCompletionMessageParam]
) -> Turn:
    chunks: Final = tuple(
        client.chat.completions.create(
            model=model,
            messages=messages,
            tools=_chat_tools(),
            tool_choice="none",
            reasoning_effort="low",
            store=omit if store is None else store,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=dict(NO_PROXY_CACHE),
        )
    )
    choices: Final = itertools.chain.from_iterable(chunk.choices for chunk in chunks)
    usages: Final = tuple(chunk.usage for chunk in chunks if chunk.usage is not None)
    usage: Final = usages[-1] if usages else None
    return Turn(
        text="".join(choice.delta.content or "" for choice in choices),
        input_tokens=usage.prompt_tokens if usage is not None else 0,
        cached_tokens=_chat_cached_tokens(usage),
    )


def _chat_sender(client: OpenAI, model: str, store: bool | None, *, stream: bool) -> TurnSender:
    turn: Final = _streamed_chat_turn if stream else _chat_turn

    def send(prompt: str, answers: tuple[str, ...]) -> Turn:
        return turn(client, model, store, _chat_messages(prompt, answers))

    return send


def _messages_reply(
    client: Anthropic, model: str, extra_body: dict[str, object], prompt: str, messages: list[MessageParam]
) -> Message:
    return client.messages.create(
        model=model,
        max_tokens=256,
        system=prompt,
        messages=messages,
        tools=_messages_tools(),
        tool_choice={"type": "none"},
        extra_body=extra_body,
    )


def _streamed_messages_reply(
    client: Anthropic, model: str, extra_body: dict[str, object], prompt: str, messages: list[MessageParam]
) -> Message:
    with client.messages.stream(
        model=model,
        max_tokens=256,
        system=prompt,
        messages=messages,
        tools=_messages_tools(),
        tool_choice={"type": "none"},
        extra_body=extra_body,
    ) as events:
        return events.get_final_message()


def _messages_sender(sdk: SdkClients, key: str, model: str, store: bool | None, *, stream: bool) -> TurnSender:
    client: Final = sdk.anthropic(key)
    extra_body: Final[dict[str, object]] = {**NO_PROXY_CACHE, **({} if store is None else {"store": store})}
    reply: Final = _streamed_messages_reply if stream else _messages_reply

    def send(prompt: str, answers: tuple[str, ...]) -> Turn:
        message: Final = reply(client, model, extra_body, prompt, _messages_history(answers))
        usage: Final = message.usage
        cached: Final = usage.cache_read_input_tokens or 0
        return Turn(
            text="".join(block.text for block in message.content if block.type == "text"),
            input_tokens=usage.input_tokens + (usage.cache_creation_input_tokens or 0) + cached,
            cached_tokens=cached,
        )

    return send


def _direct_sender(sdk: SdkClients, key: str, backend: Backend, model: str, store: bool | None) -> TurnSender:
    route: Final = "/openai_passthrough/v1" if backend.passthrough_model else "/v1"
    extra_body: Final[dict[str, object]] = {} if backend.passthrough_model else dict(NO_PROXY_CACHE)
    client: Final = OpenAI(base_url=f"{sdk.base_url}{route}", api_key=key, timeout=sdk.request_timeout, max_retries=0)

    def send(prompt: str, answers: tuple[str, ...]) -> Turn:
        response: Final = client.responses.create(
            model=backend.passthrough_model or model,
            input=_responses_input(prompt, answers),
            tools=_responses_tools(),
            tool_choice="none",
            reasoning={"effort": "low"},
            store=omit if store is None else store,
            extra_body=extra_body,
        )
        usage: Final = response.usage
        return Turn(
            text=response.output_text,
            input_tokens=usage.input_tokens if usage is not None else 0,
            cached_tokens=usage.input_tokens_details.cached_tokens if usage is not None else 0,
        )

    return send


def _continue(send: TurnSender, prompt: str, turns: tuple[Turn, ...]) -> tuple[Turn, ...]:
    if len(turns) == len(QUESTIONS):
        return turns
    return _continue(send, prompt, (*turns, send(prompt, tuple(turn.text for turn in turns))))


def _converse_until(send: TurnSender, accepts: Callable[[tuple[Turn, ...]], bool], attempts: int) -> tuple[Turn, ...]:
    turns: Final = _continue(send, _system_prompt(unique_marker()), ())
    if accepts(turns) or attempts <= 1:
        return turns
    return _converse_until(send, accepts, attempts - 1)


def _continued(turns: tuple[Turn, ...]) -> tuple[Turn, ...]:
    return turns[1:]


def _direct_caches_the_first_prompt_on_every_continued_turn(turns: tuple[Turn, ...]) -> bool:
    floor: Final = SHARE_OF_FIRST_PROMPT * turns[0].input_tokens
    return turns[0].input_tokens > 0 and all(turn.cached_tokens >= floor for turn in _continued(turns))


def _matches_direct(direct: tuple[Turn, ...]) -> Callable[[tuple[Turn, ...]], bool]:
    def accepts(turns: tuple[Turn, ...]) -> bool:
        return all(
            bridged.cached_tokens >= SHARE_OF_DIRECT_CACHE * yardstick.cached_tokens
            for bridged, yardstick in zip(_continued(turns), _continued(direct), strict=True)
        )

    return accepts


def _register(proxy: ProxyClient, resources: ResourceManager, backend: Backend) -> str:
    model: Final = f"e2e-prompt-cache-turns-{backend.id}-{unique_marker()}"
    model_id: Final = proxy.create_model(model, backend.litellm_params)
    resources.defer(lambda: proxy.delete_model(model_id))
    return model


def _assert_bridged_conversation_caches_like_direct(
    sdk: SdkClients, key: str, backend: Backend, model: str, store: bool | None, bridged: TurnSender
) -> None:
    direct: Final = _converse_until(
        _direct_sender(sdk, key, backend, model, store),
        _direct_caches_the_first_prompt_on_every_continued_turn,
        CONVERSATION_ATTEMPTS,
    )
    assert _direct_caches_the_first_prompt_on_every_continued_turn(direct), (
        f"{backend.id} cached less than {SHARE_OF_FIRST_PROMPT:.0%} of the first turn's "
        f"{direct[0].input_tokens} input tokens on a continued turn of the conversation sent as Responses input, "
        f"so it is no yardstick: {direct}"
    )

    turns: Final = _converse_until(bridged, _matches_direct(direct), CONVERSATION_ATTEMPTS)

    assert all(turn.text for turn in turns), turns
    assert _matches_direct(direct)(turns), (
        f"continued turns bridged to {backend.id} cached {[turn.cached_tokens for turn in _continued(turns)]} "
        f"tokens, the same conversation sent as Responses input cached "
        f"{[turn.cached_tokens for turn in _continued(direct)]}"
    )


class TestBridgedConversationPromptCache:
    @pytest.mark.parametrize("backend", _backends("chat_completions", Mode.NONSTREAM))
    @pytest.mark.parametrize("store", STORES)
    def test_chat_conversation_caches_the_system_prompt_on_every_continued_turn(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients, backend: Backend, store: bool | None
    ) -> None:
        key: Final = resources.key()
        model: Final = _register(proxy, resources, backend)
        _assert_bridged_conversation_caches_like_direct(
            sdk, key, backend, model, store, _chat_sender(sdk.openai(key), model, store, stream=False)
        )

    @pytest.mark.parametrize("backend", _backends("chat_completions", Mode.STREAM))
    @pytest.mark.parametrize("store", STORES)
    def test_streamed_chat_conversation_caches_the_system_prompt_on_every_continued_turn(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients, backend: Backend, store: bool | None
    ) -> None:
        key: Final = resources.key()
        model: Final = _register(proxy, resources, backend)
        _assert_bridged_conversation_caches_like_direct(
            sdk, key, backend, model, store, _chat_sender(sdk.openai(key), model, store, stream=True)
        )

    @pytest.mark.parametrize("backend", _backends("messages", Mode.NONSTREAM))
    @pytest.mark.parametrize("store", STORES)
    def test_messages_conversation_caches_the_system_prompt_on_every_continued_turn(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients, backend: Backend, store: bool | None
    ) -> None:
        key: Final = resources.key()
        model: Final = _register(proxy, resources, backend)
        _assert_bridged_conversation_caches_like_direct(
            sdk, key, backend, model, store, _messages_sender(sdk, key, model, store, stream=False)
        )

    @pytest.mark.parametrize("backend", _backends("messages", Mode.STREAM))
    @pytest.mark.parametrize("store", STORES)
    def test_streamed_messages_conversation_caches_the_system_prompt_on_every_continued_turn(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients, backend: Backend, store: bool | None
    ) -> None:
        key: Final = resources.key()
        model: Final = _register(proxy, resources, backend)
        _assert_bridged_conversation_caches_like_direct(
            sdk, key, backend, model, store, _messages_sender(sdk, key, model, store, stream=True)
        )
