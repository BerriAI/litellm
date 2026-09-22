from collections.abc import AsyncIterator
from dataclasses import dataclass
from types import TracebackType
from typing import Final, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, ValidationError

from litellm.agent_sdk.types import (
    AgentMessage,
    AgentOptions,
    AssistantMessage,
    CompletionFailure,
    CompletionOutcome,
    CompletionSuccess,
    ConversationMessage,
    ResultMessage,
    StaticModelRouter,
    TextBlock,
    TurnContext,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionSystemMessage,
    ChatCompletionUserMessage,
)

_CompletionMessages: TypeAlias = list[AllMessageValues]  # mutable-ok: LiteLLM's public completion API requires a list


class CompletionProvider(Protocol):
    async def complete(
        self,
        *,
        model: str,
        system_prompt: str | None,
        messages: tuple[ConversationMessage, ...],
        max_tokens: int | None,
    ) -> CompletionOutcome: ...


def _system_message(content: str) -> ChatCompletionSystemMessage:
    message: Final[ChatCompletionSystemMessage] = {"role": "system", "content": content}
    return message


def _conversation_message(message: ConversationMessage) -> AllMessageValues:
    if message.role == "user":
        user_message: Final[ChatCompletionUserMessage] = {"role": "user", "content": message.content}
        return user_message
    assistant_message: Final[ChatCompletionAssistantMessage] = {"role": "assistant", "content": message.content}
    return assistant_message


class _ResponseMessage(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    content: str


class _ResponseChoice(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    message: _ResponseMessage


class _CompletionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    choices: tuple[_ResponseChoice, ...]


@dataclass(frozen=True, slots=True)
class LiteLLMCompletionProvider:
    async def complete(
        self,
        *,
        model: str,
        system_prompt: str | None,
        messages: tuple[ConversationMessage, ...],
        max_tokens: int | None,
    ) -> CompletionOutcome:
        import litellm

        system_messages: Final = () if system_prompt is None else (_system_message(system_prompt),)
        request_messages: Final[_CompletionMessages] = [  # mutable-ok: LiteLLM's public completion API requires a list
            *system_messages,
            *tuple(_conversation_message(message) for message in messages),
        ]
        try:
            response: Final = await litellm.acompletion(  # pyright: ignore[reportUnknownMemberType]  # legacy API has partially untyped parameters
                model=model,
                messages=request_messages,
                max_tokens=max_tokens,
                stream=False,
            )
        except Exception as exc:
            return CompletionFailure(error=str(exc))
        try:
            validated_response: Final = _CompletionResponse.model_validate(response)
        except ValidationError:
            return CompletionFailure(error="The model returned an unsupported response")
        if not validated_response.choices:
            return CompletionFailure(error="The model response did not contain text")
        return CompletionSuccess(content=validated_response.choices[0].message.content)


class AgentClient:
    def __init__(
        self,
        options: AgentOptions,
        completion_provider: CompletionProvider | None = None,
    ) -> None:
        self._options: Final = options
        self._completion_provider: Final = completion_provider or LiteLLMCompletionProvider()
        self._messages: tuple[ConversationMessage, ...] = ()

    async def __aenter__(self) -> "AgentClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    async def query(self, prompt: str) -> AsyncIterator[AgentMessage]:
        turn_number: Final = sum(message.role == "user" for message in self._messages) + 1
        context: Final = TurnContext(prompt=prompt, messages=self._messages, turn_number=turn_number)
        router: Final = self._options.model_router or StaticModelRouter(self._options.model)
        try:
            model: Final = await router.route(context)
        except Exception as exc:
            yield ResultMessage(result=str(exc), is_error=True, num_turns=0, model=None)
            return

        user_message: Final = ConversationMessage(role="user", content=prompt)
        request_messages: Final = (*self._messages, user_message)
        outcome: Final = await self._completion_provider.complete(
            model=model,
            system_prompt=self._options.system_prompt,
            messages=request_messages,
            max_tokens=self._options.max_tokens,
        )
        match outcome:
            case CompletionFailure(error=error):
                yield ResultMessage(result=error, is_error=True, num_turns=1, model=model)
            case CompletionSuccess(content=content):
                assistant_message: Final = ConversationMessage(role="assistant", content=content)
                self._messages = (*request_messages, assistant_message)
                yield AssistantMessage(content=(TextBlock(text=content),), model=model)
                yield ResultMessage(result=content, is_error=False, num_turns=1, model=model)


async def query(
    *,
    prompt: str,
    options: AgentOptions,
    completion_provider: CompletionProvider | None = None,
) -> AsyncIterator[AgentMessage]:
    async with AgentClient(options=options, completion_provider=completion_provider) as client:
        async for message in client.query(prompt):
            yield message
