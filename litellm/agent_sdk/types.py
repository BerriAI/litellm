from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class TurnContext:
    prompt: str
    messages: tuple[ConversationMessage, ...]
    turn_number: int


@dataclass(frozen=True, slots=True)
class AgentOptions:
    model: str
    system_prompt: str | None = None
    max_tokens: int | None = None
    model_router: "ModelRouter | None" = None


@dataclass(frozen=True, slots=True)
class TextBlock:
    text: str


@dataclass(frozen=True, slots=True)
class AssistantMessage:
    content: tuple[TextBlock, ...]
    model: str


@dataclass(frozen=True, slots=True)
class ResultMessage:
    result: str
    is_error: bool
    num_turns: int
    model: str | None


AgentMessage: TypeAlias = AssistantMessage | ResultMessage


class ModelRouter(Protocol):
    async def route(self, context: TurnContext) -> str: ...


@dataclass(frozen=True, slots=True)
class StaticModelRouter(ModelRouter):
    model: str

    async def route(self, context: TurnContext) -> str:
        return self.model


@dataclass(frozen=True, slots=True)
class CompletionSuccess:
    content: str


@dataclass(frozen=True, slots=True)
class CompletionFailure:
    error: str


CompletionOutcome: TypeAlias = CompletionSuccess | CompletionFailure
