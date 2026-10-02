"""Verifiers score a finished program in [0, 1], off the critical path.

The contract is one coroutine. ORACLE ships three: any configured LiteLLM guardrail applied to the
program's final turn with its transcript as context (``llm_as_a_judge``, ``custom_code`` or a hosted
check), the score the client reported, and an import path for an operator's own class.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Protocol, runtime_checkable

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.llms.openai import AllMessageValues
from litellm.types.router import OracleVerifierConfig
from litellm.types.utils import GenericGuardrailAPIInputs, ModelResponse

from .decision import load_object

_EMPTY: Final[Mapping[str, object]] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class ProgramOutcome:
    """Everything a verifier may look at once a program has finished."""

    program_id: str
    model: str
    prompt: str
    response_text: str
    cost: float
    messages: Sequence[AllMessageValues] = ()
    score: float | None = None
    payload: Mapping[str, object] = field(default_factory=lambda: _EMPTY)


@runtime_checkable
class Verifier(Protocol):
    async def verify(self, outcome: ProgramOutcome) -> float: ...


def clamp_score(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def response_text(response: object) -> str:
    """The assistant text of a chat completion, or an empty string for anything else."""
    if not isinstance(response, ModelResponse) or not response.choices:
        return ""
    content: Final = response.choices[0].message.content
    return content if isinstance(content, str) else ""


class ReportedVerifier:
    """Trust the score the client reported with the program's last request or through the feedback endpoint."""

    def __init__(self, default: float = 0.0) -> None:
        self._default: Final[float] = default

    async def verify(self, outcome: ProgramOutcome) -> float:
        return clamp_score(outcome.score) if outcome.score is not None else self._default


def find_guardrail(name: str) -> CustomGuardrail | None:
    """The initialized guardrail registered under ``name``, if any."""
    callbacks: Final = litellm.logging_callback_manager.get_custom_loggers_for_type(callback_type=CustomGuardrail)
    guardrails: Final = tuple(callback for callback in callbacks if isinstance(callback, CustomGuardrail))
    return next((guardrail for guardrail in guardrails if guardrail.guardrail_name == name), None)


class GuardrailVerifier:
    """Apply a configured LiteLLM guardrail to the program's final turn: it passes or it is blocked.

    The guardrail sees the final assistant text as ``texts``, the program's transcript as
    ``structured_messages`` and the feedback payload as ``request_data["metadata"]``, so an
    ``llm_as_a_judge`` guardrail judges the whole conversation and a ``custom_code`` guardrail can read
    whatever the harness posted with the feedback.
    """

    def __init__(self, guardrail_name: str, lookup: Callable[[str], CustomGuardrail | None] = find_guardrail) -> None:
        self._name: Final[str] = guardrail_name
        self._lookup: Final = lookup

    async def verify(self, outcome: ProgramOutcome) -> float:
        guardrail: Final = self._lookup(self._name)
        if guardrail is None:
            raise LookupError(f"guardrail {self._name!r} is not initialized on this proxy")
        transcript: Final = list(outcome.messages) or [{"role": "user", "content": outcome.prompt}]
        inputs: Final = GenericGuardrailAPIInputs(
            texts=[outcome.response_text], model=outcome.model, structured_messages=transcript
        )
        try:
            await guardrail.apply_guardrail(  # pyright: ignore[reportUnknownMemberType]  # upstream types request_data as a bare dict
                inputs=inputs,
                request_data={
                    "model": outcome.model,
                    "metadata": {"program_id": outcome.program_id, **outcome.payload},
                },
                input_type="response",
            )
        except Exception:  # noqa: BLE001  # a guardrail signals "blocked" by raising its own exception type
            return 0.0
        return 1.0


def build_verifier(config: OracleVerifierConfig) -> Verifier:
    match config.type:
        case "reported":
            return ReportedVerifier(default=config.default_score)
        case "guardrail":
            return GuardrailVerifier(config.guardrail or "")
        case "custom":
            factory: Final = load_object(config.path or "")
            if not callable(factory):
                raise TypeError(f"oracle_router verifier.path {config.path!r} is not callable")
            instance: Final = factory()
            if not isinstance(instance, Verifier):
                raise TypeError(f"{config.path!r} must build an object with an async verify(outcome) method")
            return instance
