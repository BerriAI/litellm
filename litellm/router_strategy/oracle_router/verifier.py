"""Verifiers score a finished program in [0, 1], off the critical path.

The contract is one coroutine. ORACLE ships three: any configured LiteLLM guardrail applied to the
program's final turn with its transcript as context (``llm_as_a_judge``, ``custom_code`` or a hosted
check), the score the client reported, and an import path for an operator's own class.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Protocol, runtime_checkable

from pydantic import TypeAdapter, ValidationError

import litellm
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.llms.openai import AllMessageValues
from litellm.types.router import OracleVerifierConfig
from litellm.types.utils import GenericGuardrailAPIInputs, ModelResponse

from .decision import load_object

_EMPTY: Final[Mapping[str, object]] = MappingProxyType({})
_STR_MAPPING: Final = TypeAdapter(Mapping[str, object])
_STATUS_RECORDS: Final = TypeAdapter(Sequence[Mapping[str, object]])
_STATUS_RECORD_KEY: Final[str] = "standard_logging_guardrail_information"
_BLOCKED_STATUSES: Final[frozenset[str]] = frozenset({"guardrail_intervened", "guardrail_flagged"})
_UNVERIFIED_STATUSES: Final[frozenset[str]] = frozenset({"guardrail_failed_to_respond", "not_run"})


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
    """Clamp a score to [0, 1]. A NaN/inf score raises: it would corrupt a bandit cell for good."""
    score: Final = float(value)
    if not math.isfinite(score):
        raise ValueError(f"score must be finite, got {score!r}")
    return min(max(score, 0.0), 1.0)


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


def _verifier_metadata(outcome: ProgramOutcome) -> dict[str, object]:  # mutable-ok: the guardrail writes into it
    """The feedback payload as the guardrail's request metadata.

    The verdict is read from the status record the guardrail writes here, so the caller's payload must not
    pre-seed that key (a non-record entry would make the whole list unreadable and hide a rejection), and
    it cannot rename the program.
    """
    return {
        **{key: value for key, value in outcome.payload.items() if key != _STATUS_RECORD_KEY},
        "program_id": outcome.program_id,
    }


def _status_records(bucket: object) -> Sequence[Mapping[str, object]]:
    """The ``standard_logging_guardrail_information`` records of one metadata bucket, or none."""
    try:
        return _STATUS_RECORDS.validate_python(_STR_MAPPING.validate_python(bucket).get(_STATUS_RECORD_KEY) or ())
    except ValidationError:
        return ()


def recorded_status(request_data: Mapping[str, object], guardrail_name: str) -> str | None:
    """The last status ``guardrail_name`` recorded for itself in ``request_data`` while it ran, if any.

    Every first-party guardrail writes a ``standard_logging_guardrail_information`` record into the
    request's metadata bucket (``success``, ``guardrail_intervened``, ``guardrail_failed_to_respond``,
    ...), which is the only way to tell a judge that failed the answer from one that failed to answer.
    """
    records: Final = (
        *_status_records(request_data.get("metadata")),
        *_status_records(request_data.get("litellm_metadata")),
    )
    statuses: Final = tuple(
        str(record["guardrail_status"])
        for record in records
        if record.get("guardrail_name") in (guardrail_name, None) and record.get("guardrail_status")
    )
    return statuses[-1] if statuses else None


async def _apply(
    guardrail: CustomGuardrail,
    inputs: GenericGuardrailAPIInputs,
    request_data: dict[str, object],  # mutable-ok: the guardrail writes its status into it
) -> Exception | None:
    """Run the guardrail; the exception it raised to signal its verdict, or ``None`` when it returned."""
    try:
        await guardrail.apply_guardrail(  # pyright: ignore[reportUnknownMemberType]  # upstream types request_data as a bare dict
            inputs=inputs, request_data=request_data, input_type="response"
        )
    except Exception as error:  # noqa: BLE001  # a guardrail may signal its verdict by raising; the caller classifies it
        return error
    return None


def _is_http_block(error: Exception) -> bool:
    """A guardrail without a status record that blocks by raising FastAPI's ``HTTPException``."""
    try:
        from fastapi import HTTPException
    except ImportError:  # the SDK without the proxy extra: nothing can raise it
        return False
    return isinstance(error, HTTPException)


class GuardrailVerifier:
    """Apply a configured LiteLLM guardrail to the program's final turn: it passes or it is blocked.

    The guardrail sees the final assistant text as ``texts``, the program's transcript as
    ``structured_messages`` and the feedback payload as ``request_data["metadata"]``, so an
    ``llm_as_a_judge`` guardrail judges the whole conversation and a ``custom_code`` guardrail can read
    whatever the harness posted with the feedback.

    The verdict is read from the status the guardrail records for itself (so a judge configured with
    ``on_failure: log`` still scores 0 when the answer fails), and falls back to how it raised: a
    ``GuardrailRaisedException`` with ``blocked_content`` or an ``HTTPException`` is a block. A
    guardrail that could not evaluate the program (``guardrail_failed_to_respond``, an unreachable
    backend) raises, so the program counts as a failed verification and teaches the decision maker
    nothing instead of passing it.
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
        request_data: Final[dict[str, object]] = {  # mutable-ok: the guardrail records its status into it
            "model": outcome.model,
            "metadata": _verifier_metadata(outcome),
        }
        verdict_error: Final = await _apply(guardrail, inputs, request_data)
        status: Final = recorded_status(request_data, self._name)
        if status in _UNVERIFIED_STATUSES:
            raise RuntimeError(f"guardrail {self._name!r} did not evaluate program {outcome.program_id!r}: {status}")
        if status in _BLOCKED_STATUSES:
            return 0.0
        if status == "success" or verdict_error is None:
            return 1.0
        if isinstance(verdict_error, GuardrailRaisedException):
            if verdict_error.blocked_content:
                return 0.0
            raise verdict_error  # the guardrail could not reach a verdict
        if _is_http_block(verdict_error):
            return 0.0
        raise verdict_error


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
