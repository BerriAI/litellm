import json
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .activity import ActivityTracker, observe_operation
from .analysis import AnalysisContextExceeded, AnalysisResponseError, ModelCall, structured_response
from .models import ModelMessage, ModelRequest, Record


class Checkpoint(Record):
    working_notes: str = Field(min_length=1)


class JournalPosition(BaseModel):
    model_config = ConfigDict(extra="ignore")
    journal_turns: int = 0
    resume_history_from_turn: int | None = None


def visible_journal(messages: tuple[ModelMessage, ...]) -> int:
    positions: Final = tuple(journal_position(message) for message in messages)
    visible: Final = max((position.journal_turns for position in positions), default=0)
    return min(
        (position.resume_history_from_turn for position in positions if position.resume_history_from_turn is not None),
        default=visible,
    )


def journal_position(message: ModelMessage) -> JournalPosition:
    if message.role != "user":
        return JournalPosition()
    try:
        return JournalPosition.model_validate_json(message.content)
    except ValidationError:
        return JournalPosition()


async def checkpoint_prefix(
    request: ModelRequest,
    instruction: ModelMessage,
    model: ModelCall,
) -> tuple[Checkpoint, tuple[ModelMessage, ...]]:
    try:
        notes: Final = await structured_response(
            request.model_copy(update=MappingProxyType({"messages": (*request.messages, instruction)})),
            Checkpoint,
            model,
        )
        return notes, request.messages
    except AnalysisContextExceeded as error:
        if len(request.messages) == 1:
            raise AnalysisResponseError(
                "The Lens task alone cannot fit in the analysis model's context window. "
                "Use a model with more context or shorten the investigation instructions."
            ) from error
        shorter: Final = request.messages[: max(1, len(request.messages) // 2)]
        prefix: Final = shorter[:-1] if len(shorter) > 1 and shorter[-1].role == "assistant" else shorter
        return await checkpoint_prefix(
            request.model_copy(update=MappingProxyType({"messages": prefix})), instruction, model
        )


async def compact_context(
    request: ModelRequest,
    model: ModelCall,
    journal_turns: int,
    activity: ActivityTracker | None,
) -> tuple[ModelMessage, ...]:
    instruction: Final = ModelMessage(
        role="system",
        content=json.dumps(
            {
                "task": (
                    "Compact this analysis conversation so the investigation can continue. Return only "
                    "working_notes, a concise replacement memory of the material visible here. Preserve the "
                    "assignment, coverage, supported leads, exact evidence references, counterexamples, "
                    "existing finding IDs, statuses and feedback, unresolved questions and next steps. "
                    "Do not issue tools or finalize findings. The original "
                    "evidence and complete tool journal remain available. Some later tool results may have "
                    "been excluded from this compaction request because they exceeded the context window; "
                    "do not claim to have inspected anything you cannot see. The continuation will identify "
                    "the archived turns it must still inspect."
                ),
                "response_schema": Checkpoint.model_json_schema(),
            }
        ),
    )
    async with observe_operation(activity, "checkpoint"):
        notes, prefix = await checkpoint_prefix(request, instruction, model)
    return (
        request.messages[0],
        ModelMessage(
            role="user",
            content=json.dumps(
                {
                    "working_notes": notes.working_notes,
                    "journal_turns": journal_turns,
                    "resume_history_from_turn": visible_journal(prefix),
                    "initial_context_archived": True,
                },
                ensure_ascii=False,
            ),
        ),
    )
