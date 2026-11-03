"""What Codex recorded in a thread's rollout file, `CODEX_HOME/sessions/**/rollout-*-<thread id>.jsonl`.

`codex exec --json` prints no event for a compaction or for the model a turn ran on, so the rollout is the only place
they show. Each line is `{"timestamp", "type", "payload"}` (`codex-rs/history/src/rollout_payload.rs`) and only
`compacted` and `turn_context` lines are read. A remote compaction is one whose replacement history carries the
provider's `compaction` item, while a local one replaces the history with a summary Codex wrote itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, ConfigDict

_COMPACTED: Final = "compacted"
_TURN_CONTEXT: Final = "turn_context"
_PROVIDER_COMPACTION_ITEM: Final = "compaction"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class _Line(_Frozen):
    type: str
    payload: object = None


class _HistoryItem(_Frozen):
    type: str


class _CompactedPayload(_Frozen):
    replacement_history: tuple[_HistoryItem, ...] | None = None


class _TurnContextPayload(_Frozen):
    model: str


@dataclass(frozen=True, slots=True)
class Compaction:
    remote: bool


def rollout_compactions(rollout: str) -> tuple[Compaction, ...]:
    return tuple(_compaction(line.payload) for line in _lines(rollout) if line.type == _COMPACTED)


def rollout_turn_models(rollout: str) -> tuple[str, ...]:
    return tuple(
        _TurnContextPayload.model_validate(line.payload).model for line in _lines(rollout) if line.type == _TURN_CONTEXT
    )


def _lines(rollout: str) -> tuple[_Line, ...]:
    return tuple(_Line.model_validate_json(line) for line in rollout.splitlines() if line.strip())


def _compaction(payload: object) -> Compaction:
    history: Final = _CompactedPayload.model_validate(payload).replacement_history or ()
    return Compaction(remote=any(item.type == _PROVIDER_COMPACTION_ITEM for item in history))
