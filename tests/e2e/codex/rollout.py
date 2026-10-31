"""The compactions Codex recorded in a thread's rollout file, `CODEX_HOME/sessions/**/rollout-*-<thread id>.jsonl`.

`codex exec --json` prints no event for a compaction, so the rollout is the only place it shows. Each line is
`{"timestamp", "type", "payload"}` (`codex-rs/history/src/rollout_payload.rs`) and only `compacted` lines are read.
A remote compaction is one whose replacement history carries the provider's `compaction` item, while a local one
replaces the history with a summary Codex wrote itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, ConfigDict, TypeAdapter

_COMPACTED: Final = "compacted"
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


_LINE: Final = TypeAdapter(_Line)
_COMPACTED_PAYLOAD: Final = TypeAdapter(_CompactedPayload)


@dataclass(frozen=True, slots=True)
class Compaction:
    remote: bool


def rollout_compactions(rollout: str) -> tuple[Compaction, ...]:
    lines: Final = tuple(_LINE.validate_json(line) for line in rollout.splitlines() if line.strip())
    return tuple(_compaction(line.payload) for line in lines if line.type == _COMPACTED)


def _compaction(payload: object) -> Compaction:
    history: Final = _COMPACTED_PAYLOAD.validate_python(payload).replacement_history or ()
    return Compaction(remote=any(item.type == _PROVIDER_COMPACTION_ITEM for item in history))
