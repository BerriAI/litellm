"""What Codex recorded in a thread's rollout file, `CODEX_HOME/sessions/**/rollout-*-<thread id>.jsonl`.

`codex exec --json` prints no event for a compaction or for a sub-agent's own turn, so the rollout is the only place
they show. Each line is `{"timestamp", "type", "payload"}` (`codex-rs/history/src/rollout_payload.rs`).
A remote compaction is one whose replacement history carries the provider's `compaction` item, while a local one
replaces the history with a summary Codex wrote itself. A sub-agent gets a rollout of its own whose `session_meta`
names the thread that spawned it, and carries `forked_from_id` when it inherited that thread's conversation. Each turn
it finishes ends in a `task_complete` event with its last message.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

_COMPACTED: Final = "compacted"
_SESSION_META: Final = "session_meta"
_EVENT: Final = "event_msg"
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


class _SessionMeta(_Frozen):
    parent_thread_id: str | None = None
    forked_from_id: str | None = None


class _Event(_Frozen):
    type: str


class _TaskComplete(_Frozen):
    type: Literal["task_complete"]
    last_agent_message: str | None = None


@dataclass(frozen=True, slots=True)
class Compaction:
    remote: bool


@dataclass(frozen=True, slots=True)
class Subagent:
    forked: bool
    replies: tuple[str, ...]


def rollout_compactions(rollout: str) -> tuple[Compaction, ...]:
    return tuple(_compaction(line.payload) for line in _lines(rollout) if line.type == _COMPACTED)


def rollout_parent_thread(rollout: str) -> str | None:
    return _session_meta(rollout).parent_thread_id


def rollout_subagent(rollout: str) -> Subagent:
    return Subagent(forked=_session_meta(rollout).forked_from_id is not None, replies=rollout_final_messages(rollout))


def rollout_final_messages(rollout: str) -> tuple[str, ...]:
    events: Final = (line.payload for line in _lines(rollout) if line.type == _EVENT)
    completions: Final = (
        _TaskComplete.model_validate(event) for event in events if _Event.model_validate(event).type == "task_complete"
    )
    return tuple(done.last_agent_message for done in completions if done.last_agent_message is not None)


def _session_meta(rollout: str) -> _SessionMeta:
    meta: Final = next((line.payload for line in _lines(rollout) if line.type == _SESSION_META), None)
    return _SessionMeta.model_validate(meta) if meta is not None else _SessionMeta()


def _lines(rollout: str) -> tuple[_Line, ...]:
    return tuple(_Line.model_validate_json(line) for line in rollout.splitlines() if line.strip())


def _compaction(payload: object) -> Compaction:
    history: Final = _CompactedPayload.model_validate(payload).replacement_history or ()
    return Compaction(remote=any(item.type == _PROVIDER_COMPACTION_ITEM for item in history))
