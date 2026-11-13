from dataclasses import dataclass
from typing import Final, TypeAlias

from litellm.types.utils import Usage

ServedRound: TypeAlias = tuple[Usage, float]

_MARK_ATTRIBUTE: Final = "_litellm_agentic_followup_mark"


@dataclass(frozen=True, slots=True)
class _FollowUpMark:
    served_rounds: tuple[ServedRound, ...]
    tool_calls_sent: bool


def _mark_of(error: BaseException) -> _FollowUpMark | None:
    mark: Final[object] = getattr(error, _MARK_ATTRIBUTE, None)
    return mark if isinstance(mark, _FollowUpMark) else None


def _write_mark(error: BaseException, served_rounds: tuple[ServedRound, ...], tool_calls_sent: bool) -> None:
    setattr(error, _MARK_ATTRIBUTE, _FollowUpMark(served_rounds, tool_calls_sent))


def mark_agentic_followup_failure(error: BaseException, served_rounds: tuple[ServedRound, ...]) -> None:
    _write_mark(error, served_rounds, sent_tool_calls_before(error))


def mark_failure_after_tool_calls_sent(error: BaseException) -> None:
    _write_mark(error, rounds_served_before(error), True)


def is_agentic_followup_failure(error: BaseException) -> bool:
    return _mark_of(error) is not None


def sent_tool_calls_before(error: BaseException) -> bool:
    mark: Final = _mark_of(error)
    return mark is not None and mark.tool_calls_sent


def rounds_served_before(error: BaseException) -> tuple[ServedRound, ...]:
    mark: Final = _mark_of(error)
    return () if mark is None else mark.served_rounds


def carry_agentic_followup_mark(source: BaseException, mapped: BaseException) -> None:
    if mapped is not source and is_agentic_followup_failure(source):
        _write_mark(mapped, rounds_served_before(source), sent_tool_calls_before(source))
