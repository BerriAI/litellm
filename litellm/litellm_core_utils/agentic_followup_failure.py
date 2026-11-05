from dataclasses import dataclass
from typing import Final, TypeAlias

from litellm.types.utils import Usage

ServedRound: TypeAlias = tuple[Usage, float]

_MARK_ATTRIBUTE: Final = "_litellm_agentic_followup_mark"


@dataclass(frozen=True, slots=True)
class _FollowUpMark:
    served_rounds: tuple[ServedRound, ...]


def _mark_of(error: BaseException) -> _FollowUpMark | None:
    mark: Final[object] = getattr(error, _MARK_ATTRIBUTE, None)
    return mark if isinstance(mark, _FollowUpMark) else None


def mark_agentic_followup_failure(error: BaseException, served_rounds: tuple[ServedRound, ...]) -> None:
    setattr(error, _MARK_ATTRIBUTE, _FollowUpMark(served_rounds))


def is_agentic_followup_failure(error: BaseException) -> bool:
    return _mark_of(error) is not None


def rounds_served_before(error: BaseException) -> tuple[ServedRound, ...]:
    mark: Final = _mark_of(error)
    return () if mark is None else mark.served_rounds


def carry_agentic_followup_mark(source: BaseException, mapped: BaseException) -> None:
    if mapped is not source and is_agentic_followup_failure(source):
        mark_agentic_followup_failure(mapped, rounds_served_before(source))
