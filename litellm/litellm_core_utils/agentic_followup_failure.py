import weakref
from typing import Final, TypeAlias

from litellm.types.utils import Usage

ServedRound: TypeAlias = tuple[Usage, float]

_followup_failures: Final[weakref.WeakKeyDictionary[BaseException, tuple[ServedRound, ...]]] = (
    weakref.WeakKeyDictionary()
)


def _can_be_marked(error: BaseException) -> bool:
    return type(error).__weakrefoffset__ != 0


def mark_agentic_followup_failure(error: BaseException, served_rounds: tuple[ServedRound, ...]) -> None:
    if _can_be_marked(error):
        _followup_failures[error] = served_rounds


def is_agentic_followup_failure(error: BaseException) -> bool:
    return _can_be_marked(error) and error in _followup_failures


def rounds_served_before(error: BaseException) -> tuple[ServedRound, ...]:
    return _followup_failures.get(error, ()) if _can_be_marked(error) else ()
