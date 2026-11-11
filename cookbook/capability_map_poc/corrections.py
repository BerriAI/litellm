from collections.abc import Iterable, Mapping
from itertools import groupby
from typing import Final

from .schemas import (
    CORRECTIVE_LABELS,
    CorrectionStats,
    FeedbackLabel,
    FeedbackLabelName,
    FeedbackTurn,
    LabelOutcome,
)

TurnKey = tuple[str, int]


def turn_key(session_id: str, turn_idx: int) -> TurnKey:
    return (session_id, turn_idx)


def observed_label(turn: FeedbackTurn, outcome: LabelOutcome | None) -> FeedbackLabelName | None:
    if turn.next_user_turn is None or not isinstance(outcome, FeedbackLabel):
        return None
    if outcome.label is FeedbackLabelName.ASK_FOR_CLARIFICATION:
        return None
    return outcome.label


def correction_stats(turns: Iterable[FeedbackTurn], outcomes: Iterable[LabelOutcome]) -> tuple[CorrectionStats, ...]:
    by_key: Final[Mapping[TurnKey, LabelOutcome]] = {turn_key(o.session_id, o.turn_idx): o for o in outcomes}
    observed: Final = sorted(
        (((t.model_id, t.group_key), observed_label(t, by_key.get(turn_key(t.session_id, t.turn_idx)))) for t in turns),
        key=lambda row: row[0],
    )
    return tuple(_group_stats(key, [label for _, label in rows]) for key, rows in groupby(observed, key=lambda r: r[0]))


def _group_stats(key: tuple[str, str], labels: list[FeedbackLabelName | None]) -> CorrectionStats:
    eligible: Final = [label for label in labels if label is not None]
    count: Final = sum(label in CORRECTIVE_LABELS for label in eligible)
    return CorrectionStats(
        model_id=key[0],
        group_key=key[1],
        correction_count=count,
        correction_observations=len(eligible),
        correction_rate=count / len(eligible) if eligible else None,
    )
