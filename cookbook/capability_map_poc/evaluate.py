from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Final, Literal

from .corrections import TurnKey, turn_key
from .schemas import CORRECTIVE_LABELS, FeedbackLabel, FeedbackLabelName, Frozen, GoldFeedback, LabelOutcome

LABEL_ORDER: Final = tuple(FeedbackLabelName)
ThreeWay = Literal["POS", "NEG", "NEU"]


class BinaryScore(Frozen):
    precision: float
    recall: float
    f1: float
    support: int


class PerLabelScore(Frozen):
    label: FeedbackLabelName
    precision: float
    recall: float
    f1: float
    support: int


class FeedbackEval(Frozen):
    gold_turns: int
    labeled_turns: int
    failed_turns: int
    fine_grained_accuracy: float
    three_way_accuracy: float
    any_feedback_accuracy: float
    corrective: BinaryScore
    per_label: tuple[PerLabelScore, ...]
    confusion_labels: tuple[FeedbackLabelName, ...]
    confusion: tuple[tuple[int, ...], ...]


class Mismatch(Frozen):
    session_id: str
    turn_idx: int
    gold: FeedbackLabelName
    predicted: FeedbackLabelName
    rationale: str


def three_way(label: FeedbackLabelName) -> ThreeWay:
    match label:
        case FeedbackLabelName.POSITIVE:
            return "POS"
        case FeedbackLabelName.NO_FEEDBACK:
            return "NEU"
        case _:
            return "NEG"


def predicted_corrective(label: FeedbackLabel) -> bool:
    return label.label in CORRECTIVE_LABELS


def paired(
    gold: Iterable[GoldFeedback], outcomes: Iterable[LabelOutcome]
) -> tuple[tuple[tuple[GoldFeedback, FeedbackLabel], ...], int]:
    by_key: Final[Mapping[TurnKey, LabelOutcome]] = {turn_key(o.session_id, o.turn_idx): o for o in outcomes}
    gold_rows: Final = tuple(gold)
    matched: Final = tuple((g, by_key.get(turn_key(g.session_id, g.turn_idx))) for g in gold_rows)
    pairs: Final = tuple((g, o) for g, o in matched if isinstance(o, FeedbackLabel))
    return pairs, len(gold_rows) - len(pairs)


def binary_score(gold: Sequence[bool], predicted: Sequence[bool]) -> BinaryScore:
    true_pos: Final = sum(g and p for g, p in zip(gold, predicted))
    predicted_pos: Final = sum(predicted)
    gold_pos: Final = sum(gold)
    precision: Final = true_pos / predicted_pos if predicted_pos else 0.0
    recall: Final = true_pos / gold_pos if gold_pos else 0.0
    f1: Final = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return BinaryScore(precision=precision, recall=recall, f1=f1, support=gold_pos)


def evaluate(gold: Iterable[GoldFeedback], outcomes: Iterable[LabelOutcome]) -> FeedbackEval:
    pairs, failed = paired(gold, outcomes)
    gold_labels: Final = tuple(g.label for g, _ in pairs)
    pred_labels: Final = tuple(p.label for _, p in pairs)
    both: Final = tuple(zip(gold_labels, pred_labels))
    cells: Final = Counter(both)
    per_label: Final = tuple(
        (label, binary_score([g is label for g in gold_labels], [p is label for p in pred_labels]))
        for label in LABEL_ORDER
    )
    return FeedbackEval(
        gold_turns=len(pairs) + failed,
        labeled_turns=len(pairs),
        failed_turns=failed,
        fine_grained_accuracy=_share(g is p for g, p in both),
        three_way_accuracy=_share(three_way(g) == three_way(p) for g, p in both),
        any_feedback_accuracy=_share(
            (g is FeedbackLabelName.NO_FEEDBACK) == (p is FeedbackLabelName.NO_FEEDBACK) for g, p in both
        ),
        corrective=binary_score(
            [g.label in CORRECTIVE_LABELS for g, _ in pairs], [predicted_corrective(p) for _, p in pairs]
        ),
        per_label=tuple(
            PerLabelScore(label=label, precision=s.precision, recall=s.recall, f1=s.f1, support=s.support)
            for label, s in per_label
        ),
        confusion_labels=LABEL_ORDER,
        confusion=tuple(tuple(cells[(g, p)] for p in LABEL_ORDER) for g in LABEL_ORDER),
    )


def mismatches(gold: Iterable[GoldFeedback], outcomes: Iterable[LabelOutcome]) -> tuple[Mismatch, ...]:
    pairs, _ = paired(gold, outcomes)
    return tuple(
        Mismatch(
            session_id=g.session_id,
            turn_idx=g.turn_idx,
            gold=g.label,
            predicted=p.label,
            rationale=p.rationale,
        )
        for g, p in sorted(pairs, key=lambda pair: (pair[0].session_id, pair[0].turn_idx))
        if g.label != p.label
    )


def _share(flags: Iterable[bool]) -> float:
    values: Final = tuple(flags)
    return sum(values) / len(values) if values else 0.0
