import pytest

from capability_map_poc.evaluate import evaluate, mismatches
from capability_map_poc.schemas import FeedbackLabel, FeedbackLabelName, GoldFeedback, LabelFailure

N1, N2, N3, N4 = (FeedbackLabelName(f"NEG_{i}") for i in range(1, 5))
POS, NEU = FeedbackLabelName.POSITIVE, FeedbackLabelName.NO_FEEDBACK


def _gold(idx, label):
    return GoldFeedback(session_id="s", turn_idx=idx, label=label)


def _pred(idx, label):
    return FeedbackLabel(
        session_id="s",
        turn_idx=idx,
        label=label,
        rationale="r",
        labeler_model="m",
        prompt_version="v",
        cost_usd=None,
    )


GOLD = [_gold(0, N1), _gold(1, N2), _gold(2, N3), _gold(3, NEU), _gold(4, N4), _gold(5, POS), _gold(6, N2)]
PRED = [
    _pred(0, N2),
    _pred(1, N2),
    _pred(2, NEU),
    _pred(3, N1),
    _pred(4, N4),
    _pred(5, POS),
    LabelFailure(session_id="s", turn_idx=6, reason="x", labeler_model="m", prompt_version="v", cost_usd=None),
]


def test_correction_score_uses_neg1_to_neg3_and_excludes_failures():
    report = evaluate(GOLD, PRED)
    assert (report.gold_turns, report.labeled_turns, report.failed_turns) == (7, 6, 1)
    assert report.corrective.support == 3
    assert report.corrective.precision == pytest.approx(2 / 3)
    assert report.corrective.recall == pytest.approx(2 / 3)
    assert report.corrective.f1 == pytest.approx(2 / 3)


def test_accuracy_levels_differ_in_granularity():
    report = evaluate(GOLD, PRED)
    assert report.fine_grained_accuracy == pytest.approx(3 / 6)
    assert report.three_way_accuracy == pytest.approx(4 / 6)
    assert report.any_feedback_accuracy == pytest.approx(4 / 6)


def test_confusion_matrix_rows_are_gold_and_columns_are_predicted():
    report = evaluate(GOLD, PRED)
    order = list(report.confusion_labels)
    assert report.confusion[order.index(N1)][order.index(N2)] == 1
    assert report.confusion[order.index(NEU)][order.index(N1)] == 1
    assert report.confusion[order.index(N3)][order.index(NEU)] == 1
    assert sum(map(sum, report.confusion)) == 6


def test_mismatches_list_only_disagreements():
    assert [(m.turn_idx, m.gold, m.predicted) for m in mismatches(GOLD, PRED)] == [
        (0, N1, N2),
        (2, N3, NEU),
        (3, NEU, N1),
    ]
