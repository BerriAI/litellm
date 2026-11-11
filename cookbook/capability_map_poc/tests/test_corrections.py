from capability_map_poc.corrections import correction_stats
from capability_map_poc.schemas import FeedbackLabel, FeedbackLabelName, FeedbackTurn, LabelFailure


def _turn(session, idx, model="haiku", group="mms", next_turn="it still fails"):
    return FeedbackTurn(
        session_id=session,
        source="tau2",
        model_id=model,
        group_key=group,
        turn_idx=idx,
        request="my mms does not send",
        response="restart your phone",
        next_user_turn=next_turn,
    )


def _label(session, idx, label):
    return FeedbackLabel(
        session_id=session,
        turn_idx=idx,
        label=label,
        rationale="r",
        labeler_model="m",
        prompt_version="v",
        cost_usd=None,
    )


def _stats(turns, outcomes):
    return {(s.model_id, s.group_key): s for s in correction_stats(turns, outcomes)}


def test_neg1_to_neg3_count_as_corrections_and_neg4_is_not_observed():
    turns = [_turn("s", i) for i in range(6)]
    outcomes = [
        _label("s", 0, FeedbackLabelName.REPHRASING),
        _label("s", 1, FeedbackLabelName.MAKE_AWARE_WITH_CORRECTION),
        _label("s", 2, FeedbackLabelName.MAKE_AWARE_WITHOUT_CORRECTION),
        _label("s", 3, FeedbackLabelName.ASK_FOR_CLARIFICATION),
        _label("s", 4, FeedbackLabelName.POSITIVE),
        _label("s", 5, FeedbackLabelName.NO_FEEDBACK),
    ]
    stats = _stats(turns, outcomes)[("haiku", "mms")]
    assert (stats.correction_count, stats.correction_observations) == (3, 5)
    assert stats.correction_rate == 0.6


def test_turns_without_next_user_turn_label_or_with_failed_label_are_not_observed():
    turns = [_turn("s", 0, next_turn=None), _turn("s", 1), _turn("s", 2)]
    outcomes = [
        _label("s", 0, FeedbackLabelName.REPHRASING),
        LabelFailure(session_id="s", turn_idx=1, reason="x", labeler_model="m", prompt_version="v", cost_usd=None),
    ]
    stats = _stats(turns, outcomes)[("haiku", "mms")]
    assert (stats.correction_count, stats.correction_observations, stats.correction_rate) == (0, 0, None)


def test_stats_are_grouped_by_model_and_group_key():
    turns = [_turn("a", 0, model="haiku"), _turn("b", 0, model="sonnet"), _turn("c", 0, model="sonnet", group="data")]
    outcomes = [
        _label("a", 0, FeedbackLabelName.REPHRASING),
        _label("b", 0, FeedbackLabelName.NO_FEEDBACK),
        _label("c", 0, FeedbackLabelName.MAKE_AWARE_WITHOUT_CORRECTION),
    ]
    stats = _stats(turns, outcomes)
    assert {key: s.correction_rate for key, s in stats.items()} == {
        ("haiku", "mms"): 1.0,
        ("sonnet", "mms"): 0.0,
        ("sonnet", "data"): 1.0,
    }


def test_labels_join_on_session_and_turn_not_on_turn_index_alone():
    turns = [_turn("a", 0), _turn("b", 0)]
    outcomes = [_label("a", 0, FeedbackLabelName.REPHRASING), _label("b", 0, FeedbackLabelName.POSITIVE)]
    stats = _stats(turns, outcomes)[("haiku", "mms")]
    assert (stats.correction_count, stats.correction_observations) == (1, 2)
