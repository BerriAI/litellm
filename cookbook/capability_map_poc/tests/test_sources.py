from capability_map_poc.schemas import FeedbackLabelName
from capability_map_poc.sources import parse_emnlp


def _row(conversation, labels, texts, conversation_id="c1", source="lmsys"):
    return {
        "conversation_id": conversation_id,
        "dataset_source": source,
        "conversation": [{"role": r, "content": c} for r, c in conversation],
        "user_feedback_category": labels,
        "user_feedback_text": texts,
        "total_turns": 0,
        "eval_type": "dense",
    }


CONVERSATION = [
    ("user", "write a haiku"),
    ("assistant", "draft one"),
    ("user", "no, it must be about rain"),
    ("assistant", "rain haiku"),
    ("assistant", "second part"),
    ("user", "now a limerick"),
    ("assistant", "limerick"),
]


def test_dense_labels_align_positionally_with_the_exchange_they_judge():
    parse = parse_emnlp([_row(CONVERSATION, ["NEG_2", "NEU"], ["no, it must be about rain", ""])], "dense")
    first, second = parse.examples
    assert (first.turn.request, first.turn.response, first.turn.next_user_turn) == (
        "write a haiku",
        "draft one",
        "no, it must be about rain",
    )
    assert first.gold.label is FeedbackLabelName.MAKE_AWARE_WITH_CORRECTION
    assert (second.turn.response, second.turn.next_user_turn) == ("rain haiku\nsecond part", "now a limerick")
    assert second.gold.label is FeedbackLabelName.NO_FEEDBACK
    assert (first.turn.turn_idx, second.turn.turn_idx) == (2, 3)
    assert first.turn.group_key == "lmsys"
    assert parse.text_mismatches == 0


def test_dense_conversation_with_wrong_label_count_is_skipped_not_misaligned():
    parse = parse_emnlp([_row(CONVERSATION, ["NEG_2"], ["no, it must be about rain"], conversation_id="bad")], "dense")
    assert parse.examples == ()
    assert parse.skipped_conversations == ("bad",)


def test_dense_reports_feedback_text_that_does_not_match_its_turn():
    parse = parse_emnlp([_row(CONVERSATION, ["NEG_2", "NEU"], ["something else entirely", ""])], "dense")
    assert parse.text_mismatches == 1


def test_sparse_labels_align_by_feedback_text():
    parse = parse_emnlp([_row(CONVERSATION, ["NEG_1"], ["Now a  LIMERICK"])], "sparse")
    (only,) = parse.examples
    assert only.turn.turn_idx == 3
    assert only.turn.request == "no, it must be about rain"
    assert only.gold.label is FeedbackLabelName.REPHRASING


def test_sparse_duplicate_labels_for_one_turn_keep_the_first():
    parse = parse_emnlp([_row(CONVERSATION, ["NEG_1", "POS"], ["now a limerick", "a limerick"])], "sparse")
    (only,) = parse.examples
    assert only.gold.label is FeedbackLabelName.REPHRASING
    assert (parse.duplicate_feedback, parse.unaligned_feedback) == (1, 0)


def test_kept_conversations_carry_the_full_transcript_and_skipped_ones_do_not():
    good = _row(CONVERSATION, ["NEG_2", "NEU"], ["no, it must be about rain", ""], conversation_id="good")
    bad = _row(CONVERSATION, ["NEG_2"], ["x"], conversation_id="bad")
    parse = parse_emnlp([good, bad], "dense")
    (conversation,) = parse.conversations
    assert conversation.session_id == parse.examples[0].turn.session_id
    assert [(m.role, m.content) for m in conversation.messages] == CONVERSATION


def test_sparse_feedback_without_a_matching_turn_is_counted_as_unaligned():
    parse = parse_emnlp([_row(CONVERSATION, ["NEG_1", "POS"], ["now a limerick", "great job"])], "sparse")
    assert len(parse.examples) == 1
    assert parse.unaligned_feedback == 1
