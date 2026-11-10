import json

from capability_map_poc.corrections import turn_key
from capability_map_poc.labeler import PROMPT_VERSION, CompletionError, CompletionOk, label_turn, label_turns
from capability_map_poc.schemas import FeedbackLabel, FeedbackLabelName, FeedbackTurn, LabelFailure


class FakeCompleter:
    def __init__(self, reply, model="sonnet"):
        self._reply = reply
        self.model = model
        self.prompts = []

    def complete(self, system, user):
        self.prompts.append(user)
        return self._reply


def _turn(idx=2, session="s"):
    return FeedbackTurn(
        session_id=session,
        source="emnlp",
        model_id="m",
        group_key="lmsys",
        turn_idx=idx,
        request="write a haiku",
        response="draft",
        next_user_turn="no, about rain",
    )


def _ok(label="NEG_2", confident=True, fence=False):
    body = json.dumps({"label": label, "confident": confident, "rationale": "gives the fix"})
    return CompletionOk(text=f"```json\n{body}\n```" if fence else body, cost_usd=0.002)


def test_valid_reply_becomes_label_with_cost_and_the_prompt_carries_all_three_turns():
    completer = FakeCompleter(_ok())
    outcome = label_turn(_turn(), completer)
    assert isinstance(outcome, FeedbackLabel)
    assert (outcome.label, outcome.confident, outcome.cost_usd) == (
        FeedbackLabelName.MAKE_AWARE_WITH_CORRECTION,
        True,
        0.002,
    )
    assert all(part in completer.prompts[0] for part in ("write a haiku", "draft", "no, about rain"))


def test_code_fenced_reply_is_parsed():
    assert isinstance(label_turn(_turn(), FakeCompleter(_ok(fence=True))), FeedbackLabel)


def test_unknown_label_becomes_failure_and_keeps_cost():
    outcome = label_turn(_turn(), FakeCompleter(_ok(label="NEG_9")))
    assert isinstance(outcome, LabelFailure)
    assert outcome.cost_usd == 0.002


def test_gateway_error_becomes_failure():
    outcome = label_turn(_turn(), FakeCompleter(CompletionError(reason="RateLimitError: 429")))
    assert isinstance(outcome, LabelFailure)
    assert "429" in outcome.reason


def test_label_turns_reuses_matching_cache_and_relabels_failures_and_other_models():
    turns = [_turn(2), _turn(3), _turn(4)]
    cached_ok = label_turn(turns[0], FakeCompleter(_ok("POS")))
    cached_failure = label_turn(turns[1], FakeCompleter(CompletionError(reason="boom")))
    other_model = label_turn(turns[2], FakeCompleter(_ok("POS"), model="haiku"))
    cache = {turn_key(o.session_id, o.turn_idx): o for o in (cached_ok, cached_failure, other_model)}
    completer = FakeCompleter(_ok("NEG_1"))
    written = []
    outcomes = label_turns(turns, completer, cache, workers=2, on_label=written.append)
    by_idx = {o.turn_idx: o for o in outcomes}
    assert len(completer.prompts) == 2
    assert by_idx[2].label is FeedbackLabelName.POSITIVE
    assert by_idx[3].label is FeedbackLabelName.REPHRASING
    assert by_idx[4].labeler_model == "sonnet"
    assert sorted(o.turn_idx for o in written) == [3, 4]
    assert all(o.prompt_version == PROMPT_VERSION for o in outcomes)
