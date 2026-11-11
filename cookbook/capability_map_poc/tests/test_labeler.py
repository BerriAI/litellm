import json

from capability_map_poc.corrections import turn_key
from capability_map_poc.labeler import (
    PROMPT_VERSION,
    CompletionError,
    CompletionOk,
    label_conversation,
    label_conversations,
)
from capability_map_poc.schemas import ChatMessage, Conversation, FeedbackLabel, FeedbackLabelName, LabelFailure


class FakeCompleter:
    def __init__(self, replies, model="sonnet"):
        self._replies = list(replies)
        self.model = model
        self.prompts = []

    def complete(self, system, user, max_tokens):
        self.prompts.append(user)
        return self._replies[min(len(self.prompts), len(self._replies)) - 1]


def _conversation(session="s"):
    return Conversation(
        session_id=session,
        messages=(
            ChatMessage(role="user", content="write a haiku"),
            ChatMessage(role="assistant", content="draft"),
            ChatMessage(role="user", content="no, about rain"),
            ChatMessage(role="assistant", content="rain haiku"),
            ChatMessage(role="user", content="great, now a limerick"),
            ChatMessage(role="assistant", content="limerick"),
        ),
    )


def _ok(labels=((2, "NEG_2"), (3, "POS")), fence=False, cost=0.004):
    body = json.dumps({"labels": [{"turn": t, "label": label, "rationale": "r"} for t, label in labels]})
    return CompletionOk(text=f"```json\n{body}\n```" if fence else body, cost_usd=cost)


def test_one_call_labels_every_user_turn_after_the_first_and_splits_cost():
    completer = FakeCompleter([_ok()])
    outcomes = label_conversation(_conversation(), completer)
    assert {(o.turn_idx, o.label) for o in outcomes} == {
        (2, FeedbackLabelName.MAKE_AWARE_WITH_CORRECTION),
        (3, FeedbackLabelName.POSITIVE),
    }
    assert sum(o.cost_usd for o in outcomes) == 0.004
    assert len(completer.prompts) == 1
    prompt = completer.prompts[0]
    assert prompt.index("User turn 1:\nwrite a haiku") < prompt.index("User turn 2:\nno, about rain")
    assert "User turn 3:\ngreat, now a limerick" in prompt
    assert "Assistant:\nlimerick" in prompt


def test_code_fenced_reply_is_parsed():
    outcomes = label_conversation(_conversation(), FakeCompleter([_ok(fence=True)]))
    assert all(isinstance(o, FeedbackLabel) for o in outcomes)


def test_missing_extra_or_repeated_turns_fail_the_whole_conversation_after_one_retry():
    for labels in (((2, "NEG_1"),), ((2, "NEG_1"), (3, "POS"), (4, "NEU")), ((2, "NEG_1"), (2, "POS"))):
        completer = FakeCompleter([_ok(labels)])
        outcomes = label_conversation(_conversation(), completer)
        assert len(completer.prompts) == 2
        assert {o.turn_idx for o in outcomes} == {2, 3}
        assert all(isinstance(o, LabelFailure) for o in outcomes)
        assert sum(o.cost_usd for o in outcomes) == 0.008


def test_retry_recovers_from_unknown_label_and_gateway_error():
    for bad in (_ok(((2, "NEG_9"), (3, "POS"))), CompletionError(reason="RateLimitError: 429")):
        outcomes = label_conversation(_conversation(), FakeCompleter([bad, _ok()]))
        assert all(isinstance(o, FeedbackLabel) for o in outcomes)


def test_gateway_error_twice_keeps_the_reason():
    outcomes = label_conversation(_conversation(), FakeCompleter([CompletionError(reason="RateLimitError: 429")]))
    assert all(isinstance(o, LabelFailure) and "429" in o.reason for o in outcomes)


def test_single_user_turn_needs_no_call():
    completer = FakeCompleter([_ok()])
    conversation = Conversation(session_id="s", messages=(ChatMessage(role="user", content="hi"),))
    assert label_conversation(conversation, completer) == ()
    assert completer.prompts == []


def test_cache_is_reused_only_when_every_turn_has_a_current_label():
    done, partial, other_model = _conversation("done"), _conversation("partial"), _conversation("other")
    cached_outcomes = (
        *label_conversation(done, FakeCompleter([_ok()])),
        label_conversation(partial, FakeCompleter([_ok()]))[0],
        *label_conversation(other_model, FakeCompleter([_ok()], model="haiku")),
    )
    cache = {turn_key(o.session_id, o.turn_idx): o for o in cached_outcomes}
    completer = FakeCompleter([_ok(((2, "NEG_1"), (3, "NEU")))])
    written = []
    outcomes = label_conversations([done, partial, other_model], completer, cache, workers=2, on_labels=written.append)
    by_key = {turn_key(o.session_id, o.turn_idx): o for o in outcomes}
    assert len(completer.prompts) == 2
    assert by_key[("done", 2)].label is FeedbackLabelName.MAKE_AWARE_WITH_CORRECTION
    assert by_key[("partial", 2)].label is FeedbackLabelName.REPHRASING
    assert by_key[("other", 3)].labeler_model == "sonnet"
    assert sorted(o.session_id for batch in written for o in batch) == ["other", "other", "partial", "partial"]
    assert all(o.prompt_version == PROMPT_VERSION for o in outcomes)
