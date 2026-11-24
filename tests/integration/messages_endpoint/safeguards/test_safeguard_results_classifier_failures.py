"""A classifier that fails, stalls, refuses, or answers nonsense degrades every tool use to ``unavailable`` with the
reason Claude Code expects, and the classifier prompt carries only the context the feature documents."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import eventually
from integration._support.process import graceful_stop_seconds
from integration._support.safeguards_rig import (
    CHAT_MODEL,
    CLASSIFIER_CONTEXT,
    CLASSIFIER_MODEL,
    Responder,
    Rig,
    auto_mode_body,
    chat_backend,
    chat_completion_reply,
    classifier_answering,
    classifier_failing,
    classifier_flagging_everything,
    classifier_prompt,
    classifier_slow,
    deployment,
    flagged,
    ids_under_review,
    json_body,
    messages_body,
    nonce,
    one_classifier_call,
    safeguards_rig,
    tool_id,
    tool_use_ids,
    unavailable,
)
from integration._support.wire import Reply, Request
from pydantic import JsonValue

pytestmark = pytest.mark.timeout(graceful_stop_seconds() + 240)

BOUNDED_CLASSIFIER: Final = "safeguards-classifier-bounded"
RESERVED_CLASSIFIER: Final = "safeguards-classifier-reserved"
OMISSION_MARKER: Final = json.dumps({"meta": "earlier turns omitted"})
TRANSCRIPT_BUDGET: Final = 24_000


def verdicts_shaped(shape: str) -> Responder:
    def respond(request: Request) -> Reply:
        verdicts: Final = {identity: shape for identity in ids_under_review(request)}
        return chat_completion_reply(json.dumps({"verdicts": verdicts}))

    return respond


FAILURES: Final[Mapping[str, tuple[Responder, str]]] = {
    "server_error": (classifier_failing(500), "error"),
    "context_length": (
        classifier_failing(
            400,
            message="This model's maximum context length is 128000 tokens. However, your messages resulted in 130000 tokens.",
            error_type="invalid_request_error",
            code="context_length_exceeded",
        ),
        "input_too_long",
    ),
    "content_policy": (
        classifier_failing(
            400,
            message="Invalid prompt: your prompt was flagged as potentially violating our usage policy. content_policy_violation",
            error_type="invalid_request_error",
            code="content_policy_violation",
        ),
        "refused",
    ),
    "unparsable": (classifier_answering("I cannot classify that"), "error"),
    "missing_verdict": (classifier_answering(json.dumps({"verdicts": {}})), "error"),
    "invalid_shape": (verdicts_shaped("flagged"), "error"),
}


def classify(rig: Rig, classifier: Responder, **extra: JsonValue) -> tuple[dict[str, JsonValue], str]:
    request_nonce: Final = nonce()
    rig.register(request_nonce, chat_backend(tool_id(request_nonce), stream=False), classifier)
    return json_body(rig.post(auto_mode_body(CHAT_MODEL, request_nonce, **extra))), request_nonce


@pytest.mark.parametrize("failure", tuple(FAILURES))
def test_classifier_failure_degrades_to_unavailable_with_its_reason(rig: Rig, failure: str) -> None:
    classifier, reason = FAILURES[failure]
    body, request_nonce = classify(rig, classifier)
    ids: Final = tool_use_ids(body)
    assert len(ids) == 1, body
    assert body["safeguard_results"] == unavailable(ids, reason), body
    classifier_call, backend_calls = one_classifier_call(rig, request_nonce)
    assert ids_under_review(classifier_call) == ids
    assert len(backend_calls) == 1


def test_rejected_classifier_key_is_unavailable_with_one_warning_and_no_traceback(rig: Rig) -> None:
    rejected_key: Final = classifier_failing(
        401, message="Incorrect API key provided", error_type="invalid_request_error", code="invalid_api_key"
    )
    log_offset: Final = rig.owned.log.stat().st_size
    body, request_nonce = classify(rig, rejected_key)
    ids: Final = tool_use_ids(body)
    assert len(ids) == 1, body
    assert body["safeguard_results"] == unavailable(ids, "error"), body
    one_classifier_call(rig, request_nonce)
    named: Final = f"safeguards classifier {CLASSIFIER_MODEL}"
    tail: Final = eventually(
        lambda: rig.owned.log.read_bytes()[log_offset:].decode(errors="replace"),
        lambda text: named in text,
        seconds=30,
    )
    lines: Final = tuple(line for line in tail.splitlines() if named in line)
    assert len(lines) == 1, tail
    assert "WARNING" in lines[0], tail
    assert "Traceback" not in tail, tail


@pytest.mark.timeout(graceful_stop_seconds() + 360)
def test_classifier_stalled_past_the_deadline_is_a_timeout(rig: Rig) -> None:
    body, request_nonce = classify(rig, classifier_slow(35))
    ids: Final = tool_use_ids(body)
    assert body["safeguard_results"] == unavailable(ids, "timeout"), body
    assert len(rig.classifier_calls(rig.received(request_nonce))) == 1


def bounded_classifier(wire_url: str) -> Sequence[Mapping[str, JsonValue]]:
    return (deployment(BOUNDED_CLASSIFIER, "openai/gpt-5.4-mini", f"{wire_url}/classifier/v1", timeout=2),)


def reserved_for_another_team(start: str, end: str) -> Mapping[str, JsonValue]:
    return {"start": start, "end": end, "timezone": "UTC", "team_ids": ["safeguards-other-team"]}


def reserved_classifier(wire_url: str) -> Sequence[Mapping[str, JsonValue]]:
    whole_day: Final = (reserved_for_another_team("00:00", "12:00"), reserved_for_another_team("12:00", "00:00"))
    return (
        {
            **deployment(RESERVED_CLASSIFIER, "openai/gpt-5.4-mini", f"{wire_url}/classifier/v1"),
            "model_info": {"access_windows": [dict(window) for window in whole_day]},
        },
    )


@pytest.mark.timeout(2 * graceful_stop_seconds() + 240)
def test_classifier_deployment_timeout_bounds_the_classifier_call(tmp_path: Path) -> None:
    with safeguards_rig(tmp_path, classifier_model=BOUNDED_CLASSIFIER, extra_models=bounded_classifier) as rig:
        body, request_nonce = classify(rig, classifier_slow(10))
        ids: Final = tool_use_ids(body)
        assert len(ids) == 1, body
        assert body["safeguard_results"] == unavailable(ids, "timeout"), body
        assert len(rig.classifier_calls(rig.received(request_nonce))) == 1


@pytest.mark.timeout(2 * graceful_stop_seconds() + 240)
def test_router_refusal_other_than_no_healthy_deployments_is_unavailable(tmp_path: Path) -> None:
    with safeguards_rig(tmp_path, classifier_model=RESERVED_CLASSIFIER, extra_models=reserved_classifier) as rig:
        body, request_nonce = classify(rig, classifier_flagging_everything)
        ids: Final = tool_use_ids(body)
        assert len(ids) == 1, body
        assert body["safeguard_results"] == unavailable(ids, "error"), body
        received: Final = rig.received(request_nonce)
        assert rig.classifier_calls(received) == (), [request.target for request in received]
        assert len(rig.backend_calls(received)) == 1, [request.target for request in received]


def test_classifier_prompt_carries_only_the_documented_context(rig: Rig) -> None:
    context: Final[Mapping[str, JsonValue]] = {
        **CLASSIFIER_CONTEXT,
        "rules": ["never push to main CTX-RULE-MARKER"],
        "unknown_key": "CTX-UNKNOWN-MARKER",
        "git_state": {
            "visibility": "private CTX-VISIBILITY-MARKER",
            "remote_url": "CTX-REMOTE-MARKER",
            "token": "CTX-TOKEN-MARKER",
        },
    }
    safeguards: Final[list[JsonValue]] = [{"type": "dangerous_tool_use", "classifier_context": dict(context)}]
    request_nonce: Final = nonce()
    rig.register(request_nonce, chat_backend(tool_id(request_nonce), stream=False))
    body: Final = json_body(rig.post(messages_body(CHAT_MODEL, request_nonce, safeguards=safeguards)))
    assert body["safeguard_results"] == flagged(tool_use_ids(body)), body
    prompt: Final = classifier_prompt(one_classifier_call(rig, request_nonce)[0])
    assert "CTX-RULE-MARKER" in prompt and "CTX-VISIBILITY-MARKER" in prompt, prompt
    for hidden in ("CTX-UNKNOWN-MARKER", "CTX-REMOTE-MARKER", "CTX-TOKEN-MARKER", "never-forwarded"):
        assert hidden not in prompt, (hidden, prompt)


def test_long_transcript_keeps_the_ends_and_marks_the_omission(rig: Rig) -> None:
    request_nonce: Final = nonce()
    turns: Final[list[JsonValue]] = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"TURN-{index:02d} " + "lorem ipsum " * 82}
        for index in range(39)
    ]
    rig.register(request_nonce, chat_backend(tool_id(request_nonce), stream=False))
    request: Final = {
        **auto_mode_body(CHAT_MODEL, request_nonce),
        "messages": [*turns, {"role": "user", "content": f"TURN-39 clean up {request_nonce}"}],
    }
    body: Final = json_body(rig.post(request))
    assert body["safeguard_results"] == flagged(tool_use_ids(body)), body
    prompt: Final = classifier_prompt(one_classifier_call(rig, request_nonce)[0])
    assert "earlier turns omitted" in prompt, prompt
    assert "TURN-00" in prompt and "TURN-38" in prompt and "TURN-39" in prompt, prompt
    assert "TURN-05" not in prompt, prompt
    assert len(prompt) < 40_000, len(prompt)


def text_blocks(*texts: str) -> Sequence[JsonValue]:
    return tuple({"type": "text", "text": text} for text in texts)


def later_turns(count: int, filler: int) -> Sequence[JsonValue]:
    return tuple(
        {"role": "assistant" if index % 2 == 1 else "user", "content": f"LATER-{index:02d} " + "x" * filler}
        for index in range(1, count + 1)
    )


def transcript_sent_for(rig: Rig, opening: Sequence[JsonValue], later: Sequence[JsonValue]) -> str:
    request_nonce: Final = nonce()
    rig.register(request_nonce, chat_backend(tool_id(request_nonce), stream=False))
    request: Final = {
        **auto_mode_body(CHAT_MODEL, request_nonce),
        "messages": [
            {"role": "user", "content": list(opening)},
            *later,
            {"role": "user", "content": f"NEWEST-TURN now clean up {request_nonce}"},
        ],
    }
    body: Final = json_body(rig.post(request))
    assert body["safeguard_results"] == flagged(tool_use_ids(body)), body
    prompt: Final = classifier_prompt(one_classifier_call(rig, request_nonce)[0])
    transcript: Final = prompt.split("one JSON line per turn:\n", 1)[1].split("\n\nTool calls to classify:", 1)[0]
    assert len(transcript) <= TRANSCRIPT_BUDGET, len(transcript)
    return transcript


def test_long_session_keeps_the_whole_opening_message_and_the_newest_turns(rig: Rig) -> None:
    opening: Final = text_blocks(
        "<system-reminder>REMINDER-ONE " + "r" * 1700 + "</system-reminder>",
        "<system-reminder>REMINDER-TWO " + "s" * 1700 + "</system-reminder>",
        "OPENING-REQUEST delete the stale build cache",
    )
    transcript: Final = transcript_sent_for(rig, opening, later_turns(27, 900))
    for kept in ("REMINDER-ONE", "REMINDER-TWO", "OPENING-REQUEST", OMISSION_MARKER, "LATER-27", "NEWEST-TURN"):
        assert kept in transcript, (kept, transcript)
    assert "LATER-01 " not in transcript, transcript
    assert transcript.index("OPENING-REQUEST") < transcript.index(OMISSION_MARKER) < transcript.index("LATER-27")


def test_opening_message_past_half_the_budget_is_cut_from_its_start(rig: Rig) -> None:
    pastes: Final = tuple(f"PASTE-{index:02d} " + "p" * 1800 for index in range(8))
    opening: Final = text_blocks(*pastes, "OPENING-REQUEST summarize the pasted log")
    transcript: Final = transcript_sent_for(rig, opening, later_turns(13, 900))
    for kept in ("PASTE-02", "PASTE-07", "OPENING-REQUEST", OMISSION_MARKER, "LATER-01 ", "LATER-13", "NEWEST-TURN"):
        assert kept in transcript, (kept, transcript)
    for dropped in ("PASTE-00", "PASTE-01"):
        assert dropped not in transcript, (dropped, transcript)
    assert transcript.index("OPENING-REQUEST") < transcript.index(OMISSION_MARKER) < transcript.index("LATER-01 ")


def test_non_ascii_tool_input_reaches_the_classifier_unescaped(rig: Rig) -> None:
    request_nonce: Final = nonce()
    command: Final = "rm -rf /tmp/héllo ✓ 日本"
    rig.register(request_nonce, chat_backend(tool_id(request_nonce), stream=False, arguments={"command": command}))
    body: Final = json_body(rig.post(auto_mode_body(CHAT_MODEL, request_nonce)))
    assert body["safeguard_results"] == flagged(tool_use_ids(body)), body
    prompt: Final = classifier_prompt(one_classifier_call(rig, request_nonce)[0])
    assert command in prompt, prompt
    assert "\\u00e9" not in prompt and "\\u2713" not in prompt, prompt
