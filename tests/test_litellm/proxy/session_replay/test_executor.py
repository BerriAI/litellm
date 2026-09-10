import random

import pytest

from litellm.proxy.session_replay.executor import judge_runs, replay_session, run_arm
from litellm.proxy.session_replay.transcript import RecordedBody, build_transcript
from litellm.types.management_endpoints.session_replay_endpoints import SessionReplayArmSpec

TWO_HUMAN_TURNS = {
    "model": "recorded-router",
    "max_tokens": 1024,
    "system": [{"type": "text", "text": "be helpful"}],
    "tools": [{"name": "Read", "description": "read", "input_schema": {"type": "object", "properties": {}}}],
    "messages": [
        {"role": "user", "content": [{"type": "text", "text": "first ask"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_REC", "name": "Read"}]},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_REC", "content": "recorded output"}],
        },
        {"role": "user", "content": [{"type": "text", "text": "second ask"}]},
    ],
}


def _transcript(body=None):
    return build_transcript(RecordedBody.model_validate(body or TWO_HUMAN_TURNS))


def _text_response(text, model="served-model"):
    return {
        "model": model,
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _tool_response(tool_use_id, model="served-model"):
    return {
        "model": model,
        "stop_reason": "tool_use",
        "content": [{"type": "tool_use", "id": tool_use_id, "name": "Read", "input": {}}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


class _RecordingCaller:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def __call__(self, request):
        self.requests.append(request)
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


@pytest.mark.asyncio
async def test_arm_trajectory_accumulates_its_own_assistant_turns():
    """The arm's own outputs, not the recording's, must be what later turns see: that is
    what makes this an end-to-end replay rather than per-turn scoring."""
    caller = _RecordingCaller([_tool_response("toolu_ARM"), _text_response("after tool"), _text_response("final")])

    run = await run_arm(
        _transcript(), SessionReplayArmSpec(label="a", model="m"), max_turns=10, session_id="s", call_model=caller
    )

    assert run.final_text == "final"
    assert len(caller.requests) == 3
    roles_on_last_call = [message["role"] for message in caller.requests[-1]["messages"]]
    assert roles_on_last_call == ["user", "assistant", "user", "assistant", "user"]


@pytest.mark.asyncio
async def test_recorded_tool_result_is_rebound_to_the_arms_tool_use_id():
    caller = _RecordingCaller([_tool_response("toolu_ARM"), _text_response("ok"), _text_response("done")])

    await run_arm(
        _transcript(), SessionReplayArmSpec(label="a", model="m"), max_turns=10, session_id="s", call_model=caller
    )

    second_call_last_message = caller.requests[1]["messages"][-1]
    assert [block["tool_use_id"] for block in second_call_last_message["content"]] == ["toolu_ARM"]


@pytest.mark.asyncio
async def test_turn_whose_tool_result_cannot_attach_is_skipped_without_calling_the_model():
    caller = _RecordingCaller([_text_response("no tools here"), _text_response("second answer")])

    run = await run_arm(
        _transcript(), SessionReplayArmSpec(label="a", model="m"), max_turns=10, session_id="s", call_model=caller
    )

    assert [turn.attached for turn in run.turns] == [True, False, True]
    assert len(caller.requests) == 2


@pytest.mark.asyncio
async def test_a_failing_turn_is_recorded_and_the_walk_continues():
    """One provider error should cost a turn, not the whole job."""
    caller = _RecordingCaller([RuntimeError("upstream 401"), _text_response("recovered")])

    run = await run_arm(
        _transcript(), SessionReplayArmSpec(label="a", model="m"), max_turns=10, session_id="s", call_model=caller
    )

    assert run.turns[0].error is not None and "upstream 401" in run.turns[0].error
    assert run.final_text == "recovered"


@pytest.mark.asyncio
async def test_max_turns_bounds_the_number_of_billable_calls():
    caller = _RecordingCaller([_text_response("one"), _text_response("two"), _text_response("three")])

    await run_arm(
        _transcript(), SessionReplayArmSpec(label="a", model="m"), max_turns=1, session_id="s", call_model=caller
    )

    assert len(caller.requests) == 1


@pytest.mark.asyncio
async def test_progress_is_reported_after_every_turn():
    """A job whose pod dies is detected by a stale heartbeat, so progress must advance
    per turn rather than per arm."""
    caller = _RecordingCaller([_text_response("one"), _text_response("two")])
    seen = []

    async def on_progress(turns_completed):
        seen.append(turns_completed)

    await run_arm(
        _transcript(),
        SessionReplayArmSpec(label="a", model="m"),
        max_turns=10,
        session_id="s",
        call_model=caller,
        on_progress=on_progress,
    )

    assert seen == [1, 2, 3]


@pytest.mark.asyncio
async def test_judge_verdict_is_unmasked_back_to_the_winning_arm_label():
    """The judge sees anonymous A/B in randomized order, so a verdict read without
    unmasking would attribute wins to whichever arm happened to be shown first."""
    runs = []
    for label in ("alpha", "beta"):
        caller = _RecordingCaller([_text_response(f"{label} answer"), _text_response(f"{label} final")])
        runs.append(
            await run_arm(
                _transcript(),
                SessionReplayArmSpec(label=label, model="m"),
                max_turns=10,
                session_id="s",
                call_model=caller,
            )
        )

    async def call_judge(system_prompt, user_prompt):
        shown_first = "alpha" if "alpha final" in user_prompt.split("CANDIDATE B")[0] else "beta"
        assert shown_first in ("alpha", "beta")
        return '{"winner": "A", "confidence": 0.9, "reasoning": "clearer"}'

    for seed in range(6):
        verdict = await judge_runs(("ask",), runs, call_judge, random.Random(seed))
        order = random.Random(seed).sample(range(2), 2)
        assert verdict.winner == runs[order[0]].label


@pytest.mark.asyncio
async def test_unparseable_judge_response_is_a_recorded_outcome_not_an_exception():
    caller = _RecordingCaller([_text_response("x"), _text_response("y")])
    run = await run_arm(
        _transcript(), SessionReplayArmSpec(label="a", model="m"), max_turns=10, session_id="s", call_model=caller
    )

    async def call_judge(system_prompt, user_prompt):
        return "I prefer the first one"

    verdict = await judge_runs(("ask",), (run, run), call_judge, random.Random(0))

    assert verdict.winner is None
    assert verdict.error is not None


@pytest.mark.asyncio
async def test_judge_failure_does_not_fail_the_job():
    caller = _RecordingCaller([_text_response("x"), _text_response("y")])
    run = await run_arm(
        _transcript(), SessionReplayArmSpec(label="a", model="m"), max_turns=10, session_id="s", call_model=caller
    )

    async def call_judge(system_prompt, user_prompt):
        raise RuntimeError("judge model unavailable")

    verdict = await judge_runs(("ask",), (run, run), call_judge, random.Random(0))

    assert verdict.winner is None
    assert verdict.error is not None and "judge model unavailable" in verdict.error


@pytest.mark.asyncio
async def test_fidelity_reports_the_gap_between_recorded_and_replayed_prompt():
    """Spend logs truncate stored prompts, so a replay runs on a smaller prompt than the
    recording did and a verdict must never be read as a clean comparison."""

    async def call_model(request):
        return _text_response("answer")

    async def call_judge(system_prompt, user_prompt):
        return '{"winner": "tie", "confidence": 0.5, "reasoning": "same"}'

    outcome = await replay_session(
        transcript=_transcript(),
        arms=(SessionReplayArmSpec(label="a", model="m"), SessionReplayArmSpec(label="b", model="n")),
        max_turns=10,
        session_id="s",
        recorded_prompt_tokens=41920,
        call_model=call_model,
        call_judge=call_judge,
        rng=random.Random(0),
    )

    assert outcome.fidelity.recorded_prompt_tokens == 41920
    assert outcome.fidelity.replayed_first_turn_prompt_tokens == 10
    assert outcome.verdict.winner == "tie"


@pytest.mark.asyncio
async def test_each_arm_replays_under_its_own_session_id():
    """Arms share one recorded session, but session affinity would pin them to one tier
    if they also shared a session id at replay time."""
    seen = []

    async def call_model(request):
        seen.append(request["litellm_session_id"])
        return _text_response("answer")

    async def call_judge(system_prompt, user_prompt):
        return '{"winner": "tie", "confidence": 0.5, "reasoning": "same"}'

    await replay_session(
        transcript=_transcript(),
        arms=(SessionReplayArmSpec(label="a", model="m"), SessionReplayArmSpec(label="b", model="n")),
        max_turns=1,
        session_id="orig",
        recorded_prompt_tokens=1,
        call_model=call_model,
        call_judge=call_judge,
        rng=random.Random(0),
    )

    assert len(set(seen)) == 2
    assert "orig" not in seen


@pytest.mark.asyncio
async def test_progress_counts_turns_across_arms_not_per_arm():
    """A stale-job reaper reads this counter as a heartbeat, so restarting it at 1 when the
    second arm begins makes a live job look like it went backwards."""

    async def call_model(request):
        return _text_response("answer")

    async def call_judge(system_prompt, user_prompt):
        return '{"winner": "tie", "confidence": 0.5, "reasoning": "same"}'

    seen = []

    async def on_progress(turns_completed):
        seen.append(turns_completed)

    await replay_session(
        transcript=_transcript(),
        arms=(SessionReplayArmSpec(label="a", model="m"), SessionReplayArmSpec(label="b", model="n")),
        max_turns=10,
        session_id="s",
        recorded_prompt_tokens=1,
        call_model=call_model,
        call_judge=call_judge,
        rng=random.Random(0),
        on_progress=on_progress,
    )

    assert seen == sorted(seen)
    assert seen[-1] == len(seen)


@pytest.mark.asyncio
async def test_arm_token_totals_survive_serialization():
    """They are derived from the turns, and the job row stores the serialized result, so a
    plain property would drop them from every stored and returned job."""
    caller = _RecordingCaller([_text_response("one"), _text_response("two")])

    run = await run_arm(
        _transcript(), SessionReplayArmSpec(label="a", model="m"), max_turns=10, session_id="s", call_model=caller
    )

    assert run.model_dump()["prompt_tokens"] == 20
    assert run.model_dump()["output_tokens"] == 10


@pytest.mark.asyncio
async def test_failed_turn_does_not_leave_an_unanswered_user_message_in_history():
    """A failed call produces no assistant reply, so keeping its user turn would send two
    consecutive user messages on the next turn and rebind results onto ids no longer pending."""
    caller = _RecordingCaller([_tool_response("toolu_ARM"), RuntimeError("upstream 500"), _text_response("ok")])

    await run_arm(_transcript(), SessionReplayArmSpec(label="a", model="m"), max_turns=10, session_id="s", call_model=caller)

    for request in caller.requests:
        roles = [message["role"] for message in request["messages"]]
        assert all(a != b for a, b in zip(roles, roles[1:])), roles


@pytest.mark.asyncio
@pytest.mark.parametrize("winner", ["Tie", "TIE", "CANDIDATE A", "", "neither", "A "])
async def test_any_judge_winner_string_is_a_recorded_outcome_never_an_exception(winner):
    """Every arm is already billed by the time the judge answers, so an unparseable winner must
    not take the whole job down with it."""
    caller = _RecordingCaller([_text_response("x"), _text_response("y")])
    run = await run_arm(
        _transcript(), SessionReplayArmSpec(label="a", model="m"), max_turns=10, session_id="s", call_model=caller
    )

    async def call_judge(system_prompt, user_prompt):
        return '{"winner": "%s", "confidence": 0.5, "reasoning": "r"}' % winner

    verdict = await judge_runs(("ask",), (run, run), call_judge, random.Random(0))

    assert verdict.winner in (None, "tie", "a")


@pytest.mark.asyncio
async def test_tie_is_recognised_whatever_case_the_judge_used():
    caller = _RecordingCaller([_text_response("x"), _text_response("y")])
    run = await run_arm(
        _transcript(), SessionReplayArmSpec(label="a", model="m"), max_turns=10, session_id="s", call_model=caller
    )

    async def call_judge(system_prompt, user_prompt):
        return '{"winner": "Tie", "confidence": 0.4, "reasoning": "same"}'

    verdict = await judge_runs(("ask",), (run, run), call_judge, random.Random(0))

    assert verdict.winner == "tie"
    assert verdict.error is None
