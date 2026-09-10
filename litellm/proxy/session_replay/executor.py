"""Replay one recorded session against several model arms and judge the end results.

The environment track (recorded user and tool_result turns) is fixed; each arm generates
its own assistant turns and diverges from the recording freely. What is judged is the
final answer each arm reached, against the human asks the recording actually contained.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict

from litellm.proxy.session_replay.transcript import (
    ContentBlock,
    RecordedMessage,
    ReplayTranscript,
    attach_turn,
    tool_use_ids,
)
from litellm.types.management_endpoints.session_replay_endpoints import (
    SessionReplayArmResponse,
    SessionReplayArmSpec,
    SessionReplayFidelityResponse,
    SessionReplayResult,
    SessionReplayTurnResponse,
    SessionReplayVerdictResponse,
)

JUDGE_SYSTEM_PROMPT: Final = (
    "You are grading candidate assistants that were each given the same task and allowed to work "
    "independently. Judge only how well the final answer serves the user's request: correctness, "
    "completeness, and whether it actually did the work rather than describing it. Ignore length "
    "and style. Respond with JSON only, no prose: "
    '{"winner": "<label>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}. '
    'Use the exact label "tie" if neither is better.'
)

_MAX_JUDGE_EXCERPT_CHARS: Final = 6000


class Usage(BaseModel):
    model_config = ConfigDict(extra="allow")

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens


class ArmResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    stop_reason: str | None = None
    content: tuple[ContentBlock, ...] = ()
    usage: Usage = Usage()

    def text(self) -> str:
        return "".join(block.text or "" for block in self.content if block.type == "text")


class ModelCaller(Protocol):
    async def __call__(self, request: Mapping[str, object]) -> Mapping[str, object]: ...


class JudgeCaller(Protocol):
    async def __call__(self, system_prompt: str, user_prompt: str) -> str: ...


class ProgressSink(Protocol):
    async def __call__(self, turns_completed: int) -> None: ...


async def _noop_progress(turns_completed: int) -> None:
    return None


def _skipped_turn(index: int) -> SessionReplayTurnResponse:
    return SessionReplayTurnResponse(turn=index, attached=False)


def _failed_turn(index: int, error: str) -> SessionReplayTurnResponse:
    return SessionReplayTurnResponse(turn=index, attached=True, error=error[:500])


def _request_body(
    transcript: ReplayTranscript,
    arm: SessionReplayArmSpec,
    messages: Sequence[RecordedMessage],
    session_id: str,
) -> Mapping[str, object]:
    return {  # mutable-ok: Anthropic Messages wire body
        **{key: value for key, value in transcript.sampling_params},  # mutable-ok: Anthropic Messages wire body
        "model": arm.model,
        "messages": [  # mutable-ok: Anthropic Messages wire body
            message.model_dump(exclude_none=True) for message in messages
        ],  # mutable-ok: Anthropic Messages wire body
        "system": [  # mutable-ok: Anthropic Messages wire body
            block.model_dump(exclude_none=True) for block in transcript.system
        ],  # mutable-ok: Anthropic Messages wire body
        "tools": [  # mutable-ok: Anthropic Messages wire body
            tool.model_dump(exclude_none=True) for tool in transcript.tools
        ],  # mutable-ok: Anthropic Messages wire body
        "stream": False,
        "litellm_session_id": session_id,
    }


async def run_arm(
    transcript: ReplayTranscript,
    arm: SessionReplayArmSpec,
    max_turns: int,
    session_id: str,
    call_model: ModelCaller,
    on_progress: ProgressSink = _noop_progress,
) -> SessionReplayArmResponse:
    """Walk the recorded user turns, letting the arm build its own assistant trajectory.

    A turn that fails is recorded and the walk continues: one provider error should cost a
    turn, not the whole job.
    """
    history: Final[list[RecordedMessage]] = []  # mutable-ok: the arm's trajectory is built turn by turn
    turns: Final[list[SessionReplayTurnResponse]] = []  # mutable-ok: one append per recorded turn
    pending: tuple[str, ...] = ()  # rebind-ok: carries the previous turn's tool_use ids forward
    final_text = ""  # rebind-ok: the last turn that produced text wins
    for index, recorded_turn in enumerate(transcript.user_turns[:max_turns]):
        attached = attach_turn(recorded_turn, pending)
        if attached is None:
            turns.append(_skipped_turn(index))
            await on_progress(len(turns))
            continue
        history.append(attached)
        try:
            raw = await call_model(_request_body(transcript, arm, history, session_id))
            response = ArmResponse.model_validate(raw)
        except Exception as exc:  # noqa: BLE001  # a failed turn is recorded, never fatal to the job
            history.pop()
            turns.append(_failed_turn(index, str(exc)))
            await on_progress(len(turns))
            continue
        history.append(RecordedMessage(role="assistant", content=response.content))
        pending = tool_use_ids(response.content)
        final_text = response.text() or final_text
        turns.append(
            SessionReplayTurnResponse(
                turn=index,
                attached=True,
                served_model=response.model,
                stop_reason=response.stop_reason,
                text_chars=len(response.text()),
                tool_calls=len(pending),
                prompt_tokens=response.usage.prompt_tokens(),
                output_tokens=response.usage.output_tokens,
            )
        )
        await on_progress(len(turns))
    return SessionReplayArmResponse(
        label=arm.label,
        model=arm.model,
        served_models=tuple(dict.fromkeys(turn.served_model for turn in turns if turn.served_model)),
        turns=tuple(turns),
        final_text=final_text,
    )


def _judge_prompt(human_asks: Sequence[str], runs: Sequence[SessionReplayArmResponse], order: Sequence[int]) -> str:
    asks: Final = "\n\n".join(ask[:_MAX_JUDGE_EXCERPT_CHARS] for ask in human_asks) or "(no human turn recorded)"
    answers: Final = "\n\n".join(
        f"--- CANDIDATE {chr(ord('A') + position)} ---\n{runs[arm_index].final_text[:_MAX_JUDGE_EXCERPT_CHARS]}"
        for position, arm_index in enumerate(order)
    )
    return f"THE USER ASKED:\n{asks}\n\n{answers}"


def _unmask(position: str, order: Sequence[int], runs: Sequence[SessionReplayArmResponse]) -> str | None:
    """Map the judge's blind A/B label back to an arm, or give up.

    The judge writes this string, so it can be anything; anything unrecognized is a recorded
    verdict error rather than an exception, since every arm has already been billed by here.
    """
    label: Final = position.strip().upper()
    if len(label) != 1:
        return None
    index: Final = ord(label) - ord("A")
    if index < 0 or index >= len(order):
        return None
    return runs[order[index]].label


def _parse_verdict(raw: str) -> tuple[str, float | None, str] | None:
    start: Final = raw.find("{")
    end: Final = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload: Final = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    winner: Final = payload.get("winner")
    if not isinstance(winner, str):
        return None
    confidence: Final = payload.get("confidence")
    reasoning: Final = payload.get("reasoning")
    return (
        winner,
        float(confidence) if isinstance(confidence, (int, float)) else None,
        reasoning if isinstance(reasoning, str) else "",
    )


async def judge_runs(
    human_asks: Sequence[str],
    runs: Sequence[SessionReplayArmResponse],
    call_judge: JudgeCaller,
    rng: random.Random,
) -> SessionReplayVerdictResponse:
    """Blind pairwise judging: arms are presented as A/B in randomized order and unmasked after."""
    if len(runs) < 2:
        return SessionReplayVerdictResponse(error="need at least two arms to judge")
    order: Final = tuple(rng.sample(range(len(runs)), len(runs)))
    try:
        raw: Final = await call_judge(JUDGE_SYSTEM_PROMPT, _judge_prompt(human_asks, runs, order))
    except Exception as exc:  # noqa: BLE001  # a failed judge is a recorded outcome, not a failed job
        return SessionReplayVerdictResponse(error=str(exc)[:500])
    parsed: Final = _parse_verdict(raw)
    if parsed is None:
        return SessionReplayVerdictResponse(error=f"unparseable judge response: {raw[:200]}")
    position, confidence, reasoning = parsed
    if position.strip().casefold() == "tie":
        return SessionReplayVerdictResponse(winner="tie", confidence=confidence, reasoning=reasoning)
    unmasked: Final = _unmask(position, order, runs)
    if unmasked is None:
        return SessionReplayVerdictResponse(reasoning=reasoning, error=f"judge named unknown label {position}")
    return SessionReplayVerdictResponse(winner=unmasked, confidence=confidence, reasoning=reasoning)


async def _run_arms(
    transcript: ReplayTranscript,
    arms: Sequence[SessionReplayArmSpec],
    max_turns: int,
    session_id: str,
    call_model: ModelCaller,
    on_progress: ProgressSink,
) -> tuple[SessionReplayArmResponse, ...]:
    """Progress counts turns across the whole job, not per arm, so the heartbeat a stale-job
    reaper reads never goes backwards when the next arm starts."""
    runs: Final[list[SessionReplayArmResponse]] = []  # mutable-ok: one append per finished arm
    for arm in arms:
        done_before = sum(len(run.turns) for run in runs)

        async def cumulative(turns_completed: int, offset: int = done_before) -> None:
            await on_progress(offset + turns_completed)

        runs.append(await run_arm(transcript, arm, max_turns, f"{session_id}-{arm.label}", call_model, cumulative))
    return tuple(runs)


async def replay_session(
    transcript: ReplayTranscript,
    arms: Sequence[SessionReplayArmSpec],
    max_turns: int,
    session_id: str,
    recorded_prompt_tokens: int,
    call_model: ModelCaller,
    call_judge: JudgeCaller,
    rng: random.Random,
    on_progress: ProgressSink = _noop_progress,
) -> SessionReplayResult:
    runs: Final = await _run_arms(transcript, arms, max_turns, session_id, call_model, on_progress)
    first_turn_tokens: Final = next((turn.prompt_tokens for run in runs for turn in run.turns if turn.prompt_tokens), 0)
    return SessionReplayResult(
        human_asks=transcript.human_asks,
        arms=runs,
        verdict=await judge_runs(transcript.human_asks, runs, call_judge, rng),
        fidelity=SessionReplayFidelityResponse(
            truncated_strings=transcript.truncated_strings,
            recorded_prompt_tokens=recorded_prompt_tokens,
            replayed_first_turn_prompt_tokens=first_turn_tokens,
        ),
    )
