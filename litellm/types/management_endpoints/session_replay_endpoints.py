from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, Field, computed_field, field_validator

MAX_SESSION_REPLAY_TURNS: Final = 200
MAX_SESSION_REPLAY_ARMS: Final = 4

SessionReplayStatus = Literal["running", "completed", "failed", "stopped"]


class SessionReplayArmSpec(BaseModel):
    label: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1)


class StartSessionReplayRequest(BaseModel):
    session_id: str = Field(min_length=1)
    arms: tuple[SessionReplayArmSpec, ...]
    judge_model: str = Field(min_length=1)
    max_turns: int = Field(default=20, ge=1, le=MAX_SESSION_REPLAY_TURNS)

    @field_validator("arms")
    @classmethod
    def _at_least_two_distinct_arms(cls, arms: tuple[SessionReplayArmSpec, ...]) -> tuple[SessionReplayArmSpec, ...]:
        if len(arms) < 2:
            raise ValueError("a replay needs at least two arms to compare")
        if len(arms) > MAX_SESSION_REPLAY_ARMS:
            raise ValueError(f"at most {MAX_SESSION_REPLAY_ARMS} arms per replay")
        labels: Final = tuple(arm.label for arm in arms)
        if len(frozenset(labels)) != len(labels):
            raise ValueError("arm labels must be unique")
        return arms


class SessionReplayTurnResponse(BaseModel):
    turn: int
    attached: bool
    served_model: str | None = None
    stop_reason: str | None = None
    text_chars: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


class SessionReplayArmResponse(BaseModel):
    label: str
    model: str
    served_models: tuple[str, ...] = ()
    turns: tuple[SessionReplayTurnResponse, ...] = ()
    final_text: str = ""

    @computed_field
    @property
    def prompt_tokens(self) -> int:
        return sum(turn.prompt_tokens for turn in self.turns)

    @computed_field
    @property
    def output_tokens(self) -> int:
        return sum(turn.output_tokens for turn in self.turns)


class SessionReplayVerdictResponse(BaseModel):
    winner: str | None = None
    confidence: float | None = None
    reasoning: str = ""
    error: str | None = None


class SessionReplayFidelityResponse(BaseModel):
    """How much of the recorded prompt actually survived into the replay.

    Spend logs truncate every stored string at MAX_STRING_LENGTH_PROMPT_IN_DB, so a replay
    runs against a smaller prompt than the recording did. Reported on every job so a result
    is never read as a clean comparison when it is not one.
    """

    truncated_strings: int = 0
    recorded_prompt_tokens: int = 0
    replayed_first_turn_prompt_tokens: int = 0

    @property
    def prompt_retained_ratio(self) -> float | None:
        if not self.recorded_prompt_tokens:
            return None
        return self.replayed_first_turn_prompt_tokens / self.recorded_prompt_tokens


class SessionReplayResult(BaseModel):
    """Everything a finished replay produced, stored verbatim as the job row's result."""

    human_asks: tuple[str, ...] = ()
    arms: tuple[SessionReplayArmResponse, ...] = ()
    verdict: SessionReplayVerdictResponse = SessionReplayVerdictResponse()
    fidelity: SessionReplayFidelityResponse = SessionReplayFidelityResponse()


class SessionReplayJobResponse(BaseModel):
    job_id: str
    session_id: str
    status: SessionReplayStatus
    judge_model: str
    max_turns: int
    source_request_id: str | None = None
    turns_completed: int = 0
    created_by: str | None = None
    created_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    human_asks: tuple[str, ...] = ()
    arms: tuple[SessionReplayArmResponse, ...] = ()
    verdict: SessionReplayVerdictResponse | None = None
    fidelity: SessionReplayFidelityResponse | None = None


class SessionReplayJobListResponse(BaseModel):
    jobs: tuple[SessionReplayJobResponse, ...] = ()
