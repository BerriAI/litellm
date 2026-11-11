from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class FeedbackLabelName(StrEnum):
    REPHRASING = "NEG_1"
    MAKE_AWARE_WITH_CORRECTION = "NEG_2"
    MAKE_AWARE_WITHOUT_CORRECTION = "NEG_3"
    ASK_FOR_CLARIFICATION = "NEG_4"
    POSITIVE = "POS"
    NO_FEEDBACK = "NEU"


CORRECTIVE_LABELS: frozenset[FeedbackLabelName] = frozenset(
    {
        FeedbackLabelName.REPHRASING,
        FeedbackLabelName.MAKE_AWARE_WITH_CORRECTION,
        FeedbackLabelName.MAKE_AWARE_WITHOUT_CORRECTION,
    }
)


class FeedbackTurn(Frozen):
    session_id: str
    source: str
    model_id: str
    group_key: str
    turn_idx: int
    request: str
    response: str
    next_user_turn: str | None


class GoldFeedback(Frozen):
    session_id: str
    turn_idx: int
    label: FeedbackLabelName


class ChatMessage(Frozen):
    role: str
    content: str


class Conversation(Frozen):
    session_id: str
    messages: tuple[ChatMessage, ...]


class TurnVerdict(Frozen):
    turn: int
    label: FeedbackLabelName
    rationale: str


class ConversationVerdict(Frozen):
    labels: tuple[TurnVerdict, ...]


class FeedbackLabel(Frozen):
    kind: Literal["label"] = "label"
    session_id: str
    turn_idx: int
    label: FeedbackLabelName
    rationale: str
    labeler_model: str
    prompt_version: str
    cost_usd: float | None


class LabelFailure(Frozen):
    kind: Literal["failure"] = "failure"
    session_id: str
    turn_idx: int
    reason: str
    labeler_model: str
    prompt_version: str
    cost_usd: float | None


LabelOutcome = FeedbackLabel | LabelFailure


class CorrectionStats(Frozen):
    model_id: str
    group_key: str
    correction_count: int
    correction_observations: int
    correction_rate: float | None
