from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from itertools import accumulate, chain
from typing import Final, Literal, Protocol

import openai
from openai.types.shared_params import ResponseFormatJSONSchema
from pydantic import ValidationError

from .corrections import TurnKey, turn_key
from .schemas import (
    Conversation,
    ConversationVerdict,
    FeedbackLabel,
    FeedbackLabelName,
    Frozen,
    LabelFailure,
    LabelOutcome,
)

PROMPT_VERSION: Final = "emnlp2025-conversation-v2"

# Category definitions are quoted from Liu, Zhang and Choi, "User Feedback in Human-LLM Dialogues"
# (EMNLP 2025, Section 2 and Appendix B/H.2), which follows the Don-Yehiya et al. (2024) ontology.
# The NEG_1 examples are the paper's, the other examples were written for this POC.
SYSTEM_PROMPT: Final = """You act as a human annotator identifying user feedback for an Assistant.
You see a whole conversation. User turns are numbered.
For every user turn from 2 onward, label it as feedback on the Assistant response directly before it.
Use the whole conversation, including later turns, to understand what the user meant.

Labels:
- NEG_1 (Rephrasing / Repeat or Rephrase): the user rephrased their prior request to try and elicit a better
  response. Does the user repeat or rephrase their concern? Examples for "yes": "By house, I mean apartments,
  not condo", "Actually, I wanted". Example for "no": "Thank you".
- NEG_2 (Make Aware with Correction): the user's response indicates that the model's prior response was wrong
  and additionally provides instruction on how to correct it. Example: "That's wrong, the capital is Canberra,
  please fix the table".
- NEG_3 (Make Aware without Correction): the user's response simply indicates that the model's prior response
  was wrong. Example: "That code still doesn't work".
- NEG_4 (Ask for Clarification): the user asks the Assistant to provide additional information that was missing
  from its prior response. Example: "What does the second parameter do?"
- POS (Positive Feedback): the user praises the Assistant's response, e.g. "Great job!".
- NEU (No Feedback): the turn carries no feedback on the response, e.g. a new request or a follow-up.
  Example: "Now write one about the sea".

If several negative labels apply, pick by priority: NEG_2, then NEG_3, then NEG_4, then NEG_1.
Positive and negative labels never overlap.
Return exactly one entry per user turn from 2 onward, with a one short sentence rationale."""

RESPONSE_FORMAT: Final[ResponseFormatJSONSchema] = {
    "type": "json_schema",
    "json_schema": {
        "name": "feedback_labels",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "labels": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "turn": {"type": "integer"},
                            "label": {"type": "string", "enum": [label.value for label in FeedbackLabelName]},
                            "rationale": {"type": "string"},
                        },
                        "required": ["turn", "label", "rationale"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["labels"],
            "additionalProperties": False,
        },
    },
}
TOKENS_PER_TURN: Final = 120
BASE_TOKENS: Final = 200


class CompletionOk(Frozen):
    kind: Literal["ok"] = "ok"
    text: str
    cost_usd: float | None


class CompletionError(Frozen):
    kind: Literal["error"] = "error"
    reason: str


Completion = CompletionOk | CompletionError


class Completer(Protocol):
    @property
    def model(self) -> str: ...

    def complete(self, system: str, user: str, max_tokens: int) -> Completion: ...


class GatewayCompleter:
    def __init__(self, client: openai.OpenAI, model: str) -> None:
        self._client: Final = client
        self._model: Final = model

    @property
    def model(self) -> str:
        return self._model

    def complete(self, system: str, user: str, max_tokens: int) -> Completion:
        try:
            raw: Final = self._client.chat.completions.with_raw_response.create(
                model=self._model,
                temperature=0,
                max_tokens=max_tokens,
                response_format=RESPONSE_FORMAT,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
        except openai.OpenAIError as error:
            return CompletionError(reason=f"{type(error).__name__}: {error}")
        cost: Final = raw.headers.get("x-litellm-response-cost")
        return CompletionOk(text=raw.parse().choices[0].message.content or "", cost_usd=float(cost) if cost else None)


def labeled_turns(conversation: Conversation) -> tuple[int, ...]:
    return tuple(range(2, sum(m.role == "user" for m in conversation.messages) + 1))


def render_conversation(conversation: Conversation) -> str:
    user_numbers: Final = accumulate(int(m.role == "user") for m in conversation.messages)
    return "\n\n".join(
        f"User turn {number}:\n{m.content}" if m.role == "user" else f"Assistant:\n{m.content}"
        for m, number in zip(conversation.messages, user_numbers)
    )


def label_conversation(conversation: Conversation, completer: Completer) -> tuple[LabelOutcome, ...]:
    first: Final = _attempt(conversation, completer)
    if not _failed(first):
        return first
    second: Final = _attempt(conversation, completer)
    retry_cost: Final = sum(o.cost_usd or 0.0 for o in first)
    return tuple(o.model_copy(update={"cost_usd": (o.cost_usd or 0.0) + retry_cost / len(second)}) for o in second)


def label_conversations(
    conversations: Iterable[Conversation],
    completer: Completer,
    cached: Mapping[TurnKey, LabelOutcome],
    workers: int,
    on_labels: Callable[[tuple[LabelOutcome, ...]], None],
) -> tuple[LabelOutcome, ...]:
    pending: Final = tuple(c for c in conversations if not _reusable(c, cached, completer))

    def run(conversation: Conversation) -> tuple[LabelOutcome, ...]:
        outcomes: Final = label_conversation(conversation, completer)
        on_labels(outcomes)
        return outcomes

    with ThreadPoolExecutor(max_workers=workers) as pool:
        fresh: Final = tuple(chain.from_iterable(pool.map(run, pending)))
    return tuple({**cached, **{turn_key(o.session_id, o.turn_idx): o for o in fresh}}.values())


def _attempt(conversation: Conversation, completer: Completer) -> tuple[LabelOutcome, ...]:
    expected: Final = labeled_turns(conversation)
    if not expected:
        return ()
    completion: Final = completer.complete(
        SYSTEM_PROMPT, render_conversation(conversation), BASE_TOKENS + TOKENS_PER_TURN * len(expected)
    )
    match completion:
        case CompletionError(reason=reason):
            return _failures(conversation, completer, reason, None)
        case CompletionOk(text=text, cost_usd=cost):
            try:
                verdict: Final = ConversationVerdict.model_validate_json(_strip_fence(text))
            except ValidationError as error:
                return _failures(
                    conversation, completer, f"unparseable labeler output: {error.errors()[0]['msg']}", cost
                )
            got: Final = tuple(sorted(v.turn for v in verdict.labels))
            if got != expected:
                return _failures(conversation, completer, f"labeled turns {got}, expected {expected}", cost)
            share: Final = cost / len(expected) if cost is not None else None
            return tuple(
                FeedbackLabel(
                    session_id=conversation.session_id,
                    turn_idx=v.turn,
                    label=v.label,
                    rationale=v.rationale,
                    labeler_model=completer.model,
                    prompt_version=PROMPT_VERSION,
                    cost_usd=share,
                )
                for v in verdict.labels
            )


def _failed(outcomes: tuple[LabelOutcome, ...]) -> bool:
    return any(isinstance(o, LabelFailure) for o in outcomes)


def _reusable(conversation: Conversation, cached: Mapping[TurnKey, LabelOutcome], completer: Completer) -> bool:
    outcomes: Final = tuple(cached.get(turn_key(conversation.session_id, t)) for t in labeled_turns(conversation))
    return all(
        isinstance(o, FeedbackLabel) and o.prompt_version == PROMPT_VERSION and o.labeler_model == completer.model
        for o in outcomes
    )


def _strip_fence(text: str) -> str:
    stripped: Final = text.strip()
    if not stripped.startswith("```"):
        return stripped
    return stripped.removeprefix("```json").removeprefix("```").removesuffix("```").strip()


def _failures(
    conversation: Conversation, completer: Completer, reason: str, cost: float | None
) -> tuple[LabelFailure, ...]:
    expected: Final = labeled_turns(conversation)
    share: Final = cost / len(expected) if cost is not None else None
    return tuple(
        LabelFailure(
            session_id=conversation.session_id,
            turn_idx=t,
            reason=reason,
            labeler_model=completer.model,
            prompt_version=PROMPT_VERSION,
            cost_usd=share,
        )
        for t in expected
    )
