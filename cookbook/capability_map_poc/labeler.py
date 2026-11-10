from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Final, Literal, Protocol

import openai
from pydantic import ValidationError

from .corrections import TurnKey, turn_key
from .schemas import FeedbackLabel, FeedbackTurn, Frozen, LabelerVerdict, LabelFailure, LabelOutcome

PROMPT_VERSION: Final = "emnlp2025-taxonomy-v1"

# Category definitions are quoted from Liu, Zhang and Choi, "User Feedback in Human-LLM Dialogues"
# (EMNLP 2025, Section 2 and Appendix B/H.2), which follows the Don-Yehiya et al. (2024) ontology.
SYSTEM_PROMPT: Final = """You act as a human annotator identifying user feedback for an Assistant.
You see the user's previous request, the Assistant's response, and the user's next turn.
Label only the user's next turn, as feedback on that Assistant response.

Labels:
- NEG_1 (Rephrasing / Repeat or Rephrase): the user rephrased their prior request to try and elicit a better
  response. Does the user repeat or rephrase their concern? Examples for "yes": "By house, I mean apartments,
  not condo", "Actually, I wanted". Example for "no": "Thank you".
- NEG_2 (Make Aware with Correction): the user's response indicates that the model's prior response was wrong
  and additionally provides instruction on how to correct it.
- NEG_3 (Make Aware without Correction): the user's response simply indicates that the model's prior response
  was wrong.
- NEG_4 (Ask for Clarification): the user asks the Assistant to provide additional information that was missing
  from its prior response.
- POS (Positive Feedback): the user praises the Assistant's response, e.g. "Great job!".
- NEU (No Feedback): the next turn carries no feedback on the response, e.g. a new request or a follow-up.

If several negative labels apply, pick by priority: NEG_2, then NEG_3, then NEG_4, then NEG_1.
Positive and negative labels never overlap.
Set "confident" to false when the turn is ambiguous between labels or you cannot tell whether it is feedback.
Answer with JSON only: {"label": "<label>", "confident": <true|false>, "rationale": "<one short sentence>"}"""


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

    def complete(self, system: str, user: str) -> Completion: ...


class GatewayCompleter:
    def __init__(self, client: openai.OpenAI, model: str) -> None:
        self._client: Final = client
        self._model: Final = model

    @property
    def model(self) -> str:
        return self._model

    def complete(self, system: str, user: str) -> Completion:
        try:
            raw: Final = self._client.chat.completions.with_raw_response.create(
                model=self._model,
                temperature=0,
                max_tokens=300,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
        except openai.OpenAIError as error:
            return CompletionError(reason=f"{type(error).__name__}: {error}")
        cost: Final = raw.headers.get("x-litellm-response-cost")
        return CompletionOk(text=raw.parse().choices[0].message.content or "", cost_usd=float(cost) if cost else None)


def render_turn(turn: FeedbackTurn) -> str:
    return (
        f"Previous user request:\n{turn.request}\n\n"
        f"Assistant response:\n{turn.response}\n\n"
        f"User's next turn:\n{turn.next_user_turn}"
    )


def label_turn(turn: FeedbackTurn, completer: Completer) -> LabelOutcome:
    completion: Final = completer.complete(SYSTEM_PROMPT, render_turn(turn))
    match completion:
        case CompletionError(reason=reason):
            return _failure(turn, completer, reason, None)
        case CompletionOk(text=text, cost_usd=cost):
            try:
                verdict: Final = LabelerVerdict.model_validate_json(_strip_fence(text))
            except ValidationError as error:
                return _failure(turn, completer, f"unparseable labeler output: {error.errors()[0]['msg']}", cost)
            return FeedbackLabel(
                session_id=turn.session_id,
                turn_idx=turn.turn_idx,
                label=verdict.label,
                confident=verdict.confident,
                rationale=verdict.rationale,
                labeler_model=completer.model,
                prompt_version=PROMPT_VERSION,
                cost_usd=cost,
            )


def label_turns(
    turns: Iterable[FeedbackTurn],
    completer: Completer,
    cached: Mapping[TurnKey, LabelOutcome],
    workers: int,
    on_label: Callable[[LabelOutcome], None],
) -> tuple[LabelOutcome, ...]:
    pending: Final = tuple(t for t in turns if not _reusable(cached.get(turn_key(t.session_id, t.turn_idx)), completer))

    def run(turn: FeedbackTurn) -> LabelOutcome:
        outcome: Final = label_turn(turn, completer)
        on_label(outcome)
        return outcome

    with ThreadPoolExecutor(max_workers=workers) as pool:
        fresh: Final = tuple(pool.map(run, pending))
    return tuple({**cached, **{turn_key(o.session_id, o.turn_idx): o for o in fresh}}.values())


def _reusable(outcome: LabelOutcome | None, completer: Completer) -> bool:
    return (
        isinstance(outcome, FeedbackLabel)
        and outcome.prompt_version == PROMPT_VERSION
        and outcome.labeler_model == completer.model
    )


def _strip_fence(text: str) -> str:
    stripped: Final = text.strip()
    if not stripped.startswith("```"):
        return stripped
    return stripped.removeprefix("```json").removeprefix("```").removesuffix("```").strip()


def _failure(turn: FeedbackTurn, completer: Completer, reason: str, cost: float | None) -> LabelFailure:
    return LabelFailure(
        session_id=turn.session_id,
        turn_idx=turn.turn_idx,
        reason=reason,
        labeler_model=completer.model,
        prompt_version=PROMPT_VERSION,
        cost_usd=cost,
    )
