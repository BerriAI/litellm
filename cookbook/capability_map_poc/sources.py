from collections.abc import Iterable, Sequence
from itertools import chain, pairwise
from typing import Final, Literal

from pydantic import TypeAdapter

from .schemas import ChatMessage, Conversation, FeedbackLabelName, FeedbackTurn, Frozen, GoldFeedback

EMNLP_SOURCE: Final = "emnlp"
EMNLP_MODEL_ID: Final = "logged-chat-model"

EmnlpSplit = Literal["dense", "sparse"]


class EmnlpRow(Frozen):
    model_config = Frozen.model_config | {"extra": "ignore"}

    conversation_id: str
    dataset_source: str
    conversation: tuple[ChatMessage, ...]
    user_feedback_category: tuple[FeedbackLabelName, ...]
    user_feedback_text: tuple[str, ...]


class EmnlpExample(Frozen):
    turn: FeedbackTurn
    gold: GoldFeedback


class EmnlpParse(Frozen):
    conversations: tuple[Conversation, ...]
    examples: tuple[EmnlpExample, ...]
    skipped_conversations: tuple[str, ...]
    unaligned_feedback: int
    duplicate_feedback: int
    text_mismatches: int


class _RowParse(Frozen):
    conversation: Conversation
    examples: tuple[EmnlpExample, ...]
    unaligned_feedback: int
    duplicate_feedback: int
    text_mismatches: int


class _Exchange(Frozen):
    user_turn_number: int
    request: str
    response: str
    next_user_turn: str


ROWS_ADAPTER: Final = TypeAdapter(tuple[EmnlpRow, ...])


def parse_emnlp(raw_rows: Iterable[object], split: EmnlpSplit) -> EmnlpParse:
    rows: Final = ROWS_ADAPTER.validate_python(tuple(raw_rows))
    parsed: Final = tuple(_parse_row(row, split) for row in rows)
    kept: Final = tuple(result for result in parsed if result is not None)
    return EmnlpParse(
        conversations=tuple(result.conversation for result in kept),
        examples=tuple(chain.from_iterable(result.examples for result in kept)),
        skipped_conversations=tuple(row.conversation_id for row, result in zip(rows, parsed) if result is None),
        unaligned_feedback=sum(result.unaligned_feedback for result in kept),
        duplicate_feedback=sum(result.duplicate_feedback for result in kept),
        text_mismatches=sum(result.text_mismatches for result in kept),
    )


def _exchanges(conversation: Sequence[ChatMessage]) -> tuple[_Exchange, ...]:
    user_positions: Final = [i for i, message in enumerate(conversation) if message.role == "user"]
    return tuple(
        _Exchange(
            user_turn_number=number,
            request=conversation[prev].content,
            response="\n".join(m.content for m in conversation[prev + 1 : cur] if m.role == "assistant"),
            next_user_turn=conversation[cur].content,
        )
        for number, (prev, cur) in enumerate(pairwise(user_positions), start=2)
    )


def _parse_row(row: EmnlpRow, split: EmnlpSplit) -> _RowParse | None:
    exchanges: Final = _exchanges(row.conversation)
    pairs: Final = tuple(zip(row.user_feedback_category, row.user_feedback_text))
    matched: Final = _align(exchanges, pairs, split)
    if matched is None:
        return None
    aligned: Final = _first_per_turn(matched)
    examples: Final = tuple(_example(row, exchange, label) for exchange, (label, _) in aligned)
    mismatches: Final = sum(
        label is not FeedbackLabelName.NO_FEEDBACK and _norm(text) not in _norm(exchange.next_user_turn)
        for exchange, (label, text) in aligned
    )
    return _RowParse(
        conversation=Conversation(session_id=_session_id(row), messages=row.conversation),
        examples=examples,
        unaligned_feedback=len(pairs) - len(matched),
        duplicate_feedback=len(matched) - len(aligned),
        text_mismatches=mismatches,
    )


def _align(
    exchanges: Sequence[_Exchange], pairs: Sequence[tuple[FeedbackLabelName, str]], split: EmnlpSplit
) -> tuple[tuple[_Exchange, tuple[FeedbackLabelName, str]], ...] | None:
    match split:
        case "dense":
            return tuple(zip(exchanges, pairs)) if len(pairs) == len(exchanges) else None
        case "sparse":
            return tuple(
                (exchange, pair) for pair in pairs if (exchange := _find_exchange(exchanges, pair[1])) is not None
            )


def _first_per_turn(
    aligned: Sequence[tuple[_Exchange, tuple[FeedbackLabelName, str]]],
) -> tuple[tuple[_Exchange, tuple[FeedbackLabelName, str]], ...]:
    first: Final = {exchange.user_turn_number: (exchange, pair) for exchange, pair in reversed(aligned)}
    return tuple(sorted(first.values(), key=lambda item: item[0].user_turn_number))


def _find_exchange(exchanges: Sequence[_Exchange], text: str) -> _Exchange | None:
    return next((e for e in exchanges if _norm(text) and _norm(text) in _norm(e.next_user_turn)), None)


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


def _session_id(row: EmnlpRow) -> str:
    return f"{EMNLP_SOURCE}:{row.conversation_id}"


def _example(row: EmnlpRow, exchange: _Exchange, label: FeedbackLabelName) -> EmnlpExample:
    session_id: Final = _session_id(row)
    return EmnlpExample(
        turn=FeedbackTurn(
            session_id=session_id,
            source=EMNLP_SOURCE,
            model_id=EMNLP_MODEL_ID,
            group_key=row.dataset_source,
            turn_idx=exchange.user_turn_number,
            request=exchange.request,
            response=exchange.response,
            next_user_turn=exchange.next_user_turn,
        ),
        gold=GoldFeedback(session_id=session_id, turn_idx=exchange.user_turn_number, label=label),
    )
