"""Chat-synthetic Decisions provider for any OpenAI-compatible chat upstream.

Synthesizes a Decisions call into a single constrained Chat Completions
request so that *any* OpenAI-compatible chat upstream (vLLM, SGLang,
DeepSeek, GLM, third-party relays, openai_like deployments) can serve
Decisions workloads — no native Decisions endpoint required.

Strategy (single-letter token mapping):

- predicate (noul):  yes/true → "A", no/false → "B"
- choice:            option i → chr(ord("A") + i)
- score:             level j  → chr(ord("A") + j)

All questions are answered in ONE non-streaming chat call: the system
prompt enumerates every question with its letter table, the user message
carries the state, and the model answers with one letter per question.
When the upstream returns logprobs for the first sampled token we
softmax-normalize the matched top_logprobs entries into per-option
probabilities (unmatched options get 0.0); confidence is the chosen
option's normalized probability. Upstreams without logprobs support still
get usable answers — the chosen option gets point probability 1.0.

This is the chat-synthetic counterpart of the native Decisions providers
(OpenAI, OpenRouter, Perplexity, TypeSafe, Cloudflare, Strands): those
require the upstream to actually expose a Decisions/System One endpoint;
this one only needs /chat/completions.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Final

from litellm.llms.base_llm.decisions.transformation import (
    BaseDecisionsConfig,
    decisions_text,
)
from litellm.types.decisions import (
    DecisionsIRAnswer,
    DecisionsIRChoiceAnswer,
    DecisionsIRChoiceProbability,
    DecisionsIRChoiceQuestion,
    DecisionsIRMessages,
    DecisionsIRPredicateAnswer,
    DecisionsIRPredicateQuestion,
    DecisionsIRQuestion,
    DecisionsIRRefusal,
    DecisionsIRRequest,
    DecisionsIRResponse,
    DecisionsIRScoreAnswer,
    DecisionsIRScoreProbability,
    DecisionsIRScoreQuestion,
    DecisionsIRState,
    DecisionsIRUsage,
    UnsupportedDecisionsRequest,
    _NO_EXTRA,
)

_LETTER_A: Final = ord("A")
MAX_SYNTHETIC_OPTIONS: Final = 26  # one letter per option; A..Z

_TEXT_ONLY: Final = UnsupportedDecisionsRequest(
    reason="chat-synthetic Decisions accepts text input only; image content parts are not supported"
)
_TOO_MANY_OPTIONS: Final = UnsupportedDecisionsRequest(
    reason=f"chat-synthetic Decisions supports at most {MAX_SYNTHETIC_OPTIONS} options per question (A..Z)"
)


def _letter(index: int) -> str:
    return chr(_LETTER_A + index)


def _option_count(question: DecisionsIRQuestion) -> int:
    match question:
        case DecisionsIRPredicateQuestion():
            return 2
        case DecisionsIRChoiceQuestion():
            return len(question.choices)
        case DecisionsIRScoreQuestion():
            return len(question.levels)
        case _:
            return 0


def _instructions_or_default(instructions: object, default: str) -> str:
    if instructions is None:
        return default
    return decisions_text(instructions)


def _state_text(decision_input: DecisionsIRState | DecisionsIRMessages) -> str | UnsupportedDecisionsRequest:
    match decision_input:
        case DecisionsIRState():
            return decisions_text(decision_input.state)
        case DecisionsIRMessages():
            for message in decision_input.messages:
                if not isinstance(message.content, str) and any(
                    (isinstance(part, Mapping) and part.get("type") == "image_url")
                    or (not isinstance(part, str | Mapping) and getattr(part, "type", "") == "input_image")
                    for part in message.content
                ):
                    return _TEXT_ONLY
            return "\n\n".join(
                message.content if isinstance(message.content, str) else json.dumps(message.content)
                for message in decision_input.messages
            )
        case _:
            return _TEXT_ONLY


def _question_block(index: int, question: DecisionsIRQuestion) -> str | UnsupportedDecisionsRequest:
    match question:
        case DecisionsIRPredicateQuestion():
            return (
                f"Q{index + 1}. Answer with a single letter: A (yes/true) or B (no/false).\n"
                f"Question: {_instructions_or_default(question.instructions, 'Is this true of the input?')}"
            )
        case DecisionsIRChoiceQuestion():
            if len(question.choices) > MAX_SYNTHETIC_OPTIONS:
                return _TOO_MANY_OPTIONS
            options: Final = "\n".join(
                f"   {_letter(i)}. {decisions_text(option.value)}"
                + (f" — {decisions_text(option.description)}" if option.description is not None else "")
                for i, option in enumerate(question.choices)
            )
            return (
                f"Q{index + 1}. Answer with the single letter of the best choice.\n"
                f"Question: {_instructions_or_default(question.instructions, 'Which choice best fits the input?')}\n"
                f"{options}"
            )
        case DecisionsIRScoreQuestion():
            if len(question.levels) > MAX_SYNTHETIC_OPTIONS:
                return _TOO_MANY_OPTIONS
            levels: Final = "\n".join(
                f"   {_letter(j)}. {decisions_text(level.label)}"
                + (f" — {level.description}" if level.description is not None else "")
                for j, level in enumerate(question.levels)
            )
            return (
                f"Q{index + 1}. Answer with the single letter of the matching level.\n"
                f"Question: {_instructions_or_default(question.instructions, 'Which level best fits the input?')}\n"
                f"{levels}"
            )
        case _:
            return _TOO_MANY_OPTIONS


def build_chat_messages(request: DecisionsIRRequest) -> tuple[str, str] | UnsupportedDecisionsRequest:
    """Render the IR request into (system prompt, user state) chat content."""
    state: Final = _state_text(request.input)
    if isinstance(state, UnsupportedDecisionsRequest):
        return state
    blocks: list[str] = [
        "You are a decision engine. For each question, choose exactly one option.",
        "Respond with ONLY the letter of your chosen option for each question, in order,",
        "separated by spaces on a single line. Do not output any other text.",
        "",
    ]
    for index, question in enumerate(request.questions):
        block: Final = _question_block(index, question)
        if isinstance(block, UnsupportedDecisionsRequest):
            return block
        blocks.append(block)
        blocks.append("")
    return "\n".join(blocks), state


_ANSWER_RE: Final = re.compile(r"[A-Z]")


def _parse_letters(content: str, count: int) -> list[int] | None:
    """Extract per-question 0-based letter indices from the model output."""
    if not content:
        return None
    letters: Final = _ANSWER_RE.findall(content.strip())
    if not letters:
        return None
    indices: list[int] = []
    for letter in letters:
        index: Final = ord(letter) - _LETTER_A
        if index >= MAX_SYNTHETIC_OPTIONS:
            continue
        indices.append(index)
        if len(indices) == count:
            return indices
    return None


def _first_token_probabilities(logprobs: object, option_count: int) -> dict[int, float] | None:
    """Softmax-normalize the first token's top_logprobs into option-index space."""
    if option_count <= 0 or not isinstance(logprobs, Mapping):
        return None
    content: Final = logprobs.get("content")
    if not isinstance(content, Sequence) or not content:
        return None
    first: Final = content[0]
    if not isinstance(first, Mapping):
        return None
    top: Final = first.get("top_logprobs")
    if not isinstance(top, Sequence) or not top:
        return None
    linear: dict[int, float] = {}
    total: float = 0.0
    for entry in top:
        if not isinstance(entry, Mapping):
            continue
        token: Final = entry.get("token")
        if not isinstance(token, str) or len(token) != 1:
            continue
        index: Final = ord(token) - _LETTER_A
        if index < 0 or index >= option_count or index in linear:
            continue
        logprob: Final = entry.get("logprob")
        if not isinstance(logprob, int | float):
            continue
        probability: Final = math.exp(float(logprob))
        linear[index] = probability
        total += probability
    if total <= 0.0:
        return None
    return {index: probability / total for index, probability in linear.items()}


def _predicate_answer(index: int, probabilities: Mapping[int, float] | None) -> DecisionsIRAnswer:
    # Letter A means yes/true: its normalized probability is the predicate probability.
    if probabilities is not None:
        return DecisionsIRPredicateAnswer(probability=probabilities.get(0, 1.0 if index == 0 else 0.0))
    return DecisionsIRPredicateAnswer(probability=1.0 if index == 0 else 0.0)


def _choice_answer(
    question: DecisionsIRChoiceQuestion,
    index: int,
    probabilities: Mapping[int, float] | None,
) -> DecisionsIRAnswer:
    options: Final = question.choices
    if index >= len(options):
        return DecisionsIRRefusal()
    return DecisionsIRChoiceAnswer(
        choice=options[index].value,
        confidence=probabilities.get(index, 1.0) if probabilities is not None else 1.0,
        probabilities=tuple(
            DecisionsIRChoiceProbability(
                value=option.value,
                probability=probabilities.get(i, 0.0) if probabilities is not None else (1.0 if i == index else 0.0),
            )
            for i, option in enumerate(options)
        ),
        extra=_NO_EXTRA,
    )


def _score_answer(
    question: DecisionsIRScoreQuestion,
    index: int,
    probabilities: Mapping[int, float] | None,
) -> DecisionsIRAnswer:
    levels: Final = question.levels
    if index >= len(levels):
        return DecisionsIRRefusal()
    return DecisionsIRScoreAnswer(
        score=float(index),
        confidence=probabilities.get(index, 1.0) if probabilities is not None else 1.0,
        probabilities=tuple(
            DecisionsIRScoreProbability(
                value=j,
                label=level.label,
                probability=probabilities.get(j, 0.0) if probabilities is not None else (1.0 if j == index else 0.0),
            )
            for j, level in enumerate(levels)
        ),
        extra=_NO_EXTRA,
    )


def chat_response_to_ir(payload: object, request: DecisionsIRRequest) -> DecisionsIRResponse:
    """Convert a Chat Completions response payload into the Decisions IR.

    Parses per-question letters from the message content. When the upstream
    returned first-token logprobs, question 0 gets a normalized probability
    vector derived from the matched top_logprobs; later questions fall back
    to point probabilities. Unparseable output yields refusals rather than
    exceptions so callers can surface partial results.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("chat-synthetic Decisions expected a JSON object response")
    choices: Final = payload.get("choices")
    if not isinstance(choices, Sequence) or not choices:
        raise ValueError("chat-synthetic Decisions response has no choices")
    first: Final = choices[0]
    if not isinstance(first, Mapping):
        raise ValueError("chat-synthetic Decisions choice is not an object")
    message: Final = first.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("chat-synthetic Decisions choice has no message")
    content: Final = message.get("content")
    if not isinstance(content, str):
        content = json.dumps(content) if content is not None else ""

    questions: Final = request.questions
    indices: Final = _parse_letters(content, len(questions))
    # First-token logprobs only reflect the first question's distribution.
    first_probs: Final = (
        _first_token_probabilities(first.get("logprobs"), _option_count(questions[0])) if questions else None
    )

    answers: list[DecisionsIRAnswer] = []
    for position, question in enumerate(questions):
        probabilities: Mapping[int, float] | None = first_probs if position == 0 else None
        index: Final = indices[position] if indices is not None and position < len(indices) else -1
        if index < 0:
            answers.append(DecisionsIRRefusal())
            continue
        match question:
            case DecisionsIRPredicateQuestion():
                answers.append(_predicate_answer(index, probabilities))
            case DecisionsIRChoiceQuestion():
                answers.append(_choice_answer(question, index, probabilities))
            case DecisionsIRScoreQuestion():
                answers.append(_score_answer(question, index, probabilities))
            case _:
                answers.append(DecisionsIRRefusal())

    usage: Final = payload.get("usage")
    input_tokens: Final = usage.get("prompt_tokens", 0) if isinstance(usage, Mapping) else 0
    output_tokens: Final = usage.get("completion_tokens", 0) if isinstance(usage, Mapping) else 0
    model: Final = payload.get("model")

    return DecisionsIRResponse(
        model=model if isinstance(model, str) else None,
        answers=tuple(answers),
        usage=DecisionsIRUsage(input_tokens=int(input_tokens), output_tokens=int(output_tokens)),
        extra=_NO_EXTRA,
    )


class OpenAILikeDecisionsConfig(BaseDecisionsConfig):
    """Chat-synthetic Decisions for openai_like (OpenAI-compatible) deployments.

    Points the Decisions pipeline at the deployment's chat completions
    route and synthesizes a constrained single-letter chat call. Works for
    vLLM, SGLang, DeepSeek, GLM, llama.cpp server, and any other
    OpenAI-compatible chat upstream.
    """

    path = "/chat/completions"

    def get_default_api_base(self) -> str | None:
        return None  # deployment-specific; openai_like deployments require api_base

    def transform_decisions_request(
        self,
        model: str,
        request: DecisionsIRRequest,
        custom_llm_provider: str,
    ) -> Mapping[str, object] | UnsupportedDecisionsRequest:
        messages: Final = build_chat_messages(request)
        if isinstance(messages, UnsupportedDecisionsRequest):
            return messages
        system_prompt, state = messages
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": state},
            ],
            "temperature": 0,
            "max_tokens": 1 if len(request.questions) == 1 else 8 * len(request.questions),
            "logprobs": True,
            "top_logprobs": 20,
            "stream": False,
        }

    def parse_response(self, payload: object, request: DecisionsIRRequest) -> DecisionsIRResponse:
        return chat_response_to_ir(payload, request)
