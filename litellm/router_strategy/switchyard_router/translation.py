from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Final, Protocol

from pydantic import TypeAdapter

from litellm.types.decisions import (
    OpenAIChoiceAnswer,
    OpenAIDecisionResponse,
    systemone_choice_key,
)
from litellm.types.utils import CallTypes, ModelResponse

if TYPE_CHECKING:
    from switchyard_rust.libsy import LlmResponse

_MAPPING: Final = TypeAdapter(Mapping[str, object])
_SEQUENCE: Final = TypeAdapter(tuple[object, ...])
_CALL_TYPES: Final = frozenset(call_type.value for call_type in CallTypes)


def mapping(value: object) -> Mapping[str, object]:
    return _MAPPING.validate_python(value)


def sequence(value: object) -> tuple[object, ...]:
    return _SEQUENCE.validate_python(value)


def text_content(value: object) -> tuple[Mapping[str, object], ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return ({"type": "text", "text": value},)
    blocks: Final = tuple(mapping(block) for block in sequence(value))
    if any(block.get("type") != "text" or not isinstance(block.get("text"), str) for block in blocks):
        raise ValueError("Switchyard routing supports text content only; media and audio are unsupported")
    return tuple({"type": "text", "text": block["text"]} for block in blocks)


def tool_call(value: object) -> Mapping[str, object]:
    call: Final = mapping(value)
    function: Final = mapping(call.get("function"))
    arguments: Final = function.get("arguments")
    if call.get("type") != "function" or not isinstance(arguments, str):
        raise ValueError("Switchyard requires function calls with JSON arguments")
    return {
        "type": "tool_call",
        "id": call["id"],
        "name": function["name"],
        "arguments": _MAPPING.validate_json(arguments),
    }


def message_to_switchyard(value: object) -> Mapping[str, object]:
    message: Final = mapping(value)
    if message.get("audio") is not None:
        raise ValueError("Switchyard routing does not support audio")
    role: Final = message.get("role")
    content: Final = text_content(message.get("content"))
    if role == "tool":
        return {
            "role": "tool",
            "content": (
                {
                    "type": "tool_result",
                    "tool_call_id": message["tool_call_id"],
                    "content": content,
                    "is_error": message.get("is_error"),
                },
            ),
        }
    if role not in ("user", "system", "developer", "assistant"):
        raise ValueError("Switchyard received an unsupported message role")
    calls: Final = tuple(tool_call(call) for call in sequence(message.get("tool_calls") or ()))
    return {"role": role, "content": (*content, *calls)}


class ChatConverter(Protocol):
    def get_structured_messages(
        self,
        data: dict[str, object],  # mutable-ok: matches the request converters' existing dict parameter
        *,
        keep_tool_errors: bool = False,
    ) -> Sequence[object] | None: ...


def _instantiate(converter: Callable[[], ChatConverter]) -> ChatConverter:
    return converter()


def chat_messages(
    call_type: str | None,
    model: str,
    messages: Sequence[Mapping[str, object]] | None,
    input: object,
    request_kwargs: Mapping[str, object],
) -> tuple[Mapping[str, object], ...] | None:
    from litellm.llms import load_guardrail_translation_mappings

    converter: Final = (
        load_guardrail_translation_mappings().get(CallTypes(call_type)) if call_type in _CALL_TYPES else None
    )
    if converter is None:
        return None
    request: Final = {
        **request_kwargs,
        "model": model,
        **({"messages": [dict(message) for message in messages]} if messages is not None else {}),
        **({"input": input} if input is not None else {}),
    }
    return tuple(map(mapping, _instantiate(converter).get_structured_messages(request, keep_tool_errors=True) or ()))


def request_to_switchyard(messages: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    if not messages:
        raise ValueError("Switchyard requires at least one message")
    return {
        "model": "auto",
        "messages": tuple(message_to_switchyard(message) for message in messages),
        "stream": False,
    }


def judge_message(value: object) -> Mapping[str, object]:
    message: Final = mapping(value)
    blocks: Final = text_content(message.get("content"))
    return {"role": message["role"], "content": "".join(str(block["text"]) for block in blocks)}


def judge_request(request: Mapping[str, object], model: str) -> Mapping[str, object]:
    output: Final = mapping(request.get("output") or {})
    sampling: Final = mapping(request.get("sampling") or {})
    options: Final = {
        "max_tokens": output.get("max_output_tokens"),
        "response_format": output.get("response_format"),
        "temperature": sampling.get("temperature"),
        "top_p": sampling.get("top_p"),
    }
    return {
        "model": model,
        "stream": False,
        "messages": [
            judge_message(message)
            for message in (
                *sequence(request.get("instructions") or ()),
                *sequence(request.get("messages") or ()),
            )
        ],
        **{key: value for key, value in options.items() if value is not None},
    }


def judge_response(response: ModelResponse) -> LlmResponse.Agg:
    from switchyard_rust.libsy import LlmResponse

    body: Final = _MAPPING.validate_json(response.model_dump_json(exclude_unset=True))
    choices: Final = tuple(mapping(choice) for choice in sequence(body["choices"]))
    return LlmResponse.Agg(
        {
            "id": response.id,
            "model": response.model,
            "outputs": tuple(message_to_switchyard(choice["message"]) for choice in choices),
        }
    )


def _decision_text(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def _decision_question(name: str, question: Mapping[str, object]) -> Mapping[str, object]:
    kind: Final = mapping(question["kind"])
    if kind.get("type") != "choice":
        raise ValueError("Switchyard decision routing asks choice questions only")
    return {
        "type": "choice",
        "name": name,
        "instructions": _decision_text(question.get("instructions")),
        "choices": [
            {
                "value": option["id"],
                **({"description": _decision_text(option["description"])} if option.get("description") else {}),
            }
            for option in map(mapping, sequence(mapping(kind["data"])["options"]))
        ],
    }


def decision_request(request: Mapping[str, object], model: str) -> Mapping[str, object]:
    questions: Final = mapping(request["questions"])
    return {
        "model": model,
        "input": _decision_text(request["context"]),
        "questions": [_decision_question(name, mapping(question)) for name, question in questions.items()],
    }


def _choice_value(answer: object) -> Mapping[str, object]:
    if not isinstance(answer, OpenAIChoiceAnswer):
        raise TypeError("The decision model refused or returned a non-choice answer")
    return {
        "type": "choice",
        "data": {
            "selected": systemone_choice_key(answer.choice),
            "probabilities": {systemone_choice_key(p.value): p.probability for p in answer.probabilities},
        },
    }


def decision_response(response: OpenAIDecisionResponse, request: Mapping[str, object]) -> Mapping[str, object]:
    names: Final = tuple(mapping(request["questions"]))
    if len(response.answers) != len(names):
        raise ValueError("The decision model did not answer every question")
    return {
        "id": None,
        "model": response.model,
        "answers": {
            answer.name or name: {
                "value": _choice_value(answer),
                "provider_confidence": getattr(answer, "confidence", None),
            }
            for name, answer in zip(names, response.answers, strict=True)
        },
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }
