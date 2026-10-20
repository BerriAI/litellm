from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Annotated, Final, Literal

from pydantic import Field, JsonValue, TypeAdapter

from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.types.extraction import ExtractionAnswer, ExtractionBase64Image, ExtractionResponse
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth

_FIELDS_ADAPTER: Final[TypeAdapter[Mapping[str, Mapping[str, object]]]] = TypeAdapter(
    Mapping[str, Mapping[str, object]]
)
_IMAGES_ADAPTER: Final[TypeAdapter[tuple[ExtractionBase64Image | JsonValue, ...]]] = TypeAdapter(
    tuple[Annotated[ExtractionBase64Image | JsonValue, Field(union_mode="left_to_right")], ...]
)


@dataclass(frozen=True, slots=True)
class _AnswerText:
    name: str
    key: Literal["value", "guess"]
    item: int | None
    text: str


def _instructions(fields: Mapping[str, Mapping[str, object]]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (name, instructions)
        for name, spec in fields.items()
        if isinstance(instructions := spec.get("instructions"), str)
    )


def _image_urls(images: object) -> tuple[str, ...]:
    if not isinstance(images, list):
        return ()
    return tuple(
        f"data:{image.content_type};base64,{image.base64}" if isinstance(image, ExtractionBase64Image) else image
        for image in _IMAGES_ADAPTER.validate_python(images)
        if isinstance(image, (str, ExtractionBase64Image))
    )


def _value_texts(name: str, key: Literal["value", "guess"], value: JsonValue) -> tuple[_AnswerText, ...]:
    if isinstance(value, str):
        return (_AnswerText(name=name, key=key, item=None, text=value),)
    if isinstance(value, list):
        return tuple(
            _AnswerText(name=name, key=key, item=index, text=item)
            for index, item in enumerate(value)
            if isinstance(item, str)
        )
    return ()


def _answer_texts(name: str, answer: ExtractionAnswer) -> tuple[_AnswerText, ...]:
    return (*_value_texts(name, "value", answer.value), *_value_texts(name, "guess", answer.guess))


def _guardrailed_value(value: JsonValue, scanned: Mapping[int | None, str]) -> JsonValue:
    if isinstance(value, list):
        return [scanned.get(index, item) for index, item in enumerate(value)]
    return scanned.get(None, value)


def _scans_of(
    name: str, key: Literal["value", "guess"], scanned: Sequence[tuple[_AnswerText, str]]
) -> Mapping[int | None, str]:
    return {text.item: guardrailed for text, guardrailed in scanned if (text.name, text.key) == (name, key)}


def _guardrailed_answer(
    name: str, answer: ExtractionAnswer, scanned: Sequence[tuple[_AnswerText, str]], drop_spans: bool
) -> ExtractionAnswer:
    value_scans: Final = _scans_of(name, "value", scanned)
    guess_scans: Final = _scans_of(name, "guess", scanned)
    update: Final = {
        **({"value": _guardrailed_value(answer.value, value_scans)} if value_scans else {}),
        **({"guess": _guardrailed_value(answer.guess, guess_scans)} if guess_scans else {}),
        **({"span": None, "spans": None} if drop_spans else {}),
    }
    return answer.model_copy(update=update) if update else answer


async def _scanned_texts(
    texts: Sequence[str],
    model: object,
    guardrail_to_apply: "CustomGuardrail",
    request_data: dict[str, object],  # mutable-ok: CustomGuardrail.apply_guardrail takes the request as a plain dict
    input_type: Literal["request", "response"],
    litellm_logging_obj: "LiteLLMLoggingObj | None",
    images: Sequence[str] = (),
) -> tuple[str, ...]:
    inputs: Final = GenericGuardrailAPIInputs(texts=list(texts), model=model if isinstance(model, str) else None)
    if images:
        inputs["images"] = list(images)
    guardrailed: Final = await guardrail_to_apply.apply_guardrail(  # pyright: ignore[reportUnknownMemberType]  # CustomGuardrail.apply_guardrail types request_data as a bare dict
        inputs=inputs,
        request_data=request_data,
        input_type=input_type,
        logging_obj=litellm_logging_obj,
    )
    returned: Final = guardrailed.get("texts", [])
    return tuple(returned[index] if index < len(returned) else text for index, text in enumerate(texts))


class ExtractionHandler(BaseTranslation):
    async def process_input_messages(
        self,
        data: dict[str, object],  # mutable-ok: BaseTranslation hands over the proxy's request dict
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> Mapping[str, object]:
        state: Final = data.get("state")
        states: Final = (state,) if isinstance(state, str) and state else ()
        fields: Final = _FIELDS_ADAPTER.validate_python(data.get("fields") or {})
        instructions: Final = _instructions(fields)
        images: Final = _image_urls(data.get("images"))
        if not states and not instructions and not images:
            return data
        scanned: Final = await _scanned_texts(
            (*states, *(text for _, text in instructions)),
            data.get("model"),
            guardrail_to_apply,
            data,
            "request",
            litellm_logging_obj,
            images,
        )
        scanned_instructions: Final = dict(zip((name for name, _ in instructions), scanned[len(states) :]))
        return {
            **data,
            **({"state": scanned[0]} if states else {}),
            "fields": {
                name: {**spec, "instructions": scanned_instructions[name]} if name in scanned_instructions else spec
                for name, spec in fields.items()
            },
        }

    async def process_output_response(
        self,
        response: object,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: "LiteLLMLoggingObj | None" = None,
        user_api_key_dict: "UserAPIKeyAuth | None" = None,
        request_data: dict[str, object] | None = None,  # mutable-ok: BaseTranslation hands over the request dict
    ) -> object:
        if not isinstance(response, ExtractionResponse):
            return response
        texts: Final = tuple(
            chain.from_iterable(_answer_texts(name, answer) for name, answer in response.answers.items())
        )
        state_texts: Final = (response.state_text,) if response.state_text else ()
        if not texts and not state_texts:
            return response
        scanned: Final = await _scanned_texts(
            (*(text.text for text in texts), *state_texts),
            response.model,
            guardrail_to_apply,
            request_data or {},
            "response",
            litellm_logging_obj,
        )
        pairs: Final = tuple(zip(texts, scanned))
        state_text: Final = scanned[-1] if state_texts else response.state_text
        return response.model_copy(
            update={
                "answers": {
                    name: _guardrailed_answer(name, answer, pairs, drop_spans=state_text != response.state_text)
                    for name, answer in response.answers.items()
                },
                **({"state_text": state_text} if state_texts else {}),
            }
        )
