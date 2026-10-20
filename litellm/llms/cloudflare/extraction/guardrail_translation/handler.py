from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Annotated, Final, Literal

from pydantic import Field, JsonValue, TypeAdapter

from litellm.litellm_core_utils.core_helpers import get_metadata_variable_name_from_kwargs
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.types.extraction import (
    EXTRACTION_STATE_REWRITTEN_METADATA_KEY,
    ExtractionAnswer,
    ExtractionBase64Image,
    ExtractionResponse,
)
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth

_FIELDS_ADAPTER: Final[TypeAdapter[Mapping[str, Mapping[str, object]]]] = TypeAdapter(
    Mapping[str, Mapping[str, object]]
)
_METADATA_ADAPTER: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])
_IMAGES_ADAPTER: Final[TypeAdapter[tuple[ExtractionBase64Image | JsonValue, ...]]] = TypeAdapter(
    tuple[Annotated[ExtractionBase64Image | JsonValue, Field(union_mode="left_to_right")], ...]
)


@dataclass(frozen=True, slots=True)
class _AnswerText:
    name: str
    key: Literal["value", "guess"]
    item: int | None
    text: str


_SPEC_TEXT_KEYS: Final = ("instructions", "unit")


@dataclass(frozen=True, slots=True)
class _SpecText:
    name: str
    key: str
    text: str


def _spec_texts_of(name: str, spec: Mapping[str, object]) -> tuple[_SpecText, ...]:
    return tuple(
        _SpecText(name=name, key=key, text=text) for key in _SPEC_TEXT_KEYS if isinstance(text := spec.get(key), str)
    )


def _spec_texts(fields: Mapping[str, Mapping[str, object]]) -> tuple[_SpecText, ...]:
    return tuple(chain.from_iterable(_spec_texts_of(name, spec) for name, spec in fields.items()))


def _guardrailed_spec(
    name: str, spec: Mapping[str, object], scanned: Sequence[tuple[_SpecText, str]]
) -> Mapping[str, object]:
    return {**spec, **{text.key: guardrailed for text, guardrailed in scanned if text.name == name}}


def _metadata_key(data: Mapping[str, object]) -> Literal["metadata", "litellm_metadata"]:
    return get_metadata_variable_name_from_kwargs(data)


def _metadata(data: Mapping[str, object]) -> Mapping[str, object]:
    return _METADATA_ADAPTER.validate_python(data.get(_metadata_key(data)) or {})


def _is_text_answer(answer: ExtractionAnswer, spec: Mapping[str, object] | None) -> bool:
    return answer.type == "text" or (answer.type == "list" and (spec is None or spec.get("items") == "text"))


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
    name: str,
    answer: ExtractionAnswer,
    scanned: Sequence[tuple[_AnswerText, str]],
    drop_spans: bool,
    is_text: bool,
) -> ExtractionAnswer:
    if not is_text and any(text.name == name and text.text != guardrailed for text, guardrailed in scanned):
        return answer.model_copy(update={"value": None, "guess": None, "span": None, "spans": None})
    value_scans: Final = _scans_of(name, "value", scanned if is_text else ())
    guess_scans: Final = _scans_of(name, "guess", scanned if is_text else ())
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
        spec_texts: Final = _spec_texts(fields)
        images: Final = _image_urls(data.get("images"))
        if not states and not spec_texts and not images:
            return data
        scanned: Final = await _scanned_texts(
            (*states, *(text.text for text in spec_texts)),
            data.get("model"),
            guardrail_to_apply,
            data,
            "request",
            litellm_logging_obj,
            images,
        )
        scanned_specs: Final = tuple(zip(spec_texts, scanned[len(states) :]))
        state_rewritten: Final = bool(states) and scanned[0] != states[0]
        return {
            **data,
            **({"state": scanned[0]} if states else {}),
            **(
                {_metadata_key(data): {**_metadata(data), EXTRACTION_STATE_REWRITTEN_METADATA_KEY: True}}
                if state_rewritten
                else {}
            ),
            "fields": {name: _guardrailed_spec(name, spec, scanned_specs) for name, spec in fields.items()},
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
        request: Final = request_data or {}
        scanned: Final = (
            await _scanned_texts(
                (*(text.text for text in texts), *state_texts),
                response.model,
                guardrail_to_apply,
                request,
                "response",
                litellm_logging_obj,
            )
            if texts or state_texts
            else ()
        )
        pairs: Final = tuple(zip(texts, scanned))
        state_text: Final = next(iter(scanned[len(texts) :]), response.state_text)
        specs: Final = _FIELDS_ADAPTER.validate_python(request.get("fields") or {})
        return response.model_copy(
            update={
                "answers": {
                    name: _guardrailed_answer(
                        name,
                        answer,
                        pairs,
                        drop_spans=state_text != response.state_text,
                        is_text=_is_text_answer(answer, specs.get(name)),
                    )
                    for name, answer in response.answers.items()
                },
                **({"state_text": state_text} if state_texts else {}),
            }
        )
