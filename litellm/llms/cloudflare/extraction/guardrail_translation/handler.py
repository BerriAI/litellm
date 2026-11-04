from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain, groupby
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Final, Literal, TypeAlias, TypeVar

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


_AnswerKey: TypeAlias = Literal["value", "guess", "mean", "median", "p10", "p90", "quantiles"]
_ESTIMATE_KEYS: Final[tuple[_AnswerKey, ...]] = ("value", "guess", "mean", "median", "p10", "p90", "quantiles")
_STATISTIC_KEYS: Final[frozenset[_AnswerKey]] = frozenset(("mean", "median", "p10", "p90", "quantiles"))
_NULLED_ANSWER: Final[Mapping[str, None]] = MappingProxyType(
    dict.fromkeys(("value", "guess", "mean", "median", "p10", "p90", "quantiles", "steps", "span", "spans"), None)
)


@dataclass(frozen=True, slots=True)
class _AnswerText:
    name: str
    key: _AnswerKey
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


_NamedText = TypeVar("_NamedText", _AnswerText, _SpecText)


def _scans_by_name(scanned: Iterable[tuple[_NamedText, str]]) -> Mapping[str, tuple[tuple[_NamedText, str], ...]]:
    return {name: tuple(group) for name, group in groupby(scanned, key=lambda pair: pair[0].name)}


def _guardrailed_spec(spec: Mapping[str, object], scanned: Sequence[tuple[_SpecText, str]]) -> Mapping[str, object]:
    return {**spec, **{text.key: guardrailed for text, guardrailed in scanned}}


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


def _scalar_text(value: JsonValue, whole: bool) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and whole:
        return format(value, ".0f")
    return str(value) if isinstance(value, (str, int, float)) else None


def _item_texts(items: Sequence[JsonValue], whole: bool) -> tuple[tuple[int, str], ...]:
    return tuple((index, text) for index, item in enumerate(items) if (text := _scalar_text(item, whole)) is not None)


def _value_texts(name: str, key: _AnswerKey, value: JsonValue) -> tuple[_AnswerText, ...]:
    whole: Final = key in _STATISTIC_KEYS
    if isinstance(value, list):
        return tuple(
            _AnswerText(name=name, key=key, item=index, text=text) for index, text in _item_texts(value, whole)
        )
    text: Final = _scalar_text(value, whole)
    return () if text is None else (_AnswerText(name=name, key=key, item=None, text=text),)


def _step_texts(name: str, step: JsonValue) -> tuple[_AnswerText, ...]:
    if not isinstance(step, dict):
        return ()
    return tuple(chain.from_iterable(_value_texts(name, key, step.get(key)) for key in _ESTIMATE_KEYS))


def _answer_texts(name: str, answer: ExtractionAnswer) -> tuple[_AnswerText, ...]:
    return (
        *_value_texts(name, "value", answer.value),
        *_value_texts(name, "guess", answer.guess),
        *_value_texts(name, "mean", answer.mean),
        *_value_texts(name, "median", answer.median),
        *_value_texts(name, "p10", answer.p10),
        *_value_texts(name, "p90", answer.p90),
        *_value_texts(name, "quantiles", list(answer.quantiles or ())),
        *chain.from_iterable(_step_texts(name, step) for step in answer.steps or ()),
    )


def _guardrailed_value(value: JsonValue, scanned: Mapping[int | None, str]) -> JsonValue:
    if isinstance(value, list):
        return [scanned.get(index, item) for index, item in enumerate(value)]
    return scanned.get(None, value)


def _scans_of(key: _AnswerKey, scanned: Sequence[tuple[_AnswerText, str]]) -> Mapping[int | None, str]:
    return {text.item: guardrailed for text, guardrailed in scanned if text.key == key}


def _guardrailed_answer(
    answer: ExtractionAnswer,
    scanned: Sequence[tuple[_AnswerText, str]],
    drop_spans: bool,
    is_text: bool,
) -> ExtractionAnswer:
    if not is_text and any(text.text != guardrailed for text, guardrailed in scanned):
        return answer.model_copy(update=_NULLED_ANSWER)
    value_scans: Final = _scans_of("value", scanned if is_text else ())
    guess_scans: Final = _scans_of("guess", scanned if is_text else ())
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
    unique_texts: Final = tuple(dict.fromkeys(texts))
    inputs: Final = GenericGuardrailAPIInputs(texts=list(unique_texts), model=model if isinstance(model, str) else None)
    if images:
        inputs["images"] = list(images)
    guardrailed: Final = await guardrail_to_apply.apply_guardrail(  # pyright: ignore[reportUnknownMemberType]  # CustomGuardrail.apply_guardrail types request_data as a bare dict
        inputs=inputs,
        request_data=request_data,
        input_type=input_type,
        logging_obj=litellm_logging_obj,
    )
    returned: Final = guardrailed.get("texts", [])
    guardrailed_by_text: Final = MappingProxyType(
        {text: returned[index] if index < len(returned) else text for index, text in enumerate(unique_texts)}
    )
    return tuple(guardrailed_by_text[text] for text in texts)


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
        spec_scans: Final = _scans_by_name(zip(spec_texts, scanned[len(states) :]))
        state_rewritten: Final = bool(states) and scanned[0] != states[0]
        return {
            **data,
            **({"state": scanned[0]} if states else {}),
            **(
                {_metadata_key(data): {**_metadata(data), EXTRACTION_STATE_REWRITTEN_METADATA_KEY: True}}
                if state_rewritten
                else {}
            ),
            "fields": {name: _guardrailed_spec(spec, spec_scans.get(name, ())) for name, spec in fields.items()},
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
        answer_scans: Final = _scans_by_name(zip(texts, scanned))
        state_text: Final = next(iter(scanned[len(texts) :]), response.state_text)
        specs: Final = _FIELDS_ADAPTER.validate_python(request.get("fields") or {})
        return response.model_copy(
            update={
                "answers": {
                    name: _guardrailed_answer(
                        answer,
                        answer_scans.get(name, ()),
                        drop_spans=state_text != response.state_text,
                        is_text=_is_text_answer(answer, specs.get(name)),
                    )
                    for name, answer in response.answers.items()
                },
                **({"state_text": state_text} if state_texts else {}),
            }
        )
