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
_MAPPING_ADAPTER: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])
_SEQUENCE_ADAPTER: Final[TypeAdapter[tuple[object, ...]]] = TypeAdapter(tuple[object, ...])
_METADATA_ADAPTER: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])
_IMAGES_ADAPTER: Final[TypeAdapter[tuple[ExtractionBase64Image | JsonValue, ...]]] = TypeAdapter(
    tuple[Annotated[ExtractionBase64Image | JsonValue, Field(union_mode="left_to_right")], ...]
)


_AnswerKey: TypeAlias = Literal["value", "guess", "mean", "median", "p10", "p90", "quantiles"]
_SpecPath: TypeAlias = tuple[str | int, ...]
_ESTIMATE_KEYS: Final[tuple[_AnswerKey, ...]] = ("value", "guess", "mean", "median", "p10", "p90", "quantiles")
_STATISTIC_KEYS: Final[frozenset[_AnswerKey]] = frozenset(("mean", "median", "p10", "p90", "quantiles"))
_NULLED_ANSWER: Final[Mapping[str, None]] = MappingProxyType(
    dict.fromkeys(
        (
            "value",
            "guess",
            "mean",
            "median",
            "p10",
            "p90",
            "quantiles",
            "steps",
            "span",
            "spans",
            "property_spans",
            "raw",
            "printed_value",
        ),
        None,
    )
)


@dataclass(frozen=True, slots=True)
class _AnswerText:
    name: str
    key: _AnswerKey
    item: int | None
    prop: str | None
    text: str
    rewritable: bool


_SPEC_TEXT_KEYS: Final = ("instructions", "unit")


@dataclass(frozen=True, slots=True)
class _SpecText:
    name: str
    path: _SpecPath
    text: str


def _as_mapping(value: object) -> Mapping[str, object] | None:
    return _MAPPING_ADAPTER.validate_python(value) if isinstance(value, Mapping) else None


def _as_sequence(value: object) -> tuple[object, ...] | None:
    return (
        _SEQUENCE_ADAPTER.validate_python(value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
        else None
    )


def _as_property_specs(value: object) -> Mapping[str, Mapping[str, object]] | None:
    raw: Final = _as_mapping(value)
    if raw is None:
        return None
    return {
        key: _MAPPING_ADAPTER.validate_python(prop_spec)
        for key, prop_spec in raw.items()
        if isinstance(prop_spec, Mapping)
    }


def _leaf_spec_texts(name: str, prefix: _SpecPath, spec: Mapping[str, object]) -> tuple[_SpecText, ...]:
    scalar_texts: Final = tuple(
        _SpecText(name=name, path=(*prefix, key), text=text)
        for key in _SPEC_TEXT_KEYS
        if isinstance(text := spec.get(key), str)
    )
    enum_values: Final = _as_sequence(spec.get("enum"))
    if enum_values is None:
        return scalar_texts
    enum_texts: Final = tuple(
        _SpecText(name=name, path=(*prefix, "enum", index), text=item)
        for index, item in enumerate(enum_values)
        if isinstance(item, str)
    )
    return (*scalar_texts, *enum_texts)


def _properties_spec_texts(name: str, prefix: _SpecPath, raw_props: object) -> tuple[_SpecText, ...]:
    props: Final = _as_property_specs(raw_props)
    if props is None:
        return ()
    return tuple(
        chain.from_iterable(_leaf_spec_texts(name, (*prefix, prop), prop_spec) for prop, prop_spec in props.items())
    )


def _spec_texts_of(name: str, spec: Mapping[str, object]) -> tuple[_SpecText, ...]:
    items: Final = _as_mapping(spec.get("items"))
    return (
        *_leaf_spec_texts(name, (), spec),
        *_properties_spec_texts(name, ("properties",), spec.get("properties")),
        *(_properties_spec_texts(name, ("items", "properties"), items.get("properties")) if items is not None else ()),
    )


def _spec_texts(fields: Mapping[str, Mapping[str, object]]) -> tuple[_SpecText, ...]:
    return tuple(chain.from_iterable(_spec_texts_of(name, spec) for name, spec in fields.items()))


_NamedText = TypeVar("_NamedText", _AnswerText, _SpecText)


def _scans_by_name(scanned: Iterable[tuple[_NamedText, str]]) -> Mapping[str, tuple[tuple[_NamedText, str], ...]]:
    return {name: tuple(group) for name, group in groupby(scanned, key=lambda pair: pair[0].name)}


def _guardrailed_leaf_spec(
    spec: Mapping[str, object],
    updates: Mapping[_SpecPath, str],
    prefix: _SpecPath,
) -> Mapping[str, object]:
    top: Final[Mapping[str, str]] = {
        key: value for key in _SPEC_TEXT_KEYS if (value := updates.get((*prefix, key))) is not None
    }
    enum_values: Final = _as_sequence(spec.get("enum"))
    enum_updates: Final[Mapping[int, str]] = (
        {
            index: item_text
            for index in range(len(enum_values))
            if (item_text := updates.get((*prefix, "enum", index))) is not None
        }
        if enum_values is not None
        else {}
    )
    if not top and not enum_updates:
        return spec
    return {
        **spec,
        **top,
        **(
            {"enum": [enum_updates.get(index, item) for index, item in enumerate(enum_values)]}
            if enum_values is not None and enum_updates
            else {}
        ),
    }


def _guardrailed_properties(
    raw_props: object,
    updates: Mapping[_SpecPath, str],
    prefix: _SpecPath,
) -> Mapping[str, Mapping[str, object]] | None:
    props: Final = _as_property_specs(raw_props)
    if props is None:
        return None
    return {prop: _guardrailed_leaf_spec(prop_spec, updates, (*prefix, prop)) for prop, prop_spec in props.items()}


def _guardrailed_spec(spec: Mapping[str, object], scanned: Sequence[tuple[_SpecText, str]]) -> Mapping[str, object]:
    if not scanned:
        return spec
    updates: Final[Mapping[_SpecPath, str]] = {text.path: guardrailed for text, guardrailed in scanned}
    updated_top: Final = _guardrailed_leaf_spec(spec, updates, ())
    has_props: Final = any(path[0] == "properties" for path in updates)
    has_item_props: Final = any(path[0] == "items" for path in updates)
    updated_props: Final = (
        _guardrailed_properties(spec.get("properties"), updates, ("properties",)) if has_props else None
    )
    raw_items: Final = _as_mapping(spec.get("items")) if has_item_props else None
    updated_item_props: Final = (
        _guardrailed_properties(raw_items.get("properties"), updates, ("items", "properties"))
        if raw_items is not None
        else None
    )
    return {
        **updated_top,
        **({"properties": updated_props} if updated_props is not None else {}),
        **(
            {"items": {**raw_items, "properties": updated_item_props}}
            if raw_items is not None and updated_item_props is not None
            else {}
        ),
    }


def _metadata_key(data: Mapping[str, object]) -> Literal["metadata", "litellm_metadata"]:
    return get_metadata_variable_name_from_kwargs(data)


def _metadata(data: Mapping[str, object]) -> Mapping[str, object]:
    return _METADATA_ADAPTER.validate_python(data.get(_metadata_key(data)) or {})


def _is_text_answer(answer: ExtractionAnswer, spec: Mapping[str, object] | None) -> bool:
    return answer.type == "text" or (answer.type == "list" and (spec is None or spec.get("items") == "text"))


def _object_property_specs(
    answer_type: str, spec: Mapping[str, object] | None
) -> Mapping[str, Mapping[str, object]] | None:
    if spec is None:
        return None
    if answer_type == "object":
        return _as_property_specs(spec.get("properties"))
    if answer_type == "list":
        items: Final = _as_mapping(spec.get("items"))
        return _as_property_specs(items.get("properties")) if items is not None else None
    return None


def _is_text_property(
    prop: str,
    value: JsonValue,
    prop_specs: Mapping[str, Mapping[str, object]] | None,
) -> bool:
    if not isinstance(value, str):
        return False
    if prop_specs is None:
        return True
    prop_spec: Final = prop_specs.get(prop)
    return prop_spec is None or prop_spec.get("type") == "text"


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


def _object_texts(
    name: str,
    key: _AnswerKey,
    item: int | None,
    obj: Mapping[str, JsonValue],
    whole: bool,
    prop_specs: Mapping[str, Mapping[str, object]] | None,
) -> tuple[_AnswerText, ...]:
    rewritable_key: Final = key in ("value", "guess")
    return tuple(
        _AnswerText(
            name=name,
            key=key,
            item=item,
            prop=prop,
            text=text,
            rewritable=rewritable_key and _is_text_property(prop, prop_val, prop_specs),
        )
        for prop, prop_val in obj.items()
        if (text := _scalar_text(prop_val, whole)) is not None
    )


def _list_item_texts(
    name: str,
    key: _AnswerKey,
    index: int,
    elem: JsonValue,
    whole: bool,
    rewritable: bool,
    prop_specs: Mapping[str, Mapping[str, object]] | None,
) -> tuple[_AnswerText, ...]:
    if isinstance(elem, dict):
        return _object_texts(name, key, index, elem, whole, prop_specs)
    text: Final = _scalar_text(elem, whole)
    if text is None:
        return ()
    return (
        _AnswerText(
            name=name,
            key=key,
            item=index,
            prop=None,
            text=text,
            rewritable=rewritable and isinstance(elem, str),
        ),
    )


def _value_texts(
    name: str,
    key: _AnswerKey,
    value: JsonValue,
    *,
    rewritable: bool = False,
    prop_specs: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[_AnswerText, ...]:
    whole: Final = key in _STATISTIC_KEYS
    if isinstance(value, dict):
        return _object_texts(name, key, None, value, whole, prop_specs)
    if isinstance(value, list):
        return tuple(
            chain.from_iterable(
                _list_item_texts(name, key, index, elem, whole, rewritable, prop_specs)
                for index, elem in enumerate(value)
            )
        )
    text: Final = _scalar_text(value, whole)
    return (
        ()
        if text is None
        else (_AnswerText(name=name, key=key, item=None, prop=None, text=text, rewritable=rewritable),)
    )


def _step_texts(name: str, step: JsonValue) -> tuple[_AnswerText, ...]:
    if not isinstance(step, dict):
        return ()
    return tuple(chain.from_iterable(_value_texts(name, key, step.get(key)) for key in _ESTIMATE_KEYS))


def _answer_texts(
    name: str, answer: ExtractionAnswer, spec: Mapping[str, object] | None = None
) -> tuple[_AnswerText, ...]:
    rewritable: Final = _is_text_answer(answer, spec)
    prop_specs: Final = _object_property_specs(answer.type, spec)
    return (
        *_value_texts(name, "value", answer.value, rewritable=rewritable, prop_specs=prop_specs),
        *_value_texts(name, "guess", answer.guess, rewritable=rewritable, prop_specs=prop_specs),
        *_value_texts(name, "mean", answer.mean),
        *_value_texts(name, "median", answer.median),
        *_value_texts(name, "p10", answer.p10),
        *_value_texts(name, "p90", answer.p90),
        *_value_texts(name, "quantiles", list(answer.quantiles or ())),
        *chain.from_iterable(_step_texts(name, step) for step in answer.steps or ()),
    )


def _scans_of(
    key: _AnswerKey, scanned: Sequence[tuple[_AnswerText, str]]
) -> Mapping[tuple[int | None, str | None], str]:
    return {(text.item, text.prop): guardrailed for text, guardrailed in scanned if text.key == key}


def _guardrailed_element(
    index: int,
    elem: JsonValue,
    scans: Mapping[tuple[int | None, str | None], str],
) -> JsonValue:
    if isinstance(elem, dict):
        return {prop: scans.get((index, prop), prop_val) for prop, prop_val in elem.items()}
    return scans.get((index, None), elem)


def _guardrailed_value(value: JsonValue, scans: Mapping[tuple[int | None, str | None], str]) -> JsonValue:
    if isinstance(value, dict):
        return {prop: scans.get((None, prop), prop_val) for prop, prop_val in value.items()}
    if isinstance(value, list):
        return [_guardrailed_element(index, elem, scans) for index, elem in enumerate(value)]
    return scans.get((None, None), value)


def _guardrailed_answer(
    answer: ExtractionAnswer,
    scanned: Sequence[tuple[_AnswerText, str]],
    drop_spans: bool,
) -> ExtractionAnswer:
    changed: Final = tuple((text, guardrailed) for text, guardrailed in scanned if text.text != guardrailed)
    if any(not text.rewritable for text, _ in changed):
        return answer.model_copy(update=_NULLED_ANSWER)
    value_scans: Final = _scans_of("value", changed)
    guess_scans: Final = _scans_of("guess", changed)
    update: Final = {
        **({"value": _guardrailed_value(answer.value, value_scans)} if value_scans else {}),
        **({"guess": _guardrailed_value(answer.guess, guess_scans)} if guess_scans else {}),
        **({"raw": None, "printed_value": None} if changed else {}),
        **({"span": None, "spans": None, "property_spans": None} if drop_spans else {}),
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
        request: Final = request_data or {}
        specs: Final = _FIELDS_ADAPTER.validate_python(request.get("fields") or {})
        texts: Final = tuple(
            chain.from_iterable(
                _answer_texts(name, answer, specs.get(name)) for name, answer in response.answers.items()
            )
        )
        state_texts: Final = (response.state_text,) if response.state_text else ()
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
        return response.model_copy(
            update={
                "answers": {
                    name: _guardrailed_answer(
                        answer,
                        answer_scans.get(name, ()),
                        drop_spans=state_text != response.state_text,
                    )
                    for name, answer in response.answers.items()
                },
                **({"state_text": state_text} if state_texts else {}),
            }
        )
