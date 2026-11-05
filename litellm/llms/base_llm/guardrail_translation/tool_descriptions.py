from __future__ import annotations

import copy
import functools
import itertools
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Final, TypeAlias

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.guardrail_translation.utils import (
    effective_scan_only_tool_results_for_guardrail,
    unappliable_request_rewrite,
)
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail

DescriptionPath: TypeAlias = tuple[str | int, ...]
_Located: TypeAlias = tuple[DescriptionPath, JsonValue]

_TOOLS: Final = TypeAdapter(list[JsonValue])


def effective_scan_tool_descriptions_as_text_for_guardrail(guardrail_to_apply: CustomGuardrail) -> bool:
    return (
        guardrail_to_apply.supports_scan_tool_descriptions_as_text()
        and getattr(guardrail_to_apply, "scan_tool_descriptions_as_text", None) is True
        and not effective_scan_only_tool_results_for_guardrail(guardrail_to_apply)
    )


_TOOL_SCHEMA_KEYS: Final = ("parameters", "input_schema")
_SUBSCHEMA_KEYWORDS: Final = ("items", "additionalProperties", "not")
_SUBSCHEMA_LIST_KEYWORDS: Final = ("items", "prefixItems", "anyOf", "oneOf", "allOf")
_SUBSCHEMA_MAP_KEYWORDS: Final = ("properties", "patternProperties", "$defs", "definitions")


def tool_description_entries(tools: Sequence[JsonValue]) -> tuple[tuple[DescriptionPath, str], ...]:
    holders: Final = tuple(
        itertools.chain.from_iterable(_tool_holders(tool, (index,)) for index, tool in enumerate(tools))
    )
    return tuple(
        itertools.chain(
            itertools.chain.from_iterable(_own_description(holder, path) for path, holder in holders),
            itertools.chain.from_iterable(_own_description(schema, path) for path, schema in _schemas(holders)),
        )
    )


def _tool_holders(tool: JsonValue, path: DescriptionPath) -> Iterator[tuple[DescriptionPath, Mapping[str, JsonValue]]]:
    if not isinstance(tool, dict):
        return
    yield path, tool
    if isinstance(function := tool.get("function"), dict):
        yield (*path, "function"), function
    if tool.get("type") == "namespace" and isinstance(members := tool.get("tools"), list):
        for index, member in enumerate(members):
            if isinstance(member, dict):
                yield (*path, "tools", index), member


def _schemas(
    holders: Sequence[tuple[DescriptionPath, Mapping[str, JsonValue]]],
) -> Iterator[tuple[DescriptionPath, Mapping[str, JsonValue]]]:
    roots: Final = itertools.chain.from_iterable(
        (((*path, key), holder.get(key)) for key in _TOOL_SCHEMA_KEYS) for path, holder in holders
    )
    pending: Final[list[_Located]] = list(roots)  # mutable-ok: work stack keeps the schema walk non-recursive
    while pending:
        path, schema = pending.pop()
        if isinstance(schema, dict):
            yield path, schema
            pending.extend(_subschemas(schema, path))


def _subschemas(schema: Mapping[str, JsonValue], path: DescriptionPath) -> Iterator[tuple[DescriptionPath, JsonValue]]:
    for key in _SUBSCHEMA_KEYWORDS:
        yield (*path, key), schema.get(key)
    for key in _SUBSCHEMA_LIST_KEYWORDS:
        if isinstance(children := schema.get(key), list):
            for index, child in enumerate(children):
                yield (*path, key, index), child
    for key in _SUBSCHEMA_MAP_KEYWORDS:
        if isinstance(named := schema.get(key), dict):
            for name, child in named.items():
                yield (*path, key, name), child


def _own_description(node: Mapping[str, JsonValue], path: DescriptionPath) -> Iterator[tuple[DescriptionPath, str]]:
    description: Final = node.get("description")
    if isinstance(description, str) and description.strip():
        yield (*path, "description"), description


def _child(node: JsonValue, key: str | int) -> JsonValue:
    if isinstance(node, dict) and isinstance(key, str):
        return node.get(key)
    if isinstance(node, list) and isinstance(key, int) and 0 <= key < len(node):
        return node[key]
    return None


def with_replaced_descriptions(tools: Sequence[JsonValue], replacements: Mapping[DescriptionPath, str]) -> JsonValue:
    rewritten: Final = copy.deepcopy(list(tools))
    for path, text in replacements.items():
        holder = functools.reduce(_child, path[:-1], rewritten)
        if isinstance(holder, dict) and isinstance(key := path[-1], str):
            holder[key] = text
    return rewritten


async def scan_tool_descriptions(
    data: dict[str, object],  # mutable-ok: apply_guardrail contract takes the request body as a dict
    guardrail_to_apply: CustomGuardrail,
) -> dict[str, object]:  # mutable-ok: pre-call hooks hand the request body on as a dict
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    if not effective_scan_tool_descriptions_as_text_for_guardrail(guardrail_to_apply):
        return data
    raw_tools: Final = data.get("tools")
    if not isinstance(raw_tools, list) or not raw_tools:
        return data
    try:
        tools: Final = _TOOLS.validate_python(raw_tools)
    except ValidationError:
        verbose_proxy_logger.warning(
            "Guardrail %s: tools[] is not plain JSON, so its descriptions were not scanned",
            guardrail_to_apply.guardrail_name,
        )
        return data
    entries: Final = tool_description_entries(tools)
    if not entries:
        return data
    inputs: Final = GenericGuardrailAPIInputs(texts=[text for _, text in entries])
    model: Final = data.get("model")
    if isinstance(model, str):
        inputs["model"] = model
    logging_obj: Final = data.get("litellm_logging_obj")
    guardrailed_inputs: Final = await guardrail_to_apply.apply_guardrail(  # pyright: ignore[reportUnknownMemberType]  # apply_guardrail types request_data as a bare dict
        inputs=inputs,
        request_data=data,
        input_type="request",
        logging_obj=logging_obj if isinstance(logging_obj, LiteLLMLoggingObj) else None,
    )
    returned_texts: Final = guardrailed_inputs.get("texts")
    if returned_texts is None:
        return data
    if len(returned_texts) != len(entries):
        raise unappliable_request_rewrite(guardrail_to_apply.guardrail_name)
    replacements: Final = {
        path: new_text
        for (path, old_text), new_text in zip(entries, returned_texts, strict=True)
        if new_text != old_text
    }
    if not replacements:
        return data
    return {**data, "tools": with_replaced_descriptions(tools, replacements)}
