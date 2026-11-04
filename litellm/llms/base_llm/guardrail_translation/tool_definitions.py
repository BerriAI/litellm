from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Final, TypeAlias

from pydantic import JsonValue, TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.guardrail_translation.utils import (
    effective_scan_only_tool_results_for_guardrail,
    unappliable_request_rewrite,
)
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail

DescriptionPath: TypeAlias = tuple[str | int, ...]

_TOOLS: Final = TypeAdapter(list[JsonValue])


def effective_scan_tool_definitions_for_guardrail(guardrail_to_apply: CustomGuardrail) -> bool:
    if guardrail_to_apply.natively_scans_tool_definitions:
        return False
    if effective_scan_only_tool_results_for_guardrail(guardrail_to_apply):
        return False
    per: Final[object] = getattr(guardrail_to_apply, "scan_tool_definitions_in_guardrail", None)
    if per is not None:
        return bool(per)
    return litellm.scan_tool_definitions_in_guardrail


def tool_description_entries(tools: Sequence[JsonValue]) -> tuple[tuple[DescriptionPath, str], ...]:
    return tuple(_description_entries(list(tools), ()))


def _description_entries(node: JsonValue, path: DescriptionPath) -> Iterator[tuple[DescriptionPath, str]]:
    if isinstance(node, list):
        for index, item in enumerate(node):
            yield from _description_entries(item, (*path, index))
        return
    if not isinstance(node, dict):
        return
    for key, value in node.items():
        if key == "description" and isinstance(value, str):
            if value.strip():
                yield (*path, key), value
            continue
        yield from _description_entries(value, (*path, key))


def with_replaced_descriptions(
    node: JsonValue, replacements: Mapping[DescriptionPath, str], path: DescriptionPath = ()
) -> JsonValue:
    if isinstance(node, list):
        return [with_replaced_descriptions(item, replacements, (*path, index)) for index, item in enumerate(node)]
    if not isinstance(node, dict):
        return node
    return {
        key: (
            replacements.get((*path, key), value)
            if key == "description" and isinstance(value, str)
            else with_replaced_descriptions(value, replacements, (*path, key))
        )
        for key, value in node.items()
    }


async def scan_tool_definitions(
    data: dict[str, object],  # mutable-ok: apply_guardrail contract takes the request body as a dict
    guardrail_to_apply: CustomGuardrail,
) -> dict[str, object]:  # mutable-ok: pre-call hooks hand the request body on as a dict
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    if not effective_scan_tool_definitions_for_guardrail(guardrail_to_apply):
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
