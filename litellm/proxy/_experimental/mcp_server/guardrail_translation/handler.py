"""
MCP Guardrail Handler for Unified Guardrails.

Converts an MCP call_tool (name + arguments) into the OpenAI-compatible shape
apply_guardrail expects: the tool as a single-entry ``tools`` definition, and
every string leaf of the call arguments as ``texts`` so text guardrails can
detect and mask sensitive values in the payload. Works with the synthetic
request from ProxyLogging._convert_mcp_to_llm_format.

Note: For MCP tool definitions (schema) -> OpenAI tools=[], see
litellm.experimental_mcp_client.tools.transform_mcp_tool_to_openai_tool
when you have a full MCP Tool from list_tools. Here we only have the call
payload (name + arguments) so we just build the tool definition.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final

from fastapi import HTTPException
from mcp.types import Tool as MCPTool

from litellm._logging import verbose_proxy_logger
from litellm.experimental_mcp_client.tools import transform_mcp_tool_to_openai_tool
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.proxy._experimental.mcp_server.utils import (
    MAX_STRUCTURED_CONTENT_SCAN_DEPTH,
    JSONLeafPath,
    json_string_leaves,
    json_unrewritable_labels,
    mcp_content_item_text,
    mcp_tool_result_content_list,
    mcp_tool_result_structured_content,
    set_mcp_tool_result_structured_content,
    with_json_string_leaves,
    with_mcp_content_item_text,
)
from litellm.types.llms.openai import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from mcp.types import CallToolResult

    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


def _blocked(reason: str) -> HTTPException:
    return HTTPException(status_code=400, detail={"error": f"Content blocked: {reason}"})


def _too_deeply_nested() -> HTTPException:
    return _blocked(
        f"MCP tool call arguments exceed the maximum nesting depth of {MAX_STRUCTURED_CONTENT_SCAN_DEPTH} "
        "and cannot be scanned by the configured guardrail"
    )


def _argument_replacements(
    argument_leaves: tuple[tuple[JSONLeafPath, str], ...],
    masked_texts: Sequence[str] | None,
) -> Mapping[JSONLeafPath, str]:
    """Positionally pair the guardrail's returned texts with the leaves they came from.

    Only leaves the guardrail actually rewrote are returned, so a guardrail that
    detects nothing leaves the outbound tool call byte-identical. A guardrail that
    returns the wrong number of texts is paired against nothing, because a
    positional write-back would scramble the arguments rather than mask them.
    """
    usable: Final = masked_texts if masked_texts is not None and len(masked_texts) == len(argument_leaves) else ()
    if masked_texts is not None and len(masked_texts) != len(argument_leaves):
        verbose_proxy_logger.warning(
            "MCP Guardrail: guardrail returned %d texts for %d tool call argument strings; leaving arguments unmasked",
            len(masked_texts),
            len(argument_leaves),
        )
    return {path: masked for (path, original), masked in zip(argument_leaves, usable) if masked != original}


def _conflicting_rewrite_paths(
    scanned_leaves: tuple[tuple[JSONLeafPath, str], ...],
    current_leaves: tuple[tuple[JSONLeafPath, str], ...],
    replacements: Mapping[JSONLeafPath, str],
) -> tuple[JSONLeafPath, ...]:
    """Paths another guardrail already rewrote differently from what this one wants.

    Guardrails opted into ``run_in_parallel`` all scan the same payload snapshot, so
    each one returns a full replacement string derived from the *original* leaf. Two
    of them rewriting one leaf to different values cannot be merged: writing either
    result discards the other guardrail's redaction. A leaf still holding the text
    this guardrail was handed, or already holding this guardrail's own replacement,
    is safe to write; the latter is how a guardrail that masks the arguments itself
    as well as through ``texts`` gets there first. Anything else fails closed,
    including a payload reshaped so the leaves no longer line up, because the
    write-back is positional and would land a redaction on the wrong value.
    """
    return tuple(
        path
        for (path, scanned), (current_path, current) in zip(scanned_leaves, current_leaves)
        if path in replacements and (current_path != path or current not in (scanned, replacements[path]))
    )


def _conflicting_rewrite(paths: tuple[JSONLeafPath, ...]) -> HTTPException:
    return _blocked(
        "two guardrails running concurrently rewrote the same MCP tool call "
        f"argument{'s' if len(paths) > 1 else ''} "
        f"({', '.join('.'.join(str(part) for part in path) for path in paths)}); "
        "their redactions cannot be merged. Remove run_in_parallel from one of them so they "
        "run in sequence."
    )


class MCPGuardrailTranslationHandler(BaseTranslation):
    """Guardrail translation handler for MCP tool calls (passes a single tool_call to guardrail)."""

    async def process_input_messages(
        self,
        data: dict[str, Any],
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> dict[str, Any]:
        mcp_tool_name: Final = data.get("mcp_tool_name") or data.get("name")
        mcp_arguments: Final[object] = data.get("mcp_arguments") or data.get("arguments")
        mcp_tool_description: Final = data.get("mcp_tool_description") or data.get("description")

        if not mcp_tool_name:
            verbose_proxy_logger.debug("MCP Guardrail: mcp_tool_name missing")
            return data

        # Convert MCP input via transform_mcp_tool_to_openai_tool, then map to litellm
        # ChatCompletionToolParam (openai SDK type has incompatible strict/cache_control).
        mcp_tool: Final = MCPTool(
            name=mcp_tool_name,
            description=mcp_tool_description or "",
            inputSchema={},  # Call payload has no schema; guardrail gets args from request_data
        )
        openai_tool: Final = transform_mcp_tool_to_openai_tool(mcp_tool)
        fn: Final = openai_tool["function"]
        tool_def: Final[ChatCompletionToolParam] = {
            "type": "function",
            "function": ChatCompletionToolParamFunctionChunk(
                name=fn["name"],
                description=fn.get("description") or "",
                parameters=fn.get("parameters")
                or {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                strict=fn.get("strict", False) or False,  # Default to False if None
            ),
        }
        argument_leaves: Final = json_string_leaves(mcp_arguments)
        if argument_leaves is None:
            raise _too_deeply_nested()
        inputs: Final[GenericGuardrailAPIInputs] = GenericGuardrailAPIInputs(
            tools=[tool_def],
            texts=[text for _, text in argument_leaves],
        )

        guarded: Final = await guardrail_to_apply.apply_guardrail(
            inputs=inputs,
            request_data=data,
            input_type="request",
            logging_obj=litellm_logging_obj,
        )
        replacements: Final = _argument_replacements(
            argument_leaves=argument_leaves,
            masked_texts=guarded.get("texts") if guarded else None,
        )
        if not replacements:
            return data

        current_arguments: Final[object] = data.get("mcp_arguments") or data.get("arguments")
        current_leaves: Final = json_string_leaves(current_arguments)
        if current_leaves is None:
            raise _too_deeply_nested()
        conflicting: Final = _conflicting_rewrite_paths(argument_leaves, current_leaves, replacements)
        if conflicting:
            raise _conflicting_rewrite(conflicting)
        masked_arguments: Final = with_json_string_leaves(current_arguments, replacements)
        data["mcp_arguments"] = masked_arguments
        data["modified_arguments"] = masked_arguments
        return data

    async def process_output_response(
        self,
        response: "CallToolResult",
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: "LiteLLMLoggingObj | None" = None,
        user_api_key_dict: Any | None = None,
        request_data: dict | None = None,
    ) -> Any:
        """Scan the text content of an MCP tool result and write masked text back.

        The content list is rewritten in place (only the entries the guardrail
        actually changed) rather than returned as a new result: the same object is
        already referenced by the logging payload captured before this hook runs,
        so a copy would leave the unmasked text in the spend log / span. A
        guardrail that rejects the result raises, and the exception propagates to
        the caller.

        ``structuredContent`` is scanned and masked too, in the same
        ``apply_guardrail`` call: it is serialized to the client alongside
        ``content``, so a value living only there would otherwise reach the
        client unscanned.
        """
        content: Final = mcp_tool_result_content_list(response)
        text_blocks: Final = (
            tuple(
                (index, text) for index, item in enumerate(content) if (text := mcp_content_item_text(item)) is not None
            )
            if content is not None
            else ()
        )

        structured: Final = mcp_tool_result_structured_content(response)
        structured_leaves: Final = json_string_leaves(structured) if structured is not None else ()
        structured_labels: Final = json_unrewritable_labels(structured) if structured is not None else ()
        if structured_leaves is None or structured_labels is None:
            raise _blocked(
                "MCP tool result structuredContent is nested too deeply to be scanned by the configured guardrail"
            )

        if not text_blocks and not structured_leaves and not structured_labels:
            verbose_proxy_logger.debug("MCP Guardrail: tool result has no scannable text, nothing to do")
            return response

        originals: Final = (
            tuple(text for _, text in text_blocks) + tuple(text for _, text in structured_leaves) + structured_labels
        )
        guardrailed_inputs: Final = await guardrail_to_apply.apply_guardrail(
            inputs=GenericGuardrailAPIInputs(texts=list(originals)),
            request_data=request_data if request_data is not None else {},
            input_type="response",
            logging_obj=litellm_logging_obj,
        )
        masked_texts: Final = guardrailed_inputs.get("texts") if guardrailed_inputs else None
        if masked_texts is None:
            return response
        if len(masked_texts) != len(originals):
            verbose_proxy_logger.warning(
                "MCP Guardrail: guardrail returned %d texts for %d tool result texts; leaving the result unmasked",
                len(masked_texts),
                len(originals),
            )
            return response

        split: Final = len(text_blocks)
        if content is not None:
            for (index, original), masked in zip(text_blocks, masked_texts[:split]):
                if masked != original:
                    content[index] = with_mcp_content_item_text(content[index], masked)

        label_start: Final = split + len(structured_leaves)
        if any(masked != original for original, masked in zip(structured_labels, masked_texts[label_start:])):
            raise _blocked(
                "MCP tool result matched a masking rule on a non-rewritable field "
                "(a structuredContent key or numeric value), which cannot be redacted without changing "
                "the payload contract"
            )

        structured_replacements: Final = {
            path: masked
            for (path, original), masked in zip(structured_leaves, masked_texts[split:label_start])
            if masked != original
        }
        if structured_replacements:
            set_mcp_tool_result_structured_content(
                response, with_json_string_leaves(structured, structured_replacements)
            )
        return response
