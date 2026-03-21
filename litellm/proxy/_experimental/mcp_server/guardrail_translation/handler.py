"""
MCP Guardrail Handler for Unified Guardrails.

Converts an MCP call_tool (name + arguments) into a single OpenAI-compatible
tool_call and passes it to apply_guardrail. Works with the synthetic payload
from ProxyLogging._convert_mcp_to_llm_format.

Note: For MCP tool definitions (schema) -> OpenAI tools=[], see
litellm.experimental_mcp_client.tools.transform_mcp_tool_to_openai_tool
when you have a full MCP Tool from list_tools. Here we only have the call
payload (name + arguments) so we just build the tool_call.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

from mcp.types import Tool as MCPTool

from litellm._logging import verbose_proxy_logger
from litellm.experimental_mcp_client.tools import transform_mcp_tool_to_openai_tool
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.types.llms.openai import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from mcp.types import CallToolResult

    from litellm.integrations.custom_guardrail import CustomGuardrail


class MCPGuardrailTranslationHandler(BaseTranslation):
    """Guardrail translation handler for MCP tool calls (passes a single tool_call to guardrail)."""

    async def process_input_messages(
        self,
        data: Dict[str, Any],
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional[Any] = None,
    ) -> Dict[str, Any]:
        mcp_tool_name = data.get("mcp_tool_name") or data.get("name")
        mcp_arguments = data.get("mcp_arguments") or data.get("arguments")
        mcp_tool_description = data.get("mcp_tool_description") or data.get(
            "description"
        )
        if mcp_arguments is None or not isinstance(mcp_arguments, dict):
            mcp_arguments = {}

        if not mcp_tool_name:
            verbose_proxy_logger.debug("MCP Guardrail: mcp_tool_name missing")
            return data

        # Convert MCP input via transform_mcp_tool_to_openai_tool, then map to litellm
        # ChatCompletionToolParam (openai SDK type has incompatible strict/cache_control).
        mcp_tool = MCPTool(
            name=mcp_tool_name,
            description=mcp_tool_description or "",
            inputSchema={},  # Call payload has no schema; guardrail gets args from request_data
        )
        openai_tool = transform_mcp_tool_to_openai_tool(mcp_tool)
        fn = openai_tool["function"]
        tool_def: ChatCompletionToolParam = {
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
        inputs: GenericGuardrailAPIInputs = GenericGuardrailAPIInputs(
            tools=[tool_def],
        )

        await guardrail_to_apply.apply_guardrail(
            inputs=inputs,
            request_data=data,
            input_type="request",
            logging_obj=litellm_logging_obj,
        )
        return data

    async def process_output_response(
        self,
        response: "CallToolResult",
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional[Any] = None,
        user_api_key_dict: Optional[Any] = None,
    ) -> Any:
        verbose_proxy_logger.debug(
            "MCP Guardrail: Output processing not implemented for MCP tools",
        )
        return response
