from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union

from litellm._logging import verbose_logger
from litellm.proxy._experimental.mcp_server.utils import get_server_name_prefix_tool_mcp
from litellm.responses.main import aresponses
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.types.llms.openai import ResponsesAPIResponse, ToolParam

if TYPE_CHECKING:
    from mcp.types import Tool as MCPTool
else:
    MCPTool = Any

LITELLM_PROXY_MCP_SERVER_URL = "litellm_proxy"
LITELLM_PROXY_MCP_SERVER_URL_PREFIX = f"{LITELLM_PROXY_MCP_SERVER_URL}/mcp/"


class LiteLLM_Proxy_MCP_Handler:
    """
    Helper class with static methods for MCP integration with Responses API.

    This handles when a user passes mcp server_url="litellm_proxy" in their tools.
    """

    @staticmethod
    def _should_use_litellm_mcp_gateway(tools: Optional[Iterable[ToolParam]]) -> bool:
        """
        Returns True if the user passed a MCP tool with server_url="litellm_proxy"
        """
        if tools:
            for tool in tools:
                if isinstance(tool, dict) and tool.get("type") == "mcp":
                    server_url = tool.get("server_url", "")
                    if isinstance(server_url, str) and server_url.startswith(
                        LITELLM_PROXY_MCP_SERVER_URL
                    ):
                        return True
        return False

    @staticmethod
    def _parse_mcp_tools(
        tools: Optional[Iterable[ToolParam]],
    ) -> Tuple[List[ToolParam], List[Any]]:
        """
        Parse tools and separate MCP tools with litellm_proxy from other tools.

        Returns:
            Tuple of (mcp_tools_with_litellm_proxy, other_tools)
        """
        mcp_tools_with_litellm_proxy: List[ToolParam] = []
        other_tools: List[Any] = []

        if tools:
            for tool in tools:
                if isinstance(tool, dict) and tool.get("type") == "mcp":
                    server_url = tool.get("server_url", "")
                    if isinstance(server_url, str) and server_url.startswith(
                        LITELLM_PROXY_MCP_SERVER_URL
                    ):
                        mcp_tools_with_litellm_proxy.append(tool)
                    else:
                        other_tools.append(tool)
                else:
                    other_tools.append(tool)

        return mcp_tools_with_litellm_proxy, other_tools

    @staticmethod
    async def _get_mcp_tools_from_manager(
        user_api_key_auth: Any,
        mcp_tools_with_litellm_proxy: Optional[Iterable[ToolParam]],
    ) -> tuple[List[MCPTool], List[str]]:
        """
        Get available tools from the MCP server manager.

        Args:
            user_api_key_auth: User authentication info for access control
            mcp_tools_with_litellm_proxy: ToolParam objects with server_url starting with "litellm_proxy"

        Returns:
            List of MCP tools
            List names of allowed MCP servers
        """
        from litellm.proxy._experimental.mcp_server.server import (
            _get_tools_from_mcp_servers,
            _get_allowed_mcp_servers_from_mcp_server_names,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        mcp_servers: List[str] = []
        if mcp_tools_with_litellm_proxy:
            for _tool in mcp_tools_with_litellm_proxy:
                # if user specifies servers as server_url: litellm_proxy/mcp/zapier,github then return zapier,github
                server_url = (
                    _tool.get("server_url", "") if isinstance(_tool, dict) else ""
                )
                if isinstance(server_url, str) and server_url.startswith(
                    LITELLM_PROXY_MCP_SERVER_URL_PREFIX
                ):
                    mcp_servers.append(server_url.split("/")[-1])

        tools = await _get_tools_from_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=None,
            mcp_servers=mcp_servers,
            mcp_server_auth_headers=None,
        )
        allowed_mcp_server_ids = (
            await global_mcp_server_manager.get_allowed_mcp_servers(user_api_key_auth)
        )
        allowed_mcp_servers = global_mcp_server_manager.get_mcp_servers_from_ids(
            allowed_mcp_server_ids
        )

        allowed_mcp_servers = await _get_allowed_mcp_servers_from_mcp_server_names(
            mcp_servers=mcp_servers,
            allowed_mcp_servers=allowed_mcp_servers,
        )

        server_names: List[str] = []
        for server in allowed_mcp_servers:
            if server is None:
                continue
            server_name = getattr(server, "server_name", None) or getattr(
                server, "alias", None
            ) or getattr(server, "name", None)
            if isinstance(server_name, str):
                server_names.append(server_name)

        return tools, server_names

    @staticmethod
    def _deduplicate_mcp_tools(
        mcp_tools: List[MCPTool], allowed_mcp_servers: List[str]
    ) -> tuple[List[MCPTool], dict[str, str]]:
        """
        Deduplicate MCP tools by name, keeping the first occurrence of each tool.

        Args:
            mcp_tools: List of MCP tools that may contain duplicates

        Returns:
            List of deduplicated MCP tools
            The returned dictionary maps each tool_name to the server_name
        """
        seen_names = set()
        deduplicated_tools = []
        tool_server_map: dict[str, str] = {}

        for tool in mcp_tools:
            if isinstance(tool, dict):
                tool_name = tool.get("name")
            else:
                tool_name = getattr(tool, "name", None)

            if tool_name and tool_name not in seen_names:
                seen_names.add(tool_name)
                deduplicated_tools.append(tool)
                if len(allowed_mcp_servers) == 1:
                    tool_server_map[tool_name] = allowed_mcp_servers[0]
                else:
                    tool_server_map[tool_name], _ = get_server_name_prefix_tool_mcp(
                        tool_name
                    )

        return deduplicated_tools, tool_server_map

    @staticmethod
    def _filter_mcp_tools_by_allowed_tools(
        mcp_tools: List[MCPTool], mcp_tools_with_litellm_proxy: List[ToolParam]
    ) -> List[MCPTool]:
        """Filter MCP tools based on allowed_tools parameter from the original tool configs."""
        # Collect all allowed tool names from all MCP tool configs
        allowed_tool_names = set()
        for tool_config in mcp_tools_with_litellm_proxy:
            if isinstance(tool_config, dict) and "allowed_tools" in tool_config:
                allowed_tools = tool_config.get("allowed_tools", [])
                if isinstance(allowed_tools, list):
                    allowed_tool_names.update(allowed_tools)

        # If no allowed_tools specified, return all tools
        if not allowed_tool_names:
            return mcp_tools

        # Filter tools based on allowed names
        filtered_tools = []
        for mcp_tool in mcp_tools:
            if isinstance(mcp_tool, dict):
                tool_name = mcp_tool.get("name")
            else:
                tool_name = getattr(mcp_tool, "name", None)

            if tool_name and tool_name in allowed_tool_names:
                filtered_tools.append(mcp_tool)

        return filtered_tools

    @staticmethod
    async def _process_mcp_tools_to_openai_format(
        user_api_key_auth: Any, mcp_tools_with_litellm_proxy: List[ToolParam]
    ) -> tuple[List[Any], dict[str, str]]:
        """
        Centralized method to process MCP tools through the complete pipeline.

        Args:
            user_api_key_auth: User authentication info for access control
            mcp_tools_with_litellm_proxy: ToolParam objects with server_url starting with "litellm_proxy"

        Returns:
            List of tools in OpenAI format ready to be sent to the LLM
            The returned dictionary maps each tool_name to the server_name
        """
        (
            deduplicated_mcp_tools,
            tool_server_map,
        ) = await LiteLLM_Proxy_MCP_Handler._process_mcp_tools_without_openai_transform(
            user_api_key_auth,
            mcp_tools_with_litellm_proxy,
        )

        openai_tools = LiteLLM_Proxy_MCP_Handler._transform_mcp_tools_to_openai(
            deduplicated_mcp_tools
        )

        return openai_tools, tool_server_map

    @staticmethod
    async def _process_mcp_tools_without_openai_transform(
        user_api_key_auth: Any, mcp_tools_with_litellm_proxy: List[ToolParam]
    ) -> tuple[List[Any], dict[str, str]]:
        """
        Process MCP tools through filtering and deduplication pipeline without OpenAI transformation.
        This is useful for cases where we need the original MCP tool objects (e.g., for events).

        Args:
            user_api_key_auth: User authentication info for access control
            mcp_tools_with_litellm_proxy: ToolParam objects with server_url starting with "litellm_proxy"

        Returns:
            List of filtered and deduplicated MCP tools in their original format
        """
        if not mcp_tools_with_litellm_proxy:
            return [], {}

        # Step 1: Fetch MCP tools from manager
        (
            mcp_tools_fetched,
            allowed_mcp_servers,
        ) = await LiteLLM_Proxy_MCP_Handler._get_mcp_tools_from_manager(
            user_api_key_auth=user_api_key_auth,
            mcp_tools_with_litellm_proxy=mcp_tools_with_litellm_proxy,
        )

        # Step 2: Filter tools based on allowed_tools parameter
        filtered_mcp_tools = (
            LiteLLM_Proxy_MCP_Handler._filter_mcp_tools_by_allowed_tools(
                mcp_tools=mcp_tools_fetched,
                mcp_tools_with_litellm_proxy=mcp_tools_with_litellm_proxy,
            )
        )

        # Step 3: Deduplicate tools after filtering
        (
            deduplicated_mcp_tools,
            tool_server_map,
        ) = LiteLLM_Proxy_MCP_Handler._deduplicate_mcp_tools(
            filtered_mcp_tools, allowed_mcp_servers
        )

        return deduplicated_mcp_tools, tool_server_map

    @staticmethod
    def _transform_mcp_tools_to_openai(mcp_tools: List[Any]) -> List[Any]:
        """Transform MCP tools to OpenAI-compatible format."""
        from litellm.experimental_mcp_client.tools import (
            transform_mcp_tool_to_openai_responses_api_tool,
        )

        openai_tools = []
        for mcp_tool in mcp_tools:
            openai_tool = transform_mcp_tool_to_openai_responses_api_tool(mcp_tool)
            openai_tools.append(openai_tool)

        return openai_tools

    @staticmethod
    def _should_auto_execute_tools(
        mcp_tools_with_litellm_proxy: Union[List[Dict[str, Any]], List[ToolParam]],
    ) -> bool:
        """Check if we should auto-execute tool calls.

        Only auto-execute tools if user passed a MCP tool with require_approval set to "never".


        """
        for tool in mcp_tools_with_litellm_proxy:
            if isinstance(tool, dict):
                if tool.get("require_approval") == "never":
                    return True
            elif getattr(tool, "require_approval", None) == "never":
                return True
        return False

    @staticmethod
    def _extract_tool_calls_from_response(response: ResponsesAPIResponse) -> List[Any]:
        """Extract tool calls from the response output."""
        tool_calls: List[Any] = []
        for output_item in response.output:
            # Check if this is a function call output item
            if (
                isinstance(output_item, dict)
                and output_item.get("type") == "function_call"
            ):
                tool_calls.append(output_item)
            elif (
                hasattr(output_item, "type")
                and getattr(output_item, "type") == "function_call"
            ):
                # Handle pydantic model case
                tool_calls.append(output_item)

        return tool_calls

    @staticmethod
    def _extract_tool_call_details(
        tool_call,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract tool name, arguments, and call_id from a tool call."""
        if isinstance(tool_call, dict):
            tool_name = tool_call.get("name")
            tool_arguments = tool_call.get("arguments")
            tool_call_id = tool_call.get("call_id") or tool_call.get("id")
        else:
            tool_name = getattr(tool_call, "name", None)
            tool_arguments = getattr(tool_call, "arguments", None)
            tool_call_id = getattr(tool_call, "call_id", None) or getattr(
                tool_call, "id", None
            )

        return tool_name, tool_arguments, tool_call_id

    @staticmethod
    def _parse_tool_arguments(tool_arguments: Any) -> Dict[str, Any]:
        """Parse tool arguments, handling both string and dict formats."""
        import json

        if isinstance(tool_arguments, str):
            try:
                return json.loads(tool_arguments)
            except json.JSONDecodeError:
                return {}
        else:
            return tool_arguments or {}

    @staticmethod
    def _parse_mcp_result(result: Any) -> str:
        """Parse MCP tool call result and extract meaningful content."""
        if not result or not hasattr(result, "content") or not result.content:
            return "Tool executed successfully"

        # Import MCP content types for isinstance checks
        try:
            from mcp.types import EmbeddedResource, ImageContent, TextContent
        except ImportError:
            # Fallback to generic handling if MCP types not available
            return "Tool executed successfully"

        text_parts = []
        other_content_types = []

        for content_item in result.content:
            if isinstance(content_item, TextContent):
                # Text content - extract the text
                text_parts.append(str(content_item.text))
            elif isinstance(content_item, ImageContent):
                # Image content
                other_content_types.append("Image")
            elif isinstance(content_item, EmbeddedResource):
                # Embedded resource
                other_content_types.append("EmbeddedResource")
            else:
                # Other unknown content types
                content_type = type(content_item).__name__
                other_content_types.append(content_type)

        # Combine text parts if any
        result_text = " ".join(text_parts) if text_parts else ""

        # Add info about other content types
        if other_content_types:
            other_info = f"[Generated {', '.join(other_content_types)}]"
            result_text = f"{result_text} {other_info}".strip()

        return result_text or "Tool executed successfully"

    @staticmethod
    async def _execute_tool_calls(
        tool_server_map: dict[str, str], tool_calls: List[Any], user_api_key_auth: Any
    ) -> List[Dict[str, Any]]:
        """Execute tool calls and return results."""
        from fastapi import HTTPException

        from litellm.exceptions import BlockedPiiEntityError, GuardrailRaisedException
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        tool_results = []
        tool_call_id: Optional[str] = None
        for tool_call in tool_calls:
            try:
                (
                    tool_name,
                    tool_arguments,
                    tool_call_id,
                ) = LiteLLM_Proxy_MCP_Handler._extract_tool_call_details(tool_call)

                if not tool_name:
                    verbose_logger.warning(f"Tool call missing name: {tool_call}")
                    continue

                parsed_arguments = LiteLLM_Proxy_MCP_Handler._parse_tool_arguments(
                    tool_arguments
                )

                # Import here to avoid circular import
                from litellm.proxy.proxy_server import proxy_logging_obj

                server_name = tool_server_map[tool_name]

                result = await global_mcp_server_manager.call_tool(
                    server_name=server_name,
                    name=tool_name,
                    arguments=parsed_arguments,
                    user_api_key_auth=user_api_key_auth,
                    proxy_logging_obj=proxy_logging_obj,
                )

                # Format result for inclusion in response
                result_text = LiteLLM_Proxy_MCP_Handler._parse_mcp_result(result)
                tool_results.append(
                    {"tool_call_id": tool_call_id, "result": result_text}
                )

            except BlockedPiiEntityError as e:
                verbose_logger.error(
                    f"BlockedPiiEntityError in MCP tool call: {str(e)}"
                )
                error_message = f"Tool call blocked: PII entity '{getattr(e, 'entity_type', 'unknown')}' detected by guardrail '{getattr(e, 'guardrail_name', 'unknown')}'. {str(e)}"
                tool_results.append(
                    {"tool_call_id": tool_call_id, "result": error_message}
                )
            except GuardrailRaisedException as e:
                verbose_logger.error(
                    f"GuardrailRaisedException in MCP tool call: {str(e)}"
                )
                error_message = f"Tool call blocked: Guardrail '{getattr(e, 'guardrail_name', 'unknown')}' violation. {str(e)}"
                tool_results.append(
                    {"tool_call_id": tool_call_id, "result": error_message}
                )
            except HTTPException as e:
                verbose_logger.error(f"HTTPException in MCP tool call: {str(e)}")
                error_message = f"Tool call failed: {str(e.detail) if hasattr(e, 'detail') else str(e)}"
                tool_results.append(
                    {"tool_call_id": tool_call_id, "result": error_message}
                )
            except Exception as e:
                verbose_logger.exception(f"Error executing MCP tool call: {e}")
                tool_results.append(
                    {
                        "tool_call_id": tool_call_id,
                        "result": f"Error executing tool: {str(e)}",
                    }
                )

        return tool_results

    @staticmethod
    def _create_follow_up_input(
        response: ResponsesAPIResponse,
        tool_results: List[Dict[str, Any]],
        original_input: Any = None,
    ) -> List[Any]:
        """Create follow-up input with tool results in proper format."""
        follow_up_input: List[Any] = []

        # Add original user input if available to maintain conversation context
        if original_input:
            if isinstance(original_input, str):
                follow_up_input.append(
                    {"type": "message", "role": "user", "content": original_input}
                )
            elif isinstance(original_input, list):
                follow_up_input.extend(original_input)
            else:
                follow_up_input.append(original_input)

        # Add the assistant message with function calls
        assistant_message_content: List[Any] = []
        function_calls: List[Dict[str, Any]] = []

        for output_item in response.output:
            if isinstance(output_item, dict):
                if output_item.get("type") == "function_call":
                    call_id = output_item.get("call_id") or output_item.get("id")
                    name = output_item.get("name")
                    arguments = output_item.get("arguments")

                    # Only add if we have required fields
                    if call_id and name:
                        function_calls.append(
                            {
                                "type": "function_call",
                                "call_id": call_id,
                                "name": name,
                                "arguments": arguments,
                            }
                        )
                elif output_item.get("type") == "message":
                    # Extract content from message
                    content = output_item.get("content", [])
                    if isinstance(content, list):
                        assistant_message_content.extend(content)
                    else:
                        assistant_message_content.append(content)

        # Add assistant message only if there's actual content (not empty)
        # For example, gemini requires that function call turns come immediately after user turns,
        # so we should not add empty assistant messages
        if assistant_message_content:
            follow_up_input.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": assistant_message_content,
                }
            )

        # Add function calls (these can come directly after user message for LLM)
        for function_call in function_calls:
            follow_up_input.append(function_call)

        # Add tool results (function call outputs)
        for tool_result in tool_results:
            follow_up_input.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_result["tool_call_id"],
                    "output": tool_result["result"],
                }
            )

        return follow_up_input

    @staticmethod
    async def _make_follow_up_call(
        follow_up_input: List[Any],
        model: str,
        all_tools: Optional[List[Any]],
        response_id: str,
        **call_params: Any,
    ) -> Union[ResponsesAPIResponse, BaseResponsesAPIStreamingIterator]:
        """Make follow-up response API call with tool results."""
        return await aresponses(
            input=follow_up_input,
            model=model,
            tools=all_tools,  # Keep tools for potential future calls
            previous_response_id=response_id,  # Link to previous response
            **call_params,
        )

    @staticmethod
    def _create_mcp_streaming_response(
        input: Union[str, Any],
        model: str,
        all_tools: Optional[List[Any]],
        mcp_tools_with_litellm_proxy: List[Any],
        mcp_discovery_events: List[Any],
        call_params: Dict[str, Any],
        previous_response_id: Optional[str],
        tool_server_map: dict[str, str],
        **kwargs,
    ) -> Any:
        """
        Create MCP enhanced streaming response that handles the full MCP workflow.

        This creates a streaming iterator that:
        1. Immediately emits MCP discovery events
        2. Makes the LLM call and streams the response
        3. Handles tool execution and follow-up calls
        """
        from litellm.responses.mcp.mcp_streaming_iterator import (
            MCPEnhancedStreamingIterator,
        )

        # Build the complete request parameters by merging all sources
        request_params = LiteLLM_Proxy_MCP_Handler._build_request_params(
            input=input,
            model=model,
            all_tools=all_tools,
            call_params=call_params,
            previous_response_id=previous_response_id,
            **kwargs,
        )

        # Create the enhanced streaming iterator that will handle everything
        return MCPEnhancedStreamingIterator(
            base_iterator=None,  # Will be created internally
            mcp_events=mcp_discovery_events,  # Pre-generated MCP discovery events
            tool_server_map=tool_server_map,
            mcp_tools_with_litellm_proxy=mcp_tools_with_litellm_proxy,
            user_api_key_auth=kwargs.get("user_api_key_auth"),
            original_request_params=request_params,
        )

    @staticmethod
    def _build_request_params(
        input: Union[str, Any],
        model: str,
        all_tools: Optional[List[Any]],
        call_params: Dict[str, Any],
        previous_response_id: Optional[str],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Build a clean request parameters dictionary for MCP streaming.

        Combines input, model, tools with call_params and additional kwargs
        in a clean, maintainable way.
        """
        # Start with the core required parameters
        request_params = {
            "input": input,
            "model": model,
            "tools": all_tools,
        }

        # Add previous_response_id if provided
        if previous_response_id is not None:
            request_params["previous_response_id"] = previous_response_id

        # Merge in all call_params (which contains most of the API parameters)
        request_params.update(call_params)

        # Merge in any additional kwargs
        request_params.update(kwargs)

        return request_params

    @staticmethod
    def _create_tool_execution_events(
        tool_calls: List[Any], tool_results: List[Dict[str, Any]]
    ) -> List[Any]:
        """
        Create MCP tool execution events for streaming.

        Args:
            tool_calls: List of tool calls from the LLM response
            tool_results: List of tool execution results

        Returns:
            List of MCP tool execution events for streaming
        """
        from litellm._uuid import uuid

        from litellm.responses.mcp.mcp_streaming_iterator import create_mcp_call_events

        tool_execution_events: List[Any] = []

        # Create events for each tool execution
        for tool_result in tool_results:
            tool_call_id = tool_result.get("tool_call_id", "unknown")
            result_text = tool_result.get("result", "")

            # Extract tool name and arguments from tool calls
            tool_name = "unknown"
            tool_arguments = "{}"
            for tool_call in tool_calls:
                (
                    name,
                    args,
                    call_id,
                ) = LiteLLM_Proxy_MCP_Handler._extract_tool_call_details(tool_call)
                if call_id == tool_call_id:
                    tool_name = name or "unknown"
                    tool_arguments = args or "{}"
                    break

            execution_events = create_mcp_call_events(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                arguments=tool_arguments,  # Use actual arguments
                result=result_text,
                base_item_id=f"mcp_{uuid.uuid4().hex[:8]}",  # Unique ID for each tool call
                sequence_start=len(tool_execution_events) + 1,
            )
            tool_execution_events.extend(execution_events)

        return tool_execution_events

    @staticmethod
    def _prepare_initial_call_params(
        call_params: Dict[str, Any], should_auto_execute: bool
    ) -> Dict[str, Any]:
        """
        Prepare call parameters for the initial LLM call.

        For auto-execute scenarios, we need to disable streaming for the initial call
        so we can process the tool calls before streaming the final response.
        """
        initial_params = call_params.copy()

        if should_auto_execute:
            # Disable streaming for initial call when auto-executing tools
            initial_params["stream"] = False

        return initial_params

    @staticmethod
    def _prepare_follow_up_call_params(
        call_params: Dict[str, Any], original_stream_setting: bool
    ) -> Dict[str, Any]:
        """
        Prepare call parameters for the follow-up LLM call after tool execution.

        Restores the original streaming setting and removes tool_choice since
        we're now providing tool results, not requesting tool calls.
        """
        follow_up_params = call_params.copy()

        # Restore original streaming setting for follow-up call
        follow_up_params["stream"] = original_stream_setting

        # Remove tool_choice since we're providing results, not requesting tool calls
        follow_up_params.pop("tool_choice", None)

        return follow_up_params

    @staticmethod
    def _add_mcp_output_elements_to_response(
        response: ResponsesAPIResponse,
        mcp_tools_fetched: List[Any],
        tool_results: List[Dict[str, Any]],
    ) -> ResponsesAPIResponse:
        """Add custom output elements to the final response for MCP tool execution."""
        # Import the required classes for creating output items
        import json
        from litellm._uuid import uuid

        from litellm.types.responses.main import GenericResponseOutputItem, OutputText

        # Create output element for initial MCP tools
        mcp_tools_output = GenericResponseOutputItem(
            type="mcp_tools_fetched",
            id=f"mcp_tools_{uuid.uuid4().hex[:8]}",
            status="completed",
            role="system",
            content=[
                OutputText(
                    type="output_text",
                    text=json.dumps(mcp_tools_fetched, indent=2, default=str),
                    annotations=[],
                )
            ],
        )

        # Create output element for tool execution results
        tool_results_output = GenericResponseOutputItem(
            type="tool_execution_results",
            id=f"tool_results_{uuid.uuid4().hex[:8]}",
            status="completed",
            role="system",
            content=[
                OutputText(
                    type="output_text",
                    text=json.dumps(tool_results, indent=2, default=str),
                    annotations=[],
                )
            ],
        )

        # Add the new output elements to the response
        response.output.append(mcp_tools_output.model_dump())  # type: ignore
        response.output.append(tool_results_output.model_dump())  # type: ignore

        return response
