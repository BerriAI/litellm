from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from litellm._logging import verbose_logger
from litellm.responses.main import aresponses
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.types.llms.openai import ResponsesAPIResponse, ToolParam


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
                if (isinstance(tool, dict) and 
                    tool.get("type") == "mcp" and 
                    tool.get("server_url") == "litellm_proxy"):
                    return True
        return False
    
    @staticmethod
    def _parse_mcp_tools(tools: Optional[Iterable[ToolParam]]) -> Tuple[List[ToolParam], List[Any]]:
        """
        Parse tools and separate MCP tools with litellm_proxy from other tools.
        
        Returns:
            Tuple of (mcp_tools_with_litellm_proxy, other_tools)
        """
        mcp_tools_with_litellm_proxy: List[ToolParam] = []
        other_tools: List[Any] = []
        
        if tools:
            for tool in tools:
                if (isinstance(tool, dict) and 
                    tool.get("type") == "mcp" and 
                    tool.get("server_url") == "litellm_proxy"):
                    mcp_tools_with_litellm_proxy.append(tool)
                else:
                    other_tools.append(tool)
        
        return mcp_tools_with_litellm_proxy, other_tools
    
    @staticmethod
    async def _get_mcp_tools_from_manager(user_api_key_auth: Any) -> List[Any]:
        """Get available tools from the MCP server manager."""
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        
        return await global_mcp_server_manager.list_tools(user_api_key_auth=user_api_key_auth)
    
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
            if (isinstance(output_item, dict) and 
                output_item.get("type") == "function_call"):
                tool_calls.append(output_item)
            elif hasattr(output_item, 'type') and getattr(output_item, 'type') == "function_call":
                # Handle pydantic model case
                tool_calls.append(output_item)
        
        return tool_calls
    
    @staticmethod
    def _extract_tool_call_details(tool_call) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract tool name, arguments, and call_id from a tool call."""
        if isinstance(tool_call, dict):
            tool_name = tool_call.get("name")
            tool_arguments = tool_call.get("arguments")
            tool_call_id = tool_call.get("call_id") or tool_call.get("id")
        else:
            tool_name = getattr(tool_call, "name", None)
            tool_arguments = getattr(tool_call, "arguments", None)
            tool_call_id = getattr(tool_call, "call_id", None) or getattr(tool_call, "id", None)
        
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
        if not result or not hasattr(result, 'content') or not result.content:
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
        tool_calls: List[Any], 
        user_api_key_auth: Any
    ) -> List[Dict[str, Any]]:
        """Execute tool calls and return results."""
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        
        tool_results = []
        tool_call_id: Optional[str] = None
        for tool_call in tool_calls:
            try:
                tool_name, tool_arguments, tool_call_id = LiteLLM_Proxy_MCP_Handler._extract_tool_call_details(tool_call)
                
                if not tool_name:
                    verbose_logger.warning(f"Tool call missing name: {tool_call}")
                    continue
                
                parsed_arguments = LiteLLM_Proxy_MCP_Handler._parse_tool_arguments(tool_arguments)
                
                result = await global_mcp_server_manager.call_tool(
                    name=tool_name,
                    arguments=parsed_arguments,
                    user_api_key_auth=user_api_key_auth,
                )
                
                # Format result for inclusion in response
                result_text = LiteLLM_Proxy_MCP_Handler._parse_mcp_result(result)
                tool_results.append({
                    "tool_call_id": tool_call_id,
                    "result": result_text
                })
                
            except Exception as e:
                verbose_logger.exception(f"Error executing MCP tool call: {e}")
                tool_results.append({
                    "tool_call_id": tool_call_id,
                    "result": f"Error executing tool: {str(e)}"
                })
        
        return tool_results
    
    @staticmethod
    def _create_follow_up_input(
        response: ResponsesAPIResponse, 
        tool_results: List[Dict[str, Any]], 
        original_input: Any = None
    ) -> List[Any]:
        """Create follow-up input with tool results in proper format."""
        follow_up_input: List[Any] = []
        
        # Add original user input if available to maintain conversation context
        if original_input:
            if isinstance(original_input, str):
                follow_up_input.append({
                    "type": "message",
                    "role": "user",
                    "content": original_input
                })
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
                        function_calls.append({
                            "type": "function_call",
                            "call_id": call_id,
                            "name": name,
                            "arguments": arguments
                        })
                elif output_item.get("type") == "message":
                    # Extract content from message
                    content = output_item.get("content", [])
                    if isinstance(content, list):
                        assistant_message_content.extend(content)
                    else:
                        assistant_message_content.append(content)
        
        # Add assistant message with content and function calls
        if assistant_message_content or function_calls:
            follow_up_input.append({
                "type": "message",
                "role": "assistant",
                "content": assistant_message_content
            })
            
            # Add function calls after assistant message
            for function_call in function_calls:
                follow_up_input.append(function_call)
        
        # Add tool results (function call outputs)
        for tool_result in tool_results:
            follow_up_input.append({
                "type": "function_call_output",
                "call_id": tool_result["tool_call_id"],
                "output": tool_result["result"]
            })
        
        return follow_up_input
    
    @staticmethod
    async def _make_follow_up_call(
        follow_up_input: List[Any],
        model: str,
        all_tools: Optional[List[Any]],
        response_id: str,
        **call_params: Any
    ) -> Union[ResponsesAPIResponse, BaseResponsesAPIStreamingIterator]:
        """Make follow-up response API call with tool results."""
        return await aresponses(
            input=follow_up_input,
            model=model,
            tools=all_tools,  # Keep tools for potential future calls
            previous_response_id=response_id,  # Link to previous response
            **call_params
        )

    @staticmethod
    def _add_mcp_output_elements_to_response(
        response: ResponsesAPIResponse,
        mcp_tools_fetched: List[Any],
        tool_results: List[Dict[str, Any]]
    ) -> ResponsesAPIResponse:
        """Add custom output elements to the final response for MCP tool execution."""
        # Import the required classes for creating output items
        import json
        import uuid

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
                    annotations=[]
                )
            ]
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
                    annotations=[]
                )
            ]
        )
        
        # Add the new output elements to the response
        response.output.append(mcp_tools_output.model_dump())  # type: ignore
        response.output.append(tool_results_output.model_dump())  # type: ignore
        
        return response
