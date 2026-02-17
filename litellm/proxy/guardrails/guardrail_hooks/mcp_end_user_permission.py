"""
MCP End User Permission Guardrail Hook

This hook enforces end user permissions for MCP server access by:
1. Checking tool names in OpenAI and Anthropic responses
2. Extracting the MCP server name from tool names (split on first '-')
3. Verifying that the end user has permissions to access the MCP server

The hook runs on post_call to validate that the LLM response only includes
tools from MCP servers that the end user is authorized to access.
"""
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.callback_utils import (
    add_guardrail_to_applied_guardrails_header,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import (
    CallTypesLiteral,
    ChatCompletionMessageToolCall,
    Choices,
    LLMResponseTypes,
    ModelResponse,
    ModelResponseStream,
)

GUARDRAIL_NAME = "mcp_end_user_permission"


class MCPEndUserPermissionGuardrail(CustomGuardrail):
    """
    Guardrail that enforces end user permissions for MCP server access.

    This runs on post_call to validate that tool calls in the LLM response
    only reference MCP servers that the end user is authorized to access.
    """

    def __init__(
        self,
        **kwargs,
    ):
        """
        Initialize the MCP End User Permission Guardrail

        Args:
            **kwargs: Additional arguments passed to CustomGuardrail
        """
        # Set supported event hooks - this guardrail only works on post_call
        if "supported_event_hooks" not in kwargs:
            kwargs["supported_event_hooks"] = [
                GuardrailEventHooks.post_call,
            ]

        super().__init__(**kwargs)

        verbose_proxy_logger.debug(
            "MCP End User Permission Guardrail initialized"
        )

    @staticmethod
    def _extract_mcp_server_name_from_tool(tool_name: str) -> Optional[str]:
        """
        Extract MCP server name from tool name by splitting on first '-'.

        Args:
            tool_name: The full tool name (e.g., "github-create_issue")

        Returns:
            The server name (e.g., "github") or None if no separator found
        """
        if not tool_name or "-" not in tool_name:
            return None

        # Split on first '-' and return the first part (server name)
        parts = tool_name.split("-", 1)
        return parts[0] if parts else None

    @staticmethod
    async def _check_end_user_has_mcp_permission(
        server_name: str,
        user_api_key_auth: UserAPIKeyAuth,
    ) -> bool:
        """
        Check if the end user has permission to access the MCP server.

        Args:
            server_name: The MCP server name
            user_api_key_auth: User authentication object

        Returns:
            True if the user has permission, False otherwise
        """
        from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
            MCPRequestHandler,
        )

        # If no end_user_id, allow access (permission is at key/team level)
        if not user_api_key_auth.end_user_id:
            verbose_proxy_logger.debug(
                f"No end_user_id present for MCP server '{server_name}' - allowing access"
            )
            return True

        verbose_proxy_logger.debug(
            f"Checking end user permissions for MCP server '{server_name}' and end_user_id '{user_api_key_auth.end_user_id}'"
        )

        # Get allowed MCP servers for the end user
        allowed_mcp_servers = await MCPRequestHandler._get_allowed_mcp_servers_for_end_user(
            user_api_key_auth=user_api_key_auth
        )

        verbose_proxy_logger.debug(
            f"End user '{user_api_key_auth.end_user_id}' has access to MCP servers: {allowed_mcp_servers}"
        )

        # If end user has no specific permissions, deny access
        if not allowed_mcp_servers:
            verbose_proxy_logger.warning(
                f"End user '{user_api_key_auth.end_user_id}' has no MCP permissions - denying access to '{server_name}'"
            )
            return False

        # Check if the server is in the allowed list
        # Note: We need to match against server IDs
        return server_name in allowed_mcp_servers

    def _extract_tool_calls_from_response(
        self, response: ModelResponse
    ) -> List[ChatCompletionMessageToolCall]:
        """
        Extract tool_calls from all choices in a model response.

        Args:
            response: The model response to analyze

        Returns:
            List of tool_calls found in the response
        """
        tool_calls = []

        for choice in response.choices:
            if isinstance(choice, Choices):
                for tool in choice.message.tool_calls or []:
                    tool_calls.append(tool)

        return tool_calls

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: LLMResponseTypes,
    ):
        """
        Check MCP end user permissions after the LLM call.

        This hook validates that all tool calls in the response reference
        MCP servers that the end user is authorized to access.

        Args:
            data: Request data
            user_api_key_dict: User API key information
            response: The model response to check
        """
        if not isinstance(response, ModelResponse):
            return response

        verbose_proxy_logger.debug(
            "MCP End User Permission Guardrail Post-Call Hook: Checking response"
        )

        if not self.should_run_guardrail(
            data=data, event_type=GuardrailEventHooks.post_call
        ):
            verbose_proxy_logger.debug(
                "MCP End User Permission Guardrail: Skipping check (not enabled)"
            )
            return response

        # Extract tool_calls from the response
        tool_calls = self._extract_tool_calls_from_response(response)

        if not tool_calls:
            verbose_proxy_logger.debug("MCP End User Permission Guardrail: No tool calls found")
            return response

        verbose_proxy_logger.debug(
            f"MCP End User Permission Guardrail: Found {len(tool_calls)} tool calls"
        )

        # Check permissions for each tool call
        unauthorized_tools = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name if tool_call.function else None
            if not tool_name:
                continue

            # Extract MCP server name from tool name
            server_name = self._extract_mcp_server_name_from_tool(tool_name)
            if not server_name:
                # Not an MCP tool (no prefix), skip
                verbose_proxy_logger.debug(
                    f"Tool '{tool_name}' does not appear to be an MCP tool (no prefix) - skipping permission check"
                )
                continue

            verbose_proxy_logger.debug(
                f"Checking MCP end user permissions for tool '{tool_name}' (server: '{server_name}')"
            )

            # Check if end user has permission for this MCP server
            has_permission = await self._check_end_user_has_mcp_permission(
                server_name=server_name,
                user_api_key_auth=user_api_key_dict,
            )

            if not has_permission:
                unauthorized_tools.append((tool_name, server_name))
                verbose_proxy_logger.warning(
                    f"End user '{user_api_key_dict.end_user_id}' does not have permission "
                    f"to access MCP server '{server_name}' (tool: '{tool_name}')"
                )

        # If there are unauthorized tools, raise an exception
        if unauthorized_tools:
            tool_list = ", ".join([f"'{t[0]}' (server: '{t[1]}')" for t in unauthorized_tools])
            error_message = (
                f"End user '{user_api_key_dict.end_user_id}' does not have permission "
                f"to access the following MCP servers: {tool_list}"
            )

            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=error_message,
            )

        verbose_proxy_logger.debug(
            "MCP End User Permission Guardrail Post-Call Hook: All tool calls authorized"
        )

        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )
        return response

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        """
        Check MCP end user permissions for streaming responses.

        Args:
            user_api_key_dict: User API key information
            response: The model response stream to check
            request_data: The model request
        """
        # Import here to avoid circular imports
        from litellm.llms.base_llm.base_model_iterator import MockResponseIterator
        from litellm.main import stream_chunk_builder
        from litellm.types.utils import TextCompletionResponse

        # Collect all chunks to process them together
        all_chunks: List[ModelResponseStream] = []
        async for chunk in response:
            all_chunks.append(chunk)

        assembled_model_response: Optional[
            Union[ModelResponse, TextCompletionResponse]
        ] = stream_chunk_builder(
            chunks=all_chunks,
        )

        if isinstance(assembled_model_response, ModelResponse):
            verbose_proxy_logger.debug("MCP End User Permission Guardrail: Checking streaming response")

            # Extract tool_calls from the response
            tool_calls = self._extract_tool_calls_from_response(
                assembled_model_response
            )

            if not tool_calls:
                verbose_proxy_logger.debug(
                    "MCP End User Permission Guardrail: No tool calls found in stream"
                )
            else:
                verbose_proxy_logger.debug(
                    f"MCP End User Permission Guardrail: Found {len(tool_calls)} tool calls in stream"
                )

                # Check permissions for each tool call
                unauthorized_tools = []
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name if tool_call.function else None
                    if not tool_name:
                        continue

                    # Extract MCP server name from tool name
                    server_name = self._extract_mcp_server_name_from_tool(tool_name)
                    if not server_name:
                        # Not an MCP tool (no prefix), skip
                        continue

                    # Check if end user has permission for this MCP server
                    has_permission = await self._check_end_user_has_mcp_permission(
                        server_name=server_name,
                        user_api_key_auth=user_api_key_dict,
                    )

                    if not has_permission:
                        unauthorized_tools.append((tool_name, server_name))
                        verbose_proxy_logger.warning(
                            f"End user '{user_api_key_dict.end_user_id}' does not have permission "
                            f"to access MCP server '{server_name}' (tool: '{tool_name}')"
                        )

                # If there are unauthorized tools, raise an exception
                if unauthorized_tools:
                    tool_list = ", ".join([f"'{t[0]}' (server: '{t[1]}')" for t in unauthorized_tools])
                    error_message = (
                        f"End user '{user_api_key_dict.end_user_id}' does not have permission "
                        f"to access the following MCP servers: {tool_list}"
                    )

                    raise GuardrailRaisedException(
                        guardrail_name=self.guardrail_name,
                        message=error_message,
                    )

            mock_response = MockResponseIterator(
                model_response=assembled_model_response
            )
            # Return the reconstructed stream
            async for chunk in mock_response:
                yield chunk
        else:
            for chunk in all_chunks:
                yield chunk
