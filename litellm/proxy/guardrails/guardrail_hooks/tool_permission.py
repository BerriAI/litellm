import re
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Union

from fastapi import HTTPException

from litellm import ChatCompletionToolParam
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
from litellm.types.proxy.guardrails.guardrail_hooks.tool_permission import (
    PermissionError,
    ToolPermissionRule,
    ToolResult,
)
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    LLMResponseTypes,
    ModelResponse,
    ModelResponseStream,
)

GUARDRAIL_NAME = "tool_permission"


class ToolPermissionGuardrail(CustomGuardrail):
    def __init__(
        self,
        rules: Optional[List[Dict]] = None,
        default_action: Literal["deny", "allow"] = "deny",
        on_disallowed_action: Literal["block", "rewrite"] = "block",
        **kwargs,
    ):
        """
        Initialize the Tool Permission Guardrail

        Args:
            rules: List of permission rules
            default_action: Default action when no rule matches ("allow" or "deny")
            on_disallowed_action:
            **kwargs: Additional arguments passed to CustomGuardrail
        """
        # Set supported event hooks - this guardrail only works on post_call
        if "supported_event_hooks" not in kwargs:
            kwargs["supported_event_hooks"] = [
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.post_call,
            ]

        super().__init__(**kwargs)

        self.rules: List[ToolPermissionRule] = []
        if rules:
            for rule_dict in rules:
                self.rules.append(ToolPermissionRule(**rule_dict))

        self.default_action = default_action
        self.on_disallowed_action = on_disallowed_action

        verbose_proxy_logger.debug(
            "Tool Permission Guardrail initialized with %d rules, default_action: %s",
            len(self.rules),
            self.default_action,
        )

    def _matches_pattern(self, tool_name: str, pattern: str) -> bool:
        """
        Check if a tool name matches a pattern

        Supports patterns like:
        - "Bash" - exact match
        - "mcp__*" - prefix pattern (matches names starting wich "mcp__")
        - "*_read" - suffix wildcard (matches names ending with "_read")
        - "mcp__github_*_read" - infix wildcard (matches names like "mcp__github_mark_all_notifications_read")

        Args:
            tool_name: Name of the tool to check
            pattern: Pattern to match against

        Returns:
            True if the tool name matches the pattern
        """
        # Handle exact matches
        if tool_name == pattern:
            return True

        if "*" in pattern:
            # Escape regex special chars except '*'
            escaped_pattern = re.escape(pattern)
            # Turn \* into .*
            regex_pattern = escaped_pattern.replace(r"\*", ".*")
            return bool(re.fullmatch(regex_pattern, tool_name))

        return False

    def _check_tool_permission(
        self, tool_name: str
    ) -> tuple[bool, Optional[str], Optional[str]]:
        """
        Check if a tool is allowed based on the configured rules

        Args:
            tool_name: Name of the tool to check

        Returns:
            Tuple of (is_allowed, rule_id, message)
        """
        verbose_proxy_logger.debug(f"Checking permission for tool: {tool_name}")

        # Check each rule in order
        for rule in self.rules:
            if self._matches_pattern(tool_name, rule.tool_name):
                is_allowed = rule.decision == "allow"
                message = f"Tool '{tool_name}' {'allowed' if is_allowed else 'denied'} by rule '{rule.id}'"
                verbose_proxy_logger.debug(message)
                return is_allowed, rule.id, message

        # No rule matched, use default action
        is_allowed = self.default_action == "allow"
        message = f"Tool '{tool_name}' {'allowed' if is_allowed else 'denied'} by default action"
        verbose_proxy_logger.debug(message)
        return is_allowed, None, message

    def _extract_tool_calls_from_response(
        self, response: ModelResponse
    ) -> List[ChatCompletionMessageToolCall]:
        """
        Extract tool_calls from all choices in a model response.

        Args:
            response: The model response to analyze

        Returns:
            List of tool_calls blocks found in the response
        """
        tool_calls = []

        for choice in response.choices:
            if isinstance(choice, Choices):
                for tool in choice.message.tool_calls or []:
                    tool_calls.append(tool)

        return tool_calls

    def _modify_request_with_permission_errors(
        self,
        data: dict,
        denied_tool_names: List[str],
    ):
        """
        Modify the request to replace denied tool_calls blocks with error results

        Args:
            data: The model request to modify
            denied_tools: List of (tool_use, error) tuples for denied tools
        """
        if not denied_tool_names:
            return data

        verbose_proxy_logger.info(
            f"Blocking {len(denied_tool_names)} unauthorized tool uses"
        )

        # Create a mapping of tool_use_id to error result
        error_tool_names = set()
        for tool_use in denied_tool_names:
            error_tool_names.add(tool_use)

        # Modify the tools
        tools: Optional[List[ChatCompletionToolParam]] = data.get("tools")
        if tools is None:
            return data

        new_tools = []
        for tool in tools:
            if tool["type"] != "function":
                continue
            tool_name: str = tool["function"]["name"]
            if tool_name not in error_tool_names:
                new_tools.append(tool)
        data["tools"] = new_tools
        return data

    def _create_permission_error_result(
        self, tool_call: ChatCompletionMessageToolCall, error: PermissionError
    ) -> ToolResult:
        """
        Create a tool_result block for a permission error

        Args:
            tool_use: The tool use that was denied
            error: The permission error details

        Returns:
            A tool_result block with the error message
        """
        error_message = f"Permission denied: {error.message}"
        if error.rule_id:
            error_message += f" (Rule: {error.rule_id})"

        return ToolResult(
            tool_use_id=tool_call.id, content=error_message, is_error=True
        )

    def _modify_response_with_permission_errors(
        self,
        response: ModelResponse,
        denied_tools: List[tuple[ChatCompletionMessageToolCall, PermissionError]],
    ) -> None:
        """
        Modify the response to replace denied tool_calls blocks with error results

        Args:
            response: The model response to modify
            denied_tools: List of (tool_use, error) tuples for denied tools
        """
        if not denied_tools:
            return

        verbose_proxy_logger.info(
            f"Blocking {len(denied_tools)} unauthorized tool uses"
        )

        # Create a mapping of tool_use_id to error result
        error_results = {}
        for tool_use, error in denied_tools:
            error_result = self._create_permission_error_result(tool_use, error)
            error_results[tool_use.id] = error_result

        # Modify the response content
        for choice in response.choices:
            if isinstance(choice, Choices):
                filtered_tool_calls = []
                error_messages = []

                # Rewrite tool_calls
                for tool_call in choice.message.tool_calls or []:
                    tool_call_id = tool_call.id
                    if tool_call_id in error_results:
                        error_result = error_results[tool_call_id]
                        error_messages.append(error_result.content)
                    else:
                        filtered_tool_calls.append(tool_call)

                choice.message.tool_calls = (
                    filtered_tool_calls if filtered_tool_calls else None
                )

                # Add error messages to content
                if error_messages:
                    existing_content = choice.message.content
                    if existing_content:
                        choice.message.content = (
                            existing_content + "\n\n" + "\n".join(error_messages)
                        )
                    else:
                        choice.message.content = "\n".join(error_messages)

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
            "mcp_call",
            "anthropic_messages",
        ],
    ) -> Union[Exception, str, dict, None]:
        """ """
        verbose_proxy_logger.debug("Tool Permission Guardrail Pre-Call Hook")

        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        event_type: GuardrailEventHooks = GuardrailEventHooks.pre_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return data

        new_tools: Optional[List[ChatCompletionToolParam]] = data.get("tools")
        if new_tools is None:
            verbose_proxy_logger.warning(
                "Tool Permission Guardrail: not running guardrail. No tools in data"
            )
            return data

        # Check permissions for each tool
        denied_tool_names = []
        for tool in new_tools:
            if tool["type"] != "function":
                continue
            tool_name: str = tool["function"]["name"]

            is_allowed, _, message = self._check_tool_permission(tool_name)

            if not is_allowed and message is not None:
                verbose_proxy_logger.warning(f"Tool Permission Guardrail: {message}")
                if self.on_disallowed_action == "block":
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "Violated guardrail policy",
                            "detection_message": message,
                        },
                    )
                denied_tool_names.append(tool_name)

        if denied_tool_names:
            data = self._modify_request_with_permission_errors(data, denied_tool_names)

        verbose_proxy_logger.debug(
            "Tool Permission Guardrail Pre-Call Hook: All tools allowed"
        )

        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )
        return data

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: LLMResponseTypes,
    ):
        """
        Check tool usage permissions after the LLM call

        Args:
            data: Request data
            user_api_key_dict: User API key information (unused but required by interface)
            response: The model response to check
        """
        if not isinstance(response, ModelResponse):
            return

        verbose_proxy_logger.debug(
            "Tool Permission Guardrail Post-Call Hook: Checking response"
        )

        if not self.should_run_guardrail(
            data=data, event_type=GuardrailEventHooks.post_call
        ):
            verbose_proxy_logger.debug(
                "Tool Permission Guardrail: Skipping check (not enabled)"
            )
            return

        # Extract tool_calls from the response
        tool_calls = self._extract_tool_calls_from_response(response)

        if not tool_calls:
            verbose_proxy_logger.debug("Tool Permission Guardrail: No tool uses found")
            return

        verbose_proxy_logger.debug(
            f"Tool Permission Guardrail: Found {len(tool_calls)} tool calls"
        )

        # Check permissions for each tool use
        denied_tools = []
        for tool_call in tool_calls:
            if tool_call.function.name is None:
                continue
            is_allowed, rule_id, message = self._check_tool_permission(
                tool_call.function.name
            )

            if not is_allowed and message is not None:
                verbose_proxy_logger.warning(f"Tool Permission Guardrail: {message}")

                if self.on_disallowed_action == "block":
                    raise GuardrailRaisedException(
                        guardrail_name=self.guardrail_name,
                        message=message,
                    )
                denied_tools.append(
                    (
                        tool_call,
                        PermissionError(
                            tool_name=tool_call.function.name,
                            rule_id=rule_id,
                            message=message,
                        ),
                    )
                )

        if denied_tools:
            self._modify_response_with_permission_errors(response, denied_tools)

        verbose_proxy_logger.debug(
            "Tool Permission Guardrail Post-Call Hook: All tools allowed"
        )

        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        """
        Check tool usage permissions after the LLM stream call

        Args:
            user_api_key_dict: User API key information (unused but required by interface)
            response: The model response to check
            request_data: The model request (unused but required by interface)
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
            verbose_proxy_logger.debug("Tool Permission Guardrail: Checking response")

            # Extract tool_calls from the response
            tool_calls = self._extract_tool_calls_from_response(assembled_model_response)

            if not tool_calls:
                verbose_proxy_logger.debug(
                    "Tool Permission Guardrail: No tool uses found"
                )
                return

            verbose_proxy_logger.debug(
                f"Tool Permission Guardrail: Found {len(tool_calls)} tool calls"
            )

            # Check permissions for each tool use
            denied_tools = []
            for tool_call in tool_calls:
                if tool_call.function.name is None:
                    continue
                is_allowed, rule_id, message = self._check_tool_permission(
                    tool_call.function.name
                )

                if not is_allowed and message is not None:
                    verbose_proxy_logger.warning(
                        f"Tool Permission Guardrail: {message}"
                    )

                    if self.on_disallowed_action == "block":
                        raise GuardrailRaisedException(
                            guardrail_name=self.guardrail_name,
                            message=message,
                        )
                    denied_tools.append(
                        (
                            tool_call,
                            PermissionError(
                                tool_name=tool_call.function.name,
                                rule_id=rule_id,
                                message=message,
                            ),
                        )
                    )

                verbose_proxy_logger.debug(
                    "Tool Permission Guardrail Post-Call Hook: All tools allowed"
                )

                if denied_tools:
                    self._modify_response_with_permission_errors(
                        assembled_model_response, denied_tools
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
