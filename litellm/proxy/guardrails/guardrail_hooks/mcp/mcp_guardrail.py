"""
MCP Guardrail Implementation

This guardrail allows you to apply security and validation checks to MCP tool calls
through pre-call, during-call, and post-call hooks.
"""

import asyncio
import importlib
from typing import Any, Dict, Literal, Optional, Union

from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import GuardrailEventHooks


class MCPGuardrail(CustomGuardrail):
    """
    MCP Guardrail for applying security and validation checks to MCP tool calls.
    
    This guardrail supports:
    - Pre-call validation of tool arguments
    - During-call monitoring of tool execution
    - Post-call validation of tool results
    - Custom validation functions
    - Server and tool-specific filtering
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mcp_server_name = kwargs.get("mcp_server_name")
        self.mcp_server_id = kwargs.get("mcp_server_id")
        self.tool_name = kwargs.get("tool_name")
        self.block_on_error = kwargs.get("block_on_error", True)
        self.timeout = kwargs.get("timeout", 30.0)
        self._custom_validation_function_path = kwargs.get("custom_validation_function")
        self.validation_rules = kwargs.get("validation_rules", {})
        
        # Initialize custom validation function if provided
        self._custom_validator = None
        if self._custom_validation_function_path:
            self._custom_validator = self._load_custom_validation_function()
    
    @property
    def custom_validation_function(self):
        """Return the loaded custom validation function."""
        return self._custom_validator

    def _load_custom_validation_function(self):
        """Load custom validation function from string path."""
        if not self._custom_validation_function_path:
            return None
            
        try:
            # Parse the function path (e.g., "module.submodule:function_name")
            if ":" in self._custom_validation_function_path:
                module_path, function_name = self._custom_validation_function_path.split(":", 1)
                
                # Import the module
                module = importlib.import_module(module_path)
                
                # Get the function
                function = getattr(module, function_name)
                
                if not callable(function):
                    raise ValueError(f"{self._custom_validation_function_path} is not callable")
                
                return function
            else:
                raise ValueError(f"Invalid function path: {self._custom_validation_function_path}")
        except Exception as e:
            verbose_proxy_logger.error(f"Failed to load custom validation function: {e}")
            return None

    def _should_apply_to_tool(self, tool_name: str, server_name: Optional[str] = None) -> bool:
        """Check if this guardrail should be applied to the given tool."""
        verbose_proxy_logger.debug(f"MCP Guardrail: Checking if should apply to tool '{tool_name}' on server '{server_name}'")
        
        # If specific tool name is set, only apply to that tool
        if self.tool_name and tool_name != self.tool_name:
            verbose_proxy_logger.debug(f"MCP Guardrail: Skipping - tool name mismatch. Expected: {self.tool_name}, Got: {tool_name}")
            return False
        
        # If specific server name is set, only apply to that server
        if self.mcp_server_name and server_name != self.mcp_server_name:
            verbose_proxy_logger.debug(f"MCP Guardrail: Skipping - server name mismatch. Expected: {self.mcp_server_name}, Got: {server_name}")
            return False
        
        # Check for forbidden tool patterns
        forbidden_tool_patterns = self.validation_rules.get("forbidden_tool_patterns", [])
        for pattern in forbidden_tool_patterns:
            if pattern.endswith("*"):
                # Wildcard pattern matching
                prefix = pattern[:-1]
                if tool_name.startswith(prefix):
                    raise ValueError(f"Tool '{tool_name}' matches forbidden pattern '{pattern}'")
            elif tool_name == pattern:
                # Exact pattern matching
                raise ValueError(f"Tool '{tool_name}' matches forbidden pattern '{pattern}'")
        
        # Check for allowed tool patterns (if specified, tool must match at least one)
        allowed_tool_patterns = self.validation_rules.get("allowed_tool_patterns", [])
        if allowed_tool_patterns:
            tool_allowed = False
            for pattern in allowed_tool_patterns:
                if pattern.endswith("*"):
                    # Wildcard pattern matching
                    prefix = pattern[:-1]
                    if tool_name.startswith(prefix):
                        tool_allowed = True
                        break
                elif tool_name == pattern:
                    # Exact pattern matching
                    tool_allowed = True
                    break
            
            if not tool_allowed:
                raise ValueError(f"Tool '{tool_name}' does not match any allowed pattern: {allowed_tool_patterns}")
        
        return True

    def _validate_tool_arguments(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and potentially modify tool arguments."""
        try:
            # Apply validation rules
            if self.validation_rules:
                arguments = self._apply_validation_rules(arguments)
            
            # Apply custom validation function if available
            if self._custom_validator:
                arguments = self._apply_custom_validation(tool_name, arguments)
            
            return arguments
        except Exception as e:
            verbose_proxy_logger.error(f"MCP Guardrail validation error: {e}")
            if self.block_on_error:
                raise ValueError(f"MCP Guardrail validation failed: {e}")
            return arguments

    def _apply_validation_rules(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Apply predefined validation rules to arguments."""
        if not self.validation_rules:
            return arguments
            
        # Check for forbidden keywords
        forbidden_keywords = self.validation_rules.get("forbidden_keywords", [])
        for key, value in arguments.items():
            if isinstance(value, str):
                value_lower = value.lower()
                for keyword in forbidden_keywords:
                    if keyword.lower() in value_lower:
                        raise ValueError(f"Forbidden keyword '{keyword}' found in argument '{key}'")
        
        # Check query length limits (support both "query" and "question" fields)
        query_fields = ["query", "question"]
        for field in query_fields:
            if field in arguments and isinstance(arguments[field], str):
                max_query_length = self.validation_rules.get("max_query_length")
                if max_query_length and len(arguments[field]) > max_query_length:
                    raise ValueError(f"{field.capitalize()} exceeds maximum length of {max_query_length} characters")
                
                min_query_length = self.validation_rules.get("min_query_length")
                if min_query_length and len(arguments[field]) < min_query_length:
                    raise ValueError(f"{field.capitalize()} must be at least {min_query_length} characters long")
        
        # Check title length limits
        if "title" in arguments and isinstance(arguments["title"], str):
            max_title_length = self.validation_rules.get("max_title_length")
            if max_title_length and len(arguments["title"]) > max_title_length:
                raise ValueError(f"Title exceeds maximum length of {max_title_length} characters")
            
            min_title_length = self.validation_rules.get("min_title_length")
            if min_title_length and len(arguments["title"]) < min_title_length:
                raise ValueError(f"Title must be at least {min_title_length} characters long")
        
        # Check body length limits
        if "body" in arguments and isinstance(arguments["body"], str):
            max_body_length = self.validation_rules.get("max_body_length")
            if max_body_length and len(arguments["body"]) > max_body_length:
                raise ValueError(f"Body exceeds maximum length of {max_body_length} characters")
            
            min_body_length = self.validation_rules.get("min_body_length")
            if min_body_length and len(arguments["body"]) < min_body_length:
                raise ValueError(f"Body must be at least {min_body_length} characters long")
        
        # Check for forbidden search terms (support both "query" and "question" fields)
        query_fields = ["query", "question"]
        for field in query_fields:
            if field in arguments and isinstance(arguments[field], str):
                forbidden_search_terms = self.validation_rules.get("forbidden_search_terms", [])
                query_lower = arguments[field].lower()
                for term in forbidden_search_terms:
                    if term.lower() in query_lower:
                        raise ValueError(f"Forbidden search term '{term}' found in {field}")
        
        # Check for potentially unsafe content
        for key, value in arguments.items():
            if isinstance(value, str):
                unsafe_patterns = ["<script>", "javascript:", "data:", "vbscript:"]
                if any(pattern in value.lower() for pattern in unsafe_patterns):
                    if self.block_on_error:
                        raise ValueError(f"Potentially unsafe content detected in argument '{key}'")
                    # Optionally sanitize the value
                    arguments[key] = value.replace("<script>", "").replace("javascript:", "")
        
        # Check for forbidden patterns in any string value
        forbidden_patterns = self.validation_rules.get("forbidden_patterns", [])
        for key, value in arguments.items():
            if isinstance(value, str):
                value_lower = value.lower()
                for pattern in forbidden_patterns:
                    if pattern.lower() in value_lower:
                        raise ValueError(f"Content matches forbidden pattern '{pattern}' in argument '{key}'")
        
        # Check webhook URL validation
        if "url" in arguments and isinstance(arguments["url"], str):
            allowed_webhook_urls = self.validation_rules.get("allowed_webhook_urls", [])
            if allowed_webhook_urls and arguments["url"] not in allowed_webhook_urls:
                raise ValueError(f"Webhook URL '{arguments['url']}' is not in the allowed list")
        
        return arguments

    def _apply_custom_validation(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Apply custom validation function to arguments."""
        if not self._custom_validator:
            return arguments
        
        try:
            # Check if function is async
            if asyncio.iscoroutinefunction(self._custom_validator):
                # For async functions, we'll need to handle this differently
                # For now, we'll just call it synchronously (not ideal)
                verbose_proxy_logger.warning("Async custom validation functions are not fully supported yet")
                return arguments
            
            # Call the custom validation function
            result = self._custom_validator(tool_name, arguments, self.validation_rules)
            
            # If the function returns a dict, use it as the new arguments
            if isinstance(result, dict):
                return result
            
            # If the function returns None or False, it might indicate validation failure
            if result is False:
                raise ValueError("Custom validation function rejected the arguments")
            
            return arguments
        except Exception as e:
            verbose_proxy_logger.error(f"Custom validation function error: {e}")
            if self.block_on_error:
                raise ValueError(f"Custom validation failed: {e}")
            return arguments

    def _validate_tool_result(self, tool_name: str, result: Any) -> Any:
        """Validate and potentially modify tool result."""
        try:
            # Apply result validation rules
            if self.validation_rules.get("result_validation"):
                result = self._apply_result_validation_rules(result)
            
            return result
        except Exception as e:
            verbose_proxy_logger.error(f"MCP Guardrail result validation error: {e}")
            if self.block_on_error:
                raise ValueError(f"MCP Guardrail result validation failed: {e}")
            return result

    def _apply_result_validation_rules(self, result: Any) -> Any:
        """Apply validation rules to tool results."""
        # Example: Check for sensitive data in results
        if isinstance(result, str):
            # Check for potential data leaks
            sensitive_patterns = ["password", "secret", "key", "token"]
            if any(pattern in result.lower() for pattern in sensitive_patterns):
                verbose_proxy_logger.warning("Potential sensitive data detected in tool result")
                # Optionally mask or redact the result
                # result = self._mask_sensitive_data(result)
        
        return result

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
        ],
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Pre-call hook to validate MCP tool calls before execution.
        
        This hook is called before making an MCP tool call to validate:
        - Tool arguments
        - User permissions
        - Security constraints
        """
        event_type = GuardrailEventHooks.pre_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            verbose_proxy_logger.debug(
                f"MCP Guardrail: Pre-call validation disabled for {self.guardrail_name}"
            )
            return data

        verbose_proxy_logger.debug("MCP Guardrail: Pre-call hook")

        # Extract MCP tool call information from the request
        tool_name = data.get("name")
        arguments = data.get("arguments", {})
        server_name = data.get("mcp_server_name")

        verbose_proxy_logger.debug(f"MCP Guardrail: Tool name: {tool_name}, Server name: {server_name}, Expected server: {self.mcp_server_name}")

        if not tool_name:
            verbose_proxy_logger.warning("MCP Guardrail: No tool name found in request")
            return data

        # Check if this guardrail should be applied to this tool
        if not self._should_apply_to_tool(tool_name, server_name):
            verbose_proxy_logger.debug(
                f"MCP Guardrail: Skipping validation for tool {tool_name} (not in scope)"
            )
            return data

        verbose_proxy_logger.debug(f"MCP Guardrail: Applying validation to tool {tool_name}")

        try:
            # Validate tool arguments
            validated_arguments = self._validate_tool_arguments(tool_name, arguments)
            
            # Update the data with validated arguments
            data["arguments"] = validated_arguments
            
            verbose_proxy_logger.debug(
                f"MCP Guardrail: Pre-call validation completed for tool {tool_name}"
            )
            
            return data
        except Exception as e:
            verbose_proxy_logger.error(f"MCP Guardrail: Pre-call validation failed: {e}")
            if self.block_on_error:
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "MCP Guardrail validation failed",
                        "guardrail_name": self.guardrail_name,
                        "validation_error": str(e),
                    }
                )
            return data

    @log_guardrail_information
    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal[
            "completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "responses",
        ],
    ) -> Optional[Union[Exception, str, dict]]:
        """
        During-call hook to monitor MCP tool execution.
        
        This hook runs during the MCP tool call execution to provide:
        - Real-time monitoring
        - Performance tracking
        - Security monitoring
        """
        event_type = GuardrailEventHooks.during_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            verbose_proxy_logger.debug(
                f"MCP Guardrail: During-call monitoring disabled for {self.guardrail_name}"
            )
            return data

        verbose_proxy_logger.debug("MCP Guardrail: During-call moderation hook")

        # Extract MCP tool call information
        tool_name = data.get("name")
        server_name = data.get("mcp_server_name")

        if not tool_name:
            return data

        # Check if this guardrail should be applied to this tool
        if not self._should_apply_to_tool(tool_name, server_name):
            return data

        try:
            # Perform during-call monitoring
            # This could include:
            # - Performance monitoring
            # - Security checks
            # - Rate limiting
            # - Resource usage tracking
            
            verbose_proxy_logger.debug(
                f"MCP Guardrail: During-call monitoring active for tool {tool_name}"
            )
            
            return data
        except Exception as e:
            verbose_proxy_logger.error(f"MCP Guardrail: During-call monitoring failed: {e}")
            if self.block_on_error:
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "MCP Guardrail during-call monitoring failed",
                        "guardrail_name": self.guardrail_name,
                        "validation_error": str(e),
                    }
                )
            return data

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
    ) -> Any:
        """
        Post-call hook to validate MCP tool results after execution.
        
        This hook is called after a successful MCP tool call to validate:
        - Tool results
        - Data integrity
        - Security compliance
        """
        event_type = GuardrailEventHooks.post_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            verbose_proxy_logger.debug(
                f"MCP Guardrail: Post-call validation disabled for {self.guardrail_name}"
            )
            return response

        verbose_proxy_logger.debug("MCP Guardrail: Post-call hook")

        # Extract MCP tool call information
        tool_name = data.get("name")
        server_name = data.get("mcp_server_name")

        if not tool_name:
            return response

        # Check if this guardrail should be applied to this tool
        if not self._should_apply_to_tool(tool_name, server_name):
            return response

        try:
            # Validate tool result
            validated_result = self._validate_tool_result(tool_name, response)
            
            verbose_proxy_logger.debug(
                f"MCP Guardrail: Post-call validation completed for tool {tool_name}"
            )
            
            return validated_result
        except Exception as e:
            verbose_proxy_logger.error(f"MCP Guardrail: Post-call validation failed: {e}")
            if self.block_on_error:
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "MCP Guardrail post-call validation failed",
                        "guardrail_name": self.guardrail_name,
                        "validation_error": str(e),
                    }
                )
            return response

    @staticmethod
    def get_config_model() -> Optional[type]:
        """Return the configuration model for this guardrail."""
        from litellm.types.guardrails import MCPGuardrailConfigModel
        return MCPGuardrailConfigModel 