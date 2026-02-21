"""
Custom code guardrail for LiteLLM.

This module provides a guardrail that executes user-defined Python-like code
to implement custom guardrail logic. The code runs in a sandboxed environment
with access to LiteLLM-provided primitives for common guardrail operations.

Example custom code (sync):

    def apply_guardrail(inputs, request_data, input_type):
        '''Block messages containing SSNs'''
        for text in inputs["texts"]:
            if regex_match(text, r"\\d{3}-\\d{2}-\\d{4}"):
                return block("Social Security Number detected")
        return allow()

Example custom code (async with HTTP):

    async def apply_guardrail(inputs, request_data, input_type):
        '''Call external moderation API'''
        for text in inputs["texts"]:
            response = await http_post(
                "https://api.example.com/moderate",
                body={"text": text}
            )
            if response["success"] and response["body"].get("flagged"):
                return block("Content flagged by moderation API")
        return allow()
"""

import asyncio
import threading
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Type, cast

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel
from litellm.types.utils import GenericGuardrailAPIInputs

from .primitives import get_custom_code_primitives

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class CustomCodeGuardrailError(Exception):
    """Raised when custom code guardrail execution fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.details = details or {}


class CustomCodeCompilationError(CustomCodeGuardrailError):
    """Raised when custom code fails to compile."""


class CustomCodeExecutionError(CustomCodeGuardrailError):
    """Raised when custom code fails during execution."""


class CustomCodeGuardrailConfigModel(GuardrailConfigModel):
    """Configuration parameters for the custom code guardrail."""

    custom_code: str
    """The Python-like code containing the apply_guardrail function."""


class CustomCodeGuardrail(CustomGuardrail):
    """
    Guardrail that executes user-defined Python-like code.

    The code runs in a sandboxed environment that provides:
    - Access to LiteLLM primitives (regex_match, json_parse, etc.)
    - No file I/O or network access
    - No imports allowed

    Users write an `apply_guardrail(inputs, request_data, input_type)` function
    that returns one of:
    - allow() - let the request/response through
    - block(reason) - reject with a message
    - modify(texts=...) - transform the content

    Example:
        def apply_guardrail(inputs, request_data, input_type):
            for text in inputs["texts"]:
                if regex_match(text, r"password"):
                    return block("Sensitive content detected")
            return allow()
    """

    def __init__(
        self,
        custom_code: str,
        guardrail_name: Optional[str] = "custom_code",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the custom code guardrail.

        Args:
            custom_code: The source code containing apply_guardrail function
            guardrail_name: Name of this guardrail instance
            **kwargs: Additional arguments passed to CustomGuardrail
        """
        self.custom_code = custom_code
        self._compiled_function: Optional[Any] = None
        self._compile_lock = threading.Lock()
        self._compile_error: Optional[str] = None

        supported_event_hooks = [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.during_call,
            GuardrailEventHooks.post_call,
            GuardrailEventHooks.pre_mcp_call,
            GuardrailEventHooks.during_mcp_call,
            GuardrailEventHooks.logging_only,
        ]

        super().__init__(
            guardrail_name=guardrail_name,
            supported_event_hooks=supported_event_hooks,
            **kwargs,
        )

        # Compile the code on initialization
        self._compile_custom_code()

    @staticmethod
    def get_config_model() -> Optional[Type[GuardrailConfigModel]]:
        """Returns the config model for the UI."""
        return CustomCodeGuardrailConfigModel

    def _compile_custom_code(self) -> None:
        """
        Compile the custom code and extract the apply_guardrail function.

        The code runs in a sandboxed environment with only the allowed primitives.
        """
        with self._compile_lock:
            if self._compiled_function is not None:
                return

            try:
                # Create a restricted execution environment
                # Only include our safe primitives
                exec_globals = get_custom_code_primitives().copy()

                # Execute the user code in the restricted environment
                exec(compile(self.custom_code, "<guardrail>", "exec"), exec_globals)

                # Extract the apply_guardrail function
                if "apply_guardrail" not in exec_globals:
                    raise CustomCodeCompilationError(
                        "Custom code must define an 'apply_guardrail' function. "
                        "Expected signature: apply_guardrail(inputs, request_data, input_type)"
                    )

                apply_fn = exec_globals["apply_guardrail"]
                if not callable(apply_fn):
                    raise CustomCodeCompilationError(
                        "'apply_guardrail' must be a callable function"
                    )

                self._compiled_function = apply_fn
                verbose_proxy_logger.debug(
                    f"Custom code guardrail '{self.guardrail_name}' compiled successfully"
                )

            except SyntaxError as e:
                self._compile_error = f"Syntax error in custom code: {e}"
                raise CustomCodeCompilationError(self._compile_error) from e
            except CustomCodeCompilationError:
                raise
            except Exception as e:
                self._compile_error = f"Failed to compile custom code: {e}"
                raise CustomCodeCompilationError(self._compile_error) from e

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        """
        Apply the custom code guardrail to the inputs.

        This method calls the user-defined apply_guardrail function and
        processes its result to determine the appropriate action.

        The user-defined function can be either sync or async:
        - Sync: def apply_guardrail(inputs, request_data, input_type): ...
        - Async: async def apply_guardrail(inputs, request_data, input_type): ...

        Async functions are recommended when using http_request, http_get, or
        http_post primitives to avoid blocking the event loop.

        Args:
            inputs: Dictionary containing texts, images, tool_calls
            request_data: The original request data with metadata
            input_type: "request" for pre-call, "response" for post-call
            logging_obj: Optional logging object

        Returns:
            GenericGuardrailAPIInputs - possibly modified

        Raises:
            HTTPException: If content is blocked
            CustomCodeExecutionError: If execution fails
        """
        if self._compiled_function is None:
            if self._compile_error:
                raise CustomCodeExecutionError(
                    f"Custom code guardrail not compiled: {self._compile_error}"
                )
            raise CustomCodeExecutionError("Custom code guardrail not compiled")

        try:
            # Prepare inputs dict for the function

            # Prepare request_data with safe subset of information
            safe_request_data = self._prepare_safe_request_data(request_data)

            # Execute the custom function - handle both sync and async functions
            result = self._compiled_function(inputs, safe_request_data, input_type)

            # If the function is async (returns a coroutine), await it
            if asyncio.iscoroutine(result):
                result = await result

            # Process the result
            return self._process_result(
                result=result,
                inputs=inputs,
                request_data=request_data,
                input_type=input_type,
            )

        except HTTPException:
            # Re-raise HTTP exceptions (from block action)
            raise
        except Exception as e:
            verbose_proxy_logger.error(
                f"Custom code guardrail '{self.guardrail_name}' execution error: {e}"
            )
            raise CustomCodeExecutionError(
                f"Custom code guardrail execution failed: {e}",
                details={
                    "guardrail_name": self.guardrail_name,
                    "input_type": input_type,
                },
            ) from e

    def _prepare_safe_request_data(self, request_data: dict) -> Dict[str, Any]:
        """
        Prepare a safe subset of request_data for code execution.

        This filters out sensitive information and provides only what's
        needed for guardrail logic.

        Args:
            request_data: The full request data

        Returns:
            Safe subset of request data
        """
        return {
            "model": request_data.get("model"),
            "user_id": request_data.get("user_api_key_user_id"),
            "team_id": request_data.get("user_api_key_team_id"),
            "end_user_id": request_data.get("user_api_key_end_user_id"),
            "metadata": request_data.get("metadata", {}),
        }

    def _process_result(
        self,
        result: Any,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
    ) -> GenericGuardrailAPIInputs:
        """
        Process the result from the custom code function.

        Args:
            result: The return value from apply_guardrail
            inputs: The original inputs
            request_data: The request data
            input_type: "request" or "response"

        Returns:
            GenericGuardrailAPIInputs - possibly modified

        Raises:
            HTTPException: If action is "block"
        """
        if not isinstance(result, dict):
            verbose_proxy_logger.warning(
                f"Custom code guardrail '{self.guardrail_name}': "
                f"Expected dict result, got {type(result).__name__}. Treating as allow."
            )
            return inputs

        action = result.get("action", "allow")

        if action == "allow":
            verbose_proxy_logger.debug(
                f"Custom code guardrail '{self.guardrail_name}': Allowing {input_type}"
            )
            return inputs

        elif action == "block":
            reason = result.get("reason", "Blocked by custom code guardrail")
            detection_info = result.get("detection_info", {})

            verbose_proxy_logger.info(
                f"Custom code guardrail '{self.guardrail_name}': Blocking {input_type} - {reason}"
            )

            is_output = input_type == "response"

            # For pre-call, raise passthrough exception to return synthetic response
            if not is_output:
                self.raise_passthrough_exception(
                    violation_message=reason,
                    request_data=request_data,
                    detection_info=detection_info,
                )

            # For post-call, raise HTTP exception
            raise HTTPException(
                status_code=400,
                detail={
                    "error": reason,
                    "guardrail": self.guardrail_name,
                    "detection_info": detection_info,
                },
            )

        elif action == "modify":
            verbose_proxy_logger.debug(
                f"Custom code guardrail '{self.guardrail_name}': Modifying {input_type}"
            )

            # Apply modifications
            modified_inputs = dict(inputs)

            if "texts" in result and result["texts"] is not None:
                modified_inputs["texts"] = result["texts"]

            if "images" in result and result["images"] is not None:
                modified_inputs["images"] = result["images"]

            if "tool_calls" in result and result["tool_calls"] is not None:
                modified_inputs["tool_calls"] = result["tool_calls"]

            return cast(GenericGuardrailAPIInputs, modified_inputs)

        else:
            verbose_proxy_logger.warning(
                f"Custom code guardrail '{self.guardrail_name}': "
                f"Unknown action '{action}'. Treating as allow."
            )
            return inputs

    def update_custom_code(self, new_code: str) -> None:
        """
        Update the custom code and recompile.

        This method allows hot-reloading of guardrail logic without
        restarting the server.

        Args:
            new_code: The new source code

        Raises:
            CustomCodeCompilationError: If the new code fails to compile
        """
        with self._compile_lock:
            # Reset state
            old_function = self._compiled_function
            old_code = self.custom_code
            self._compiled_function = None
            self._compile_error = None

            try:
                self.custom_code = new_code
                self._compile_custom_code()
                verbose_proxy_logger.info(
                    f"Custom code guardrail '{self.guardrail_name}': Code updated successfully"
                )
            except CustomCodeCompilationError:
                # Rollback on failure
                self.custom_code = old_code
                self._compiled_function = old_function
                raise
