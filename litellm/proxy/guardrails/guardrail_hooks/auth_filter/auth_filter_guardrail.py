"""
Auth filter guardrail for LiteLLM.

This module provides a guardrail that executes user-defined Python-like code
to implement custom authentication enrichment/validation logic. The code runs
in a sandboxed environment with access to LiteLLM-provided primitives for
HTTP requests and other operations.

The auth_filter runs AFTER standard authentication completes, receiving the
validated UserAPIKeyAuth object. It can enrich, validate, or block based on
custom logic.

Example custom code (sync):

    def auth_filter(request, api_key, user_api_key_auth):
        '''Check organization-specific rules'''
        org_id = user_api_key_auth.organization_id
        if org_id == "restricted-org":
            return block("Organization access restricted")
        return allow()

Example custom code (async with HTTP):

    async def auth_filter(request, api_key, user_api_key_auth):
        '''Call external validation API'''
        org_id = user_api_key_auth.organization_id

        response = await http_post(
            "https://api.example.com/validate-org",
            body={"org_id": org_id}
        )

        if not response["success"] or not response["body"].get("valid"):
            return block("Organization validation failed")

        # Enrich with external data
        user_api_key_auth.metadata["validation_session"] = response["body"]["session_id"]
        return modify(user_api_key_auth=user_api_key_auth)
"""

import asyncio
import threading
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, Union

from fastapi import HTTPException, Request

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.custom_code.custom_code_guardrail import (
    CustomCodeCompilationError, CustomCodeExecutionError, CustomCodeGuardrail)
from litellm.proxy.guardrails.guardrail_hooks.custom_code.primitives import \
    get_custom_code_primitives
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.base import \
    GuardrailConfigModel

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import \
        Logging as LiteLLMLoggingObj


class AuthFilterGuardrailError(Exception):
    """Raised when auth filter guardrail execution fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.details = details or {}


class AuthFilterGuardrailConfigModel(GuardrailConfigModel):
    """Configuration parameters for the auth filter guardrail."""

    custom_code: str
    """The Python-like code containing the auth_filter function."""


class AuthFilterGuardrail(CustomCodeGuardrail):
    """
    Guardrail that executes user-defined auth filter code.

    The auth_filter runs AFTER standard authentication and receives:
    - request: Original FastAPI Request object
    - api_key: The API key used for authentication
    - user_api_key_auth: The validated UserAPIKeyAuth object

    Users write an `auth_filter(request, api_key, user_api_key_auth)` function
    that returns one of:
    - allow() - continue without modification
    - block(reason) - reject with a message
    - modify(user_api_key_auth={...}) - return enriched auth object

    Example:
        async def auth_filter(request, api_key, user_api_key_auth):
            org_id = user_api_key_auth.organization_id
            if org_id == "restricted":
                return block("Access denied")
            return allow()
    """

    def __init__(
        self,
        custom_code: str,
        guardrail_name: Optional[str] = "auth_filter",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the auth filter guardrail.

        Args:
            custom_code: The source code containing auth_filter function
            guardrail_name: Name of this guardrail instance
            **kwargs: Additional arguments passed to CustomGuardrail
        """
        self.custom_code = custom_code
        self._compiled_function: Optional[Any] = None
        self._compile_lock = threading.Lock()
        self._compile_error: Optional[str] = None

        # Auth filter only supports post_auth_check event hook
        supported_event_hooks = [GuardrailEventHooks.post_auth_check]

        # Initialize parent - do NOT call super().__init__() which would
        # call _compile_custom_code() with wrong function name
        from litellm.integrations.custom_guardrail import CustomGuardrail

        CustomGuardrail.__init__(
            self,
            guardrail_name=guardrail_name,
            supported_event_hooks=supported_event_hooks,
            **kwargs,
        )

        # Compile the code on initialization
        self._compile_custom_code()

    @staticmethod
    def get_config_model() -> Optional[Type[GuardrailConfigModel]]:
        """Returns the config model for the UI."""
        return AuthFilterGuardrailConfigModel

    def _compile_custom_code(self) -> None:
        """
        Compile the custom code and extract the auth_filter function.

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
                exec(compile(self.custom_code, "<auth_filter>", "exec"), exec_globals)

                # Extract the auth_filter function
                if "auth_filter" not in exec_globals:
                    raise CustomCodeCompilationError(
                        "Custom code must define an 'auth_filter' function. "
                        "Expected signature: auth_filter(request, api_key, user_api_key_auth)"
                    )

                auth_fn = exec_globals["auth_filter"]
                if not callable(auth_fn):
                    raise CustomCodeCompilationError(
                        "'auth_filter' must be a callable function"
                    )

                self._compiled_function = auth_fn
                verbose_proxy_logger.debug(
                    f"Auth filter guardrail '{self.guardrail_name}' compiled successfully"
                )

            except SyntaxError as e:
                self._compile_error = f"Syntax error in custom code: {e}"
                raise CustomCodeCompilationError(self._compile_error) from e
            except CustomCodeCompilationError:
                raise
            except Exception as e:
                self._compile_error = f"Failed to compile custom code: {e}"
                raise CustomCodeCompilationError(self._compile_error) from e

    async def execute_auth_filter(
        self,
        request: Request,
        api_key: str,
        user_api_key_auth: UserAPIKeyAuth,
    ) -> Union[UserAPIKeyAuth, None]:
        """
        Execute auth filter with the custom_auth-compatible signature.

        This method calls the user-defined auth_filter function and processes
        its result.

        Args:
            request: FastAPI Request object
            api_key: The API key used for authentication
            user_api_key_auth: The validated UserAPIKeyAuth object from standard auth

        Returns:
            - UserAPIKeyAuth: Modified/enriched auth object (replaces original)
            - None: Allow without modification (continue with original)

        Raises:
            HTTPException: If the filter blocks the request
            CustomCodeExecutionError: If execution fails
        """
        if self._compiled_function is None:
            raise AuthFilterGuardrailError("Auth filter not compiled")

        if self._compile_error:
            raise AuthFilterGuardrailError(
                f"Auth filter has compilation error: {self._compile_error}"
            )

        try:
            # Call the user's auth_filter function
            result = self._compiled_function(request, api_key, user_api_key_auth)

            # Handle async functions
            if asyncio.iscoroutine(result):
                result = await result

            # Process the result
            return self._process_auth_result(result, user_api_key_auth)

        except HTTPException:
            # Re-raise HTTP exceptions from block() calls
            raise
        except Exception as e:
            verbose_proxy_logger.error(
                f"Auth filter '{self.guardrail_name}' execution failed: {e}",
                exc_info=True,
            )
            raise CustomCodeExecutionError(
                f"Auth filter execution failed: {str(e)}",
                details={"guardrail_name": self.guardrail_name, "error": str(e)},
            ) from e

    def _process_auth_result(
        self, result: Any, original_auth: UserAPIKeyAuth
    ) -> Union[UserAPIKeyAuth, None]:
        """
        Convert auth_filter result to auth response.

        Expected result formats from custom code:
        - allow() -> {"action": "allow"} -> return None (no changes)
        - block(reason) -> {"action": "block", "reason": str} -> raise HTTPException
        - modify(user_api_key_auth=obj) -> {"action": "modify", "user_api_key_auth": dict} -> return modified auth

        Args:
            result: Result from user's auth_filter function
            original_auth: The original UserAPIKeyAuth object

        Returns:
            - UserAPIKeyAuth: If modified
            - None: If allow (no changes)

        Raises:
            HTTPException: If blocked
        """
        # If result is not a dict, assume allow
        if not isinstance(result, dict):
            return None

        action = result.get("action", "allow")

        if action == "allow":
            return None  # No changes, continue with original

        elif action == "block":
            reason = result.get("reason", "Blocked by auth filter")
            raise HTTPException(status_code=403, detail={"error": reason})

        elif action == "modify":
            # Return modified UserAPIKeyAuth
            if "user_api_key_auth" in result:
                modified_data = result["user_api_key_auth"]
                # If it's already a UserAPIKeyAuth, return it
                if isinstance(modified_data, UserAPIKeyAuth):
                    return modified_data
                # Otherwise try to construct from dict
                elif isinstance(modified_data, dict):
                    # Merge with original to preserve required fields
                    merged = original_auth.model_dump()
                    merged.update(modified_data)
                    return UserAPIKeyAuth(**merged)

        # Default: allow without changes
        return None

    def update_custom_code(self, new_code: str) -> None:
        """
        Update the custom code and recompile.

        This method is called by the hot-reload mechanism when the guardrail
        configuration changes in the database.

        Args:
            new_code: The new custom code to compile
        """
        with self._compile_lock:
            self.custom_code = new_code
            self._compiled_function = None
            self._compile_error = None
            self._compile_custom_code()

        verbose_proxy_logger.info(
            f"Auth filter guardrail '{self.guardrail_name}' hot-reloaded successfully"
        )
