"""Alice WonderFence guardrail integration for LiteLLM."""

import os
from typing import TYPE_CHECKING, List, Optional, Type, Union

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.callback_utils import (
    add_guardrail_to_applied_guardrails_header,
)
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.utils import ModelResponse

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


class WonderFenceMissingSecrets(Exception):
    """Raised when WonderFence API key is missing."""


class WonderFenceGuardrail(CustomGuardrail):
    """
    Alice WonderFence guardrail handler to evaluate prompts and responses.

    This class implements hooks to call the WonderFence SDK for:
    - Pre-call: Evaluating user prompts before sending to the LLM
    - Post-call: Evaluating LLM responses before returning to users
    """

    def __init__(
        self,
        guardrail_name: str,
        api_key: str = "",
        api_base: Optional[str] = None,
        app_name: Optional[str] = None,
        api_timeout: float = 10.0,
        platform: Optional[str] = None,
        event_hook: Optional[Union[GuardrailEventHooks, List[GuardrailEventHooks], Mode]] = None,
        default_on: bool = True,
    ):
        """
        Initialize the WonderFence guardrail.

        Args:
            guardrail_name: The name of the guardrail instance
            api_key: WonderFence API key (can also be set via WONDERFENCE_API_KEY)
            api_base: Optional base URL for WonderFence API
            app_name: Application name (reads from WONDERFENCE_APP_NAME or defaults to 'litellm')
            api_timeout: Timeout in seconds for API calls
            platform: Cloud platform (e.g., aws, azure, databricks)
            event_hook: Event hook mode
            default_on: Whether the guardrail is enabled by default
        """
        try:
            from wonderfence_sdk.client import WonderFenceClient
        except ImportError:
            raise ImportError(
                "WonderFence SDK not installed. Install with: pip install wonderfence-sdk"
            )

        # Allow fallback to environment variable if api_key is empty string
        self.api_key = api_key if api_key else os.environ.get("WONDERFENCE_API_KEY")
        if not self.api_key:
            raise WonderFenceMissingSecrets(
                "WonderFence API key not found. Set WONDERFENCE_API_KEY environment variable or pass it in litellm_params."
            )

        self.app_name = app_name or os.environ.get("WONDERFENCE_APP_NAME", "litellm")
        self.api_base = api_base
        self.api_timeout = api_timeout
        self.platform = platform

        # Initialize WonderFence client
        client_kwargs = {
            "api_key": self.api_key,
            "app_name": self.app_name,
            "api_timeout": int(self.api_timeout),
        }
        if self.api_base:
            client_kwargs["base_url"] = self.api_base
        if self.platform:
            client_kwargs["platform"] = self.platform

        self.client = WonderFenceClient(**client_kwargs)

        # Declare supported hooks
        supported_event_hooks = [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.post_call,
        ]

        super().__init__(
            guardrail_name=guardrail_name,
            event_hook=event_hook,
            default_on=default_on,
            supported_event_hooks=supported_event_hooks,
        )

        verbose_proxy_logger.debug(
            f"Initialized WonderFence Guardrail: name={guardrail_name}, app_name={self.app_name}"
        )

    def _build_analysis_context(self, user_api_key_dict: UserAPIKeyAuth, data: dict):
        """
        Build WonderFence AnalysisContext from request data.

        Args:
            user_api_key_dict: User API key authentication details
            data: Request data dictionary

        Returns:
            AnalysisContext instance for WonderFence SDK
        """
        from wonderfence_sdk.models import AnalysisContext
        import litellm

        metadata = data.get("metadata") or data.get("litellm_metadata") or {}
        model_str = data.get("model", "")

        # Extract provider and clean model name using LiteLLM's utility
        provider = None
        model_name = model_str
        if model_str:
            try:
                model_name, provider, _, _ = litellm.get_llm_provider(model=model_str)
            except Exception:
                # Fallback: simple prefix extraction
                if "/" in model_str:
                    provider, model_name = model_str.split("/", 1)

        return AnalysisContext(
            session_id=metadata.get("session_id"),
            user_id=user_api_key_dict.end_user_id or metadata.get("user_id"),
            model_name=model_name,
            provider=provider,
            platform=self.platform,
        )

    def _handle_evaluation_result(
        self, result, data: dict, message_obj: dict, hook_type: str
    ):
        """
        Handle WonderFence evaluation result and take appropriate action.

        Args:
            result: WonderFence evaluation result
            data: Request data dictionary
            message_obj: Message object being evaluated
            hook_type: Type of hook ('pre_call' or 'post_call')

        Returns:
            Modified data dictionary

        Raises:
            HTTPException: If action is BLOCK
        """
        # Access action as enum value or string
        action = result.action.value if hasattr(result.action, "value") else result.action

        if action == "BLOCK":
            # Include detections in error detail if available
            detail = {
                "error": "Blocked by WonderFence guardrail",
                "guardrail_name": self.guardrail_name,
                "action": "BLOCK",
            }
            if hasattr(result, "detections") and result.detections:
                detail["detections"] = result.detections
            raise HTTPException(
                status_code=400,
                detail=detail,
            )
        elif action == "MASK":
            # Replace content with masked version
            masked_content = result.action_text or "[MASKED]"
            if "content" in message_obj:
                message_obj["content"] = masked_content
            verbose_proxy_logger.info(
                f"WonderFence ({hook_type}): MASK action applied to {self.guardrail_name}"
            )
        elif action == "DETECT":
            # Log detection but continue
            verbose_proxy_logger.warning(
                f"WonderFence ({hook_type}): DETECT action"
            )
        # NO_ACTION: pass through unchanged

        return data

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ):
        """
        Guardrail hook to evaluate user prompts before the LLM call.

        Args:
            user_api_key_dict: User API key authentication details
            cache: Dual cache instance
            data: Request data dictionary
            call_type: Type of call being made

        Returns:
            Modified data dictionary or None
        """
        event_type = GuardrailEventHooks.pre_call
        if not self.should_run_guardrail(data=data, event_type=event_type):
            verbose_proxy_logger.debug(
                f"WonderFence Guardrail (async_pre_call_hook): Guardrail is disabled {self.guardrail_name}."
            )
            return data

        # Extract messages
        messages = data.get("messages", [])
        if not messages:
            verbose_proxy_logger.debug(
                "WonderFence (async_pre_call_hook): No messages found in request"
            )
            return data

        # Build context
        context = self._build_analysis_context(user_api_key_dict, data)

        # Evaluate only the last user message
        try:
            # Find the last user message (iterate backwards)
            last_user_message = None
            for message in reversed(messages):
                if message.get("role") == "user":
                    last_user_message = message
                    break

            if last_user_message:
                content = last_user_message.get("content")
                if isinstance(content, str) and content:
                    verbose_proxy_logger.debug(
                        f"WonderFence (async_pre_call_hook): Evaluating prompt for {self.guardrail_name}"
                    )
                    result = await self.client.evaluate_prompt(
                        prompt=content, context=context
                    )
                    data = self._handle_evaluation_result(
                        result, data, last_user_message, "pre_call"
                    )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Error in WonderFence Guardrail",
                    "guardrail_name": self.guardrail_name,
                    "exception": str(e),
                },
            ) from e

        # Add to applied guardrails header
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )
        return data

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        """
        Guardrail hook to evaluate LLM responses after the call.

        Args:
            data: Original request data dictionary
            user_api_key_dict: User API key authentication details
            response: LLM response object

        Returns:
            Modified response or original response
        """
        event_type = GuardrailEventHooks.post_call
        if not self.should_run_guardrail(data=data, event_type=event_type):
            verbose_proxy_logger.debug(
                f"WonderFence Guardrail (async_post_call_success_hook): Guardrail is disabled {self.guardrail_name}."
            )
            return response

        # Only handle ModelResponse
        if not isinstance(response, ModelResponse):
            verbose_proxy_logger.debug(
                "WonderFence (async_post_call_success_hook): Response is not ModelResponse, skipping"
            )
            return response

        # Build context
        context = self._build_analysis_context(user_api_key_dict, data)

        # Evaluate only the last choice
        try:
            if response.choices:
                choice = response.choices[-1]  # Get the last choice
                if hasattr(choice, "message") and choice.message:
                    content = choice.message.content
                    if isinstance(content, str) and content:
                        verbose_proxy_logger.debug(
                            f"WonderFence (async_post_call_success_hook): Evaluating response for {self.guardrail_name}"
                        )
                        result = await self.client.evaluate_response(
                            response=content, context=context
                        )
                        # For post-call, we need to handle the choice.message dict
                        message_dict = {"content": content}
                        self._handle_evaluation_result(
                            result, data, message_dict, "post_call"
                        )
                        # Update the actual choice content if it was masked
                        if message_dict["content"] != content:
                            choice.message.content = message_dict["content"]
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Error in WonderFence Guardrail",
                    "guardrail_name": self.guardrail_name,
                    "exception": str(e),
                },
            ) from e

        # Add to applied guardrails header
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )
        return response

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        """Return the config model for UI rendering."""
        from litellm.types.proxy.guardrails.guardrail_hooks.alice import (
            WonderFenceGuardrailConfigModel,
        )

        return WonderFenceGuardrailConfigModel
