"""Alice WonderFence guardrail integration for LiteLLM."""

import os
from typing import TYPE_CHECKING, List, Literal, Optional, Type, Union

from fastapi import HTTPException

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_last_user_message,
)
from litellm.proxy.common_utils.callback_utils import (
    add_guardrail_to_applied_guardrails_header,
)
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.proxy.guardrails.guardrail_hooks.alice_wonderfence import (
    WonderFenceGuardrailConfigModel,
)
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


class WonderFenceMissingSecrets(Exception):
    """Raised when Alice API key is missing."""


class WonderFenceGuardrail(CustomGuardrail):
    """
    Alice WonderFence guardrail handler to evaluate prompts and responses.

    This class implements hooks to call the Alice WonderFence SDK for:
    - Pre-call: Evaluating user prompts before sending to the LLM
    - Post-call: Evaluating LLM responses before returning to users
    """

    def __init__(
        self,
        guardrail_name: str,
        api_key: str = "",
        api_base: Optional[str] = None,
        app_name: Optional[str] = None,
        api_timeout: float =20.0,
        platform: Optional[str] = None,
        event_hook: Optional[Union[GuardrailEventHooks, List[GuardrailEventHooks], Mode]] = None,
        default_on: bool = True,
    ):
        """
        Initialize the Alice WonderFence guardrail.

        Args:
            guardrail_name: The name of the guardrail instance
            api_key: Alice API key (can also be set via ALICE_API_KEY)
            api_base: Optional base URL for Alice WonderFence API
            app_name: Application name (reads from ALICE_APP_NAME or defaults to 'litellm')
            api_timeout: Timeout in seconds for API calls
            platform: Cloud platform (e.g., aws, azure, databricks)
            event_hook: Event hook mode
            default_on: Whether the guardrail is enabled by default
        """
        try:
            from wonderfence_sdk.client import WonderFenceClient
        except ImportError:
            raise ImportError(
                "Alice WonderFence SDK not installed. Install with: pip install wonderfence-sdk"
            )

        # Allow fallback to environment variable if api_key is empty string
        self.api_key = api_key if api_key else os.environ.get("ALICE_API_KEY")
        if not self.api_key:
            raise WonderFenceMissingSecrets(
                "Alice WonderFence API key not found. Set ALICE_API_KEY environment variable or pass it in litellm_params."
            )

        self.app_name = app_name or os.environ.get("ALICE_APP_NAME", "litellm")
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
            GuardrailEventHooks.during_call,
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

    def _build_analysis_context(self, request_data: dict):
        """
        Build WonderFence AnalysisContext from request data.

        Args:
            request_data: Request data dictionary

        Returns:
            AnalysisContext instance for WonderFence SDK
        """
        from wonderfence_sdk.models import AnalysisContext

        # request_data contains: model, metadata, litellm_metadata (with user_api_key_* fields)
        metadata = request_data.get("metadata") or request_data.get("litellm_metadata") or {}
        model_str = request_data.get("model", "")

        # Extract provider and clean model name
        provider = None
        model_name = model_str
        if model_str:
            try:
                model_name, provider, _, _ = litellm.get_llm_provider(model=model_str)
            except Exception:
                # Fallback: simple prefix extraction
                if "/" in model_str:
                    provider, model_name = model_str.split("/", 1)

        # User ID from API key metadata (prefixed by framework)
        user_id = (
            metadata.get("user_api_key_end_user_id")
            or metadata.get("end_user_id")
            or metadata.get("user_id")
        )

        # Session ID: Users pass `litellm_session_id` in request extra_body
        session_id = (
            request_data.get("litellm_session_id")  # Top-level field
            or metadata.get("litellm_session_id")  # In metadata
            or metadata.get("session_id")  # Legacy field name
        )

        return AnalysisContext(
            session_id=session_id,
            user_id=user_id,
            model_name=model_name,
            provider=provider,
            platform=self.platform,
        )

    def _extract_relevant_text(
        self, inputs: GenericGuardrailAPIInputs, input_type: Literal["request", "response"]
    ) -> Optional[str]:
        """
        Extract the relevant text based on input_type.

        For requests: Extract only the latest user message
        For responses: Extract only the latest assistant message

        Args:
            inputs: Standardized inputs with texts and structured_messages
            input_type: "request" or "response"

        Returns:
            The extracted text or None if not found
        """
        if input_type == "request":
            structured_messages = inputs.get("structured_messages", [])
            if structured_messages:
                # Use existing LiteLLM utility - extracts last user message
                return get_last_user_message(structured_messages)
            # Fallback to texts if no structured messages
            texts = inputs.get("texts", [])
            return texts[-1] if texts else None
        else:  # response
            # For responses, structured_messages is NOT populated
            # texts contains only the assistant response content from choices
            texts = inputs.get("texts", [])
            return texts[-1] if texts else None

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        """
        Apply WonderFence guardrail to inputs.

        This method is called by LiteLLM's guardrail framework for ALL endpoints:
        - /chat/completions
        - /responses
        - /messages (Anthropic)
        - /embeddings
        - /image/generations
        - /audio/transcriptions
        - /rerank
        - MCP server
        - and more...

        For requests: Only evaluates the latest user message
        For responses: Only evaluates the latest assistant message

        Args:
            inputs: Standardized inputs with texts, images, tool_calls, etc.
            request_data: Original request data
            input_type: "request" or "response"
            logging_obj: Optional logging object

        Returns:
            The inputs (potentially modified if action is MASK)

        Raises:
            HTTPException: If action is BLOCK
        """
        # Extract only the relevant text (latest user or assistant message)
        text = self._extract_relevant_text(inputs, input_type)

        if not text:
            verbose_proxy_logger.debug(
                f"Alice WonderFence (apply_guardrail): No relevant text found for {input_type}"
            )
            return inputs

        # Build context from request_data
        context = self._build_analysis_context(request_data)

        try:
            # Call appropriate WonderFence API
            if input_type == "request":
                verbose_proxy_logger.debug(
                    f"Alice WonderFence (apply_guardrail): Evaluating prompt for {self.guardrail_name}"
                )
                result = await self.client.evaluate_prompt(prompt=text, context=context)
            else:
                verbose_proxy_logger.debug(
                    f"Alice WonderFence (apply_guardrail): Evaluating response for {self.guardrail_name}"
                )
                result = await self.client.evaluate_response(response=text, context=context)

            action = result.action.value if hasattr(result.action, "value") else result.action

            if action == "BLOCK":
                # Hard block - return 400 error
                detail = {
                    "error": "Blocked by Alice WonderFence guardrail",
                    "guardrail_name": self.guardrail_name,
                    "action": "BLOCK",
                }
                if hasattr(result, "detections") and result.detections:
                    detail["detections"] = result.detections
                raise HTTPException(status_code=400, detail=detail)

            elif action == "MASK":
                # Mask - replace with sanitized text in the last relevant message
                masked_text = result.action_text or "[MASKED]"

                # Update only the last relevant text in the texts array
                texts = inputs.get("texts", [])
                if texts:
                    texts[-1] = masked_text
                    inputs["texts"] = texts

                verbose_proxy_logger.info(
                    f"Alice WonderFence (apply_guardrail): MASK action applied to {self.guardrail_name}"
                )

            else:  # DETECT, NO_ACTION
                if action == "DETECT":
                    verbose_proxy_logger.warning(
                        "Alice WonderFence (apply_guardrail): DETECT action"
                    )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Error in Alice WonderFence Guardrail",
                    "guardrail_name": self.guardrail_name,
                    "exception": str(e),
                },
            ) from e

        # Add to applied guardrails header
        add_guardrail_to_applied_guardrails_header(
            request_data=request_data, guardrail_name=self.guardrail_name
        )

        return inputs

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        """Return the config model for UI rendering."""
        return WonderFenceGuardrailConfigModel
