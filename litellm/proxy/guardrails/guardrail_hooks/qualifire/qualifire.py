# +-------------------------------------------------------------+
#
#           Use Qualifire for your LLM calls
#
# +-------------------------------------------------------------+
#  Qualifire - Evaluate LLM outputs for quality, safety, and reliability

import os
from typing import Any, Dict, List, Literal, Optional, Type

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.litellm_logging import (
    Logging as LiteLLMLoggingObj,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel
from litellm.types.utils import GenericGuardrailAPIInputs

GUARDRAIL_NAME = "qualifire"


class QualifireGuardrail(CustomGuardrail):
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        evaluation_id: Optional[str] = None,
        prompt_injections: Optional[bool] = None,
        hallucinations_check: Optional[bool] = None,
        grounding_check: Optional[bool] = None,
        pii_check: Optional[bool] = None,
        content_moderation_check: Optional[bool] = None,
        tool_selection_quality_check: Optional[bool] = None,
        assertions: Optional[List[str]] = None,
        on_flagged: Optional[str] = "block",
        **kwargs,
    ):
        """
        Initialize the QualifireGuardrail class.

        Args:
            api_key: API key for Qualifire (or use QUALIFIRE_API_KEY env var)
            api_base: Optional custom API base URL
            evaluation_id: Pre-configured evaluation ID from Qualifire dashboard
            prompt_injections: Enable prompt injection detection (default if no other checks)
            hallucinations_check: Enable hallucination detection
            grounding_check: Enable grounding verification
            pii_check: Enable PII detection
            content_moderation_check: Enable content moderation
            tool_selection_quality_check: Enable tool selection quality check
            assertions: Custom assertions to validate against the output
            on_flagged: Action when content is flagged: "block" or "monitor"
        """
        self.qualifire_api_key = (
            api_key
            or get_secret_str("QUALIFIRE_API_KEY")
            or os.environ.get("QUALIFIRE_API_KEY")
        )
        self.qualifire_api_base = (
            api_base
            or get_secret_str("QUALIFIRE_BASE_URL")
            or os.environ.get("QUALIFIRE_BASE_URL")
        )
        self.evaluation_id = evaluation_id
        self.prompt_injections = prompt_injections
        self.hallucinations_check = hallucinations_check
        self.grounding_check = grounding_check
        self.pii_check = pii_check
        self.content_moderation_check = content_moderation_check
        self.tool_selection_quality_check = tool_selection_quality_check
        self.assertions = assertions
        self.on_flagged = on_flagged or "block"

        # If no checks are specified and no evaluation_id, default to prompt_injections
        if not self._has_any_check_enabled() and not self.evaluation_id:
            self.prompt_injections = True

        self._client = None
        super().__init__(**kwargs)

    def _has_any_check_enabled(self) -> bool:
        """Check if any evaluation check is explicitly enabled."""
        return any(
            [
                self.prompt_injections,
                self.hallucinations_check,
                self.grounding_check,
                self.pii_check,
                self.content_moderation_check,
                self.tool_selection_quality_check,
                self.assertions,
            ]
        )

    def _get_client(self):
        """Lazy initialization of Qualifire client."""
        if self._client is None:
            try:
                from qualifire.client import Client
            except ImportError:
                raise ImportError(
                    "qualifire package is required for QualifireGuardrail. "
                    "Install it with: pip install qualifire"
                )

            client_kwargs: Dict[str, Any] = {}
            if self.qualifire_api_key:
                client_kwargs["api_key"] = self.qualifire_api_key
            if self.qualifire_api_base:
                client_kwargs["base_url"] = self.qualifire_api_base

            self._client = Client(**client_kwargs)

        return self._client

    def _convert_messages_to_qualifire_format(
        self, messages: List[AllMessageValues]
    ) -> List[Any]:
        """
        Convert LiteLLM messages to Qualifire's LLMMessage format.
        Supports tool calls for tool_selection_quality_check.
        """
        try:
            from qualifire.types import LLMMessage, LLMToolCall
        except ImportError:
            raise ImportError(
                "qualifire package is required for QualifireGuardrail. "
                "Install it with: pip install qualifire"
            )

        qualifire_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Handle content that might be a list (multimodal)
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                content = "\n".join(text_parts)

            llm_message_kwargs: Dict[str, Any] = {
                "role": role,
                "content": content if isinstance(content, str) else str(content),
            }

            # Handle tool calls if present
            tool_calls = msg.get("tool_calls")
            if tool_calls and isinstance(tool_calls, list):
                qualifire_tool_calls = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        function_info = tc.get("function", {})
                        # Arguments can be a string (JSON) or dict
                        args = function_info.get("arguments", {})
                        if isinstance(args, str):
                            import json

                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {}
                        qualifire_tool_calls.append(
                            LLMToolCall(
                                id=tc.get("id") or "",
                                name=function_info.get("name") or "",
                                arguments=args if isinstance(args, dict) else {},
                            )
                        )
                if qualifire_tool_calls:
                    llm_message_kwargs["tool_calls"] = qualifire_tool_calls

            qualifire_messages.append(LLMMessage(**llm_message_kwargs))

        return qualifire_messages

    def _check_if_flagged(self, result: Any) -> bool:
        """
        Check if the Qualifire evaluation result indicates flagged content.

        Returns True only if there are explicitly flagged items in the evaluation results.
        A high score (close to 100) indicates GOOD content, low score indicates problems.
        """
        # Check evaluation results for any flagged items
        evaluation_results = getattr(result, "evaluationResults", None) or []
        if isinstance(result, dict):
            evaluation_results = result.get("evaluationResults", []) or []

        for eval_result in evaluation_results:
            results: List[Any] = []
            if isinstance(eval_result, dict):
                results = eval_result.get("results", []) or []
            else:
                results = getattr(eval_result, "results", []) or []

            for r in results:
                flagged = (
                    r.get("flagged")
                    if isinstance(r, dict)
                    else getattr(r, "flagged", False)
                )
                if flagged:
                    return True

        return False

    def _build_evaluate_kwargs(
        self,
        qualifire_messages: List[Any],
        output: Optional[str],
        assertions: Optional[List[str]],
        available_tools: Optional[List[Any]],
    ) -> Dict[str, Any]:
        """Build kwargs dictionary for the evaluate call."""
        kwargs: Dict[str, Any] = {"messages": qualifire_messages}

        if output is not None:
            kwargs["output"] = output

        # Add enabled checks
        if self.prompt_injections:
            kwargs["prompt_injections"] = True
        if self.hallucinations_check:
            kwargs["hallucinations_check"] = True
        if self.grounding_check:
            kwargs["grounding_check"] = True
        if self.pii_check:
            kwargs["pii_check"] = True
        if self.content_moderation_check:
            kwargs["content_moderation_check"] = True
        if self.tool_selection_quality_check:
            # Only enable tool_selection_quality_check if available_tools is provided
            if available_tools:
                kwargs["tool_selection_quality_check"] = True
                kwargs["available_tools"] = available_tools
            else:
                verbose_proxy_logger.debug(
                    "Qualifire Guardrail: tool_selection_quality_check enabled but no available_tools provided, skipping this check"
                )
        if assertions:
            kwargs["assertions"] = assertions

        return kwargs

    async def _run_qualifire_check(
        self,
        messages: List[AllMessageValues],
        output: Optional[str],
        dynamic_params: Dict[str, Any],
        available_tools: Optional[List[Any]] = None,
    ) -> None:
        """
        Core Qualifire check logic - shared between hooks.

        Args:
            messages: The conversation messages
            output: The LLM output text (for post_call)
            dynamic_params: Dynamic parameters from request body
            available_tools: Available tools from the request (for tool_selection_quality_check)

        Raises:
            HTTPException: If content is blocked
        """
        # Apply dynamic param overrides
        evaluation_id = dynamic_params.get("evaluation_id") or self.evaluation_id
        assertions = dynamic_params.get("assertions") or self.assertions
        on_flagged = dynamic_params.get("on_flagged") or self.on_flagged

        try:
            client = self._get_client()
            qualifire_messages = self._convert_messages_to_qualifire_format(messages)

            # Use invoke_evaluation if evaluation_id is provided
            if evaluation_id:
                # For invoke_evaluation, we need to extract input/output
                input_text = ""

                # Get the last user message as input
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            input_text = content
                        break

                result = client.invoke_evaluation(
                    evaluation_id=evaluation_id,
                    input=input_text,
                    output=output or "",
                )
            else:
                # Use evaluate with individual checks
                kwargs = self._build_evaluate_kwargs(
                    qualifire_messages=qualifire_messages,
                    output=output,
                    assertions=assertions,
                    available_tools=available_tools,
                )
                result = client.evaluate(**kwargs)

            # Convert result to dict for logging
            qualifire_response = {
                "score": getattr(result, "score", None),
                "status": getattr(result, "status", None),
            }

            verbose_proxy_logger.debug(
                "Qualifire Guardrail: Got result from API, score=%s, status=%s",
                qualifire_response["score"],
                qualifire_response["status"],
            )

            # Check if any evaluation flagged the content
            is_flagged = self._check_if_flagged(result)

            if is_flagged:
                if on_flagged == "monitor":
                    verbose_proxy_logger.warning(
                        "Qualifire Guardrail: Monitoring mode - violation detected but allowing request. "
                        f"Response: {qualifire_response}"
                    )
                else:
                    # Block the request
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "Violated guardrail policy",
                            "qualifire_response": qualifire_response,
                        },
                    )

        except HTTPException:
            raise
        except Exception as e:
            verbose_proxy_logger.exception(f"Qualifire Guardrail error: {e}")
            raise

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[LiteLLMLoggingObj] = None,
    ) -> GenericGuardrailAPIInputs:
        """
        Apply Qualifire guardrail to the given inputs.

        This method is called by the unified guardrail system for both
        input (request) and output (response) validation.

        Args:
            inputs: Dictionary containing:
                - texts: List of texts to check
                - structured_messages: Structured messages from the request (pre-call only)
                - tool_calls: Tool calls if present
            request_data: The original request data
            input_type: "request" for pre-call, "response" for post-call
            logging_obj: Optional logging object

        Returns:
            GenericGuardrailAPIInputs - unchanged if allowed through

        Raises:
            HTTPException: If content is blocked
        """
        # Get dynamic params from request body (allows runtime overrides)
        dynamic_params = self.get_guardrail_dynamic_request_body_params(
            request_data=request_data
        )

        # Extract messages from structured_messages or request_data
        messages: Optional[List[AllMessageValues]] = inputs.get("structured_messages")
        if not messages:
            messages = request_data.get("messages")

        # For response (post_call), messages may not be available in the inputs
        # We need to work with texts instead and construct messages if needed
        output: Optional[str] = None
        texts = inputs.get("texts", [])

        if input_type == "response":
            # For post_call, extract output from texts
            if texts:
                output = texts[-1] if isinstance(texts, list) else str(texts)

            # If no structured messages available, construct from texts
            if not messages and texts:
                # Create a simple message structure for the output
                messages = [{"role": "assistant", "content": output or ""}]  # type: ignore

        if not messages:
            # For pre_call with no messages, try to construct from texts
            if texts:
                messages = [{"role": "user", "content": texts[-1] if texts else ""}]  # type: ignore
            else:
                verbose_proxy_logger.debug(
                    "Qualifire Guardrail: No messages or texts found, skipping"
                )
                return inputs

        # Get available tools from request_data for tool_selection_quality_check
        available_tools = request_data.get("tools")

        await self._run_qualifire_check(
            messages=messages,
            output=output,
            dynamic_params=dynamic_params,
            available_tools=available_tools,
        )

        return inputs

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:  # type: ignore
        from litellm.types.proxy.guardrails.guardrail_hooks.qualifire import (
            QualifireGuardrailConfigModel,
        )

        return QualifireGuardrailConfigModel
