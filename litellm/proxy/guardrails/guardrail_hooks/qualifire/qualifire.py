# +-------------------------------------------------------------+
#
#           Use Qualifire for your LLM calls
#
# +-------------------------------------------------------------+
#  Qualifire - Evaluate LLM outputs for quality, safety, and reliability

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Union

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.litellm_core_utils.logging_utils import (
    convert_litellm_response_object_to_str,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.secret_managers.main import get_secret_str
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import CallTypesLiteral

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
            if tool_calls:
                qualifire_tool_calls = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        function_info = tc.get("function", {})
                        qualifire_tool_calls.append(
                            LLMToolCall(
                                id=tc.get("id", ""),
                                name=function_info.get("name", ""),
                                arguments=function_info.get("arguments", {}),
                            )
                        )
                if qualifire_tool_calls:
                    llm_message_kwargs["tool_calls"] = qualifire_tool_calls

            qualifire_messages.append(LLMMessage(**llm_message_kwargs))

        return qualifire_messages

    def _build_evaluate_kwargs(
        self,
        messages: List[Any],
        output: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build kwargs for client.evaluate() call."""
        kwargs: Dict[str, Any] = {"messages": messages}

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
            kwargs["tool_selection_quality_check"] = True
        if self.assertions:
            kwargs["assertions"] = self.assertions

        return kwargs

    async def _run_qualifire_evaluation(
        self,
        messages: List[AllMessageValues],
        request_data: Dict,
        event_type: GuardrailEventHooks,
        output: Optional[str] = None,
    ) -> None:
        """
        Run Qualifire evaluation on messages (and optionally output).
        Raises HTTPException if content is flagged and on_flagged is "block".
        """
        start_time = datetime.now()
        status = "success"
        qualifire_response: Optional[Dict] = None

        try:
            client = self._get_client()
            qualifire_messages = self._convert_messages_to_qualifire_format(messages)

            # Use invoke_evaluation if evaluation_id is provided
            if self.evaluation_id:
                # For invoke_evaluation, we need to extract input/output
                input_text = ""
                output_text = output or ""

                # Get the last user message as input
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            input_text = content
                        break

                result = client.invoke_evaluation(
                    evaluation_id=self.evaluation_id,
                    input=input_text,
                    output=output_text,
                )
            else:
                # Use evaluate with individual checks
                evaluate_kwargs = self._build_evaluate_kwargs(
                    messages=qualifire_messages,
                    output=output,
                )
                result = client.evaluate(**evaluate_kwargs)

            # Convert result to dict for logging
            qualifire_response = {
                "score": getattr(result, "score", None),
                "status": getattr(result, "status", None),
            }

            # Check if any evaluation flagged the content
            is_flagged = self._check_if_flagged(result)

            if is_flagged:
                if self.on_flagged == "monitor":
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
            status = "failure"
            raise
        except Exception as e:
            status = "guardrail_failed_to_respond"
            verbose_proxy_logger.error(f"Qualifire Guardrail error: {e}")
            raise
        finally:
            # Log guardrail information for observability
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_json_response=qualifire_response or {},
                request_data=request_data,
                guardrail_status=status,
                start_time=start_time.timestamp(),
                end_time=datetime.now().timestamp(),
                duration=(datetime.now() - start_time).total_seconds(),
                guardrail_provider="qualifire",
                event_type=event_type,
            )

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
            results = []
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

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: Any,
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
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Run Qualifire evaluation before the LLM call.
        Runs on input only.
        """
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        event_type = GuardrailEventHooks.pre_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return data

        messages: Optional[List[AllMessageValues]] = data.get("messages")
        if messages is None:
            verbose_proxy_logger.warning(
                "Qualifire Guardrail: not running guardrail. No messages in data"
            )
            return data

        await self._run_qualifire_evaluation(
            messages=messages,
            request_data=data,
            event_type=event_type,
        )

        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )

        return data

    @log_guardrail_information
    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal[
            "completion",
            "moderation",
            "responses",
            "mcp_call",
            "anthropic_messages",
        ],
    ):
        """
        Run Qualifire evaluation during the LLM call (in parallel).
        Runs on input only.
        """
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        event_type = GuardrailEventHooks.during_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return

        messages: Optional[List[AllMessageValues]] = data.get("messages")
        if messages is None:
            verbose_proxy_logger.warning(
                "Qualifire Guardrail: not running guardrail. No messages in data"
            )
            return

        await self._run_qualifire_evaluation(
            messages=messages,
            request_data=data,
            event_type=event_type,
        )

        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        """
        Run Qualifire evaluation after the LLM call.
        Runs on both input and output.
        """
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        event_type = GuardrailEventHooks.post_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return

        messages: Optional[List[AllMessageValues]] = data.get("messages")
        if messages is None:
            verbose_proxy_logger.warning(
                "Qualifire Guardrail: not running guardrail. No messages in data"
            )
            return

        # Get the response text
        response_str: Optional[str] = convert_litellm_response_object_to_str(response)

        await self._run_qualifire_evaluation(
            messages=messages,
            request_data=data,
            event_type=event_type,
            output=response_str,
        )

        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:  # type: ignore
        from litellm.types.proxy.guardrails.guardrail_hooks.qualifire import (
            QualifireGuardrailConfigModel,
        )

        return QualifireGuardrailConfigModel
