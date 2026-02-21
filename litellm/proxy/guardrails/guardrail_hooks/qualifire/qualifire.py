# +-------------------------------------------------------------+
#
#           Use Qualifire for your LLM calls
#
# +-------------------------------------------------------------+
#  Qualifire - Evaluate LLM outputs for quality, safety, and reliability

import json
import os
from typing import Any, Dict, List, Literal, Optional, Type

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel
from litellm.types.utils import GenericGuardrailAPIInputs

GUARDRAIL_NAME = "qualifire"
DEFAULT_QUALIFIRE_API_BASE = "https://proxy.qualifire.ai"


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
            api_base: Optional custom API base URL (defaults to https://api.qualifire.ai)
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
            or DEFAULT_QUALIFIRE_API_BASE
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

        # Initialize async HTTP client for direct API calls
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )

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

    def _convert_messages_to_api_format(
        self, messages: List[AllMessageValues]
    ) -> List[Dict[str, Any]]:
        """
        Convert LiteLLM messages to Qualifire API format.
        Supports tool calls for tool_selection_quality_check.

        Returns a list of dicts matching the API's ModelInvocationCanonicalMessage schema:
        {
            "role": "user" | "assistant" | "system" | "tool",
            "content": "...",
            "tool_call_id": "...",  # optional
            "tool_calls": [{"id": "...", "name": "...", "arguments": {...}}]  # optional
        }
        """
        api_messages = []
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

            api_message: Dict[str, Any] = {
                "role": role,
                "content": content if isinstance(content, str) else str(content),
            }

            # Handle tool_call_id for tool response messages
            tool_call_id = msg.get("tool_call_id")
            if tool_call_id:
                api_message["tool_call_id"] = tool_call_id

            # Handle tool calls if present
            tool_calls = msg.get("tool_calls")
            if tool_calls and isinstance(tool_calls, list):
                api_tool_calls = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        function_info = tc.get("function", {})
                        # Arguments can be a string (JSON) or dict
                        args = function_info.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {}
                        api_tool_calls.append(
                            {
                                "id": tc.get("id") or "",
                                "name": function_info.get("name") or "",
                                "arguments": args if isinstance(args, dict) else {},
                            }
                        )
                if api_tool_calls:
                    api_message["tool_calls"] = api_tool_calls

            api_messages.append(api_message)

        return api_messages

    def _convert_tools_to_api_format(
        self, tools: Optional[List[Any]]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Convert OpenAI-format tools to Qualifire API format.

        Returns a list of dicts matching the API's ModelInvocationToolDefinition schema:
        {
            "name": "...",
            "description": "...",
            "parameters": {...}
        }
        """
        if not tools:
            return None

        api_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                # Handle OpenAI function tool format
                if tool.get("type") == "function":
                    function_def = tool.get("function", {})
                    api_tools.append(
                        {
                            "name": function_def.get("name", ""),
                            "description": function_def.get("description", ""),
                            "parameters": function_def.get("parameters", {}),
                        }
                    )
                # Handle direct tool format
                elif "name" in tool:
                    api_tools.append(
                        {
                            "name": tool.get("name", ""),
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters", {}),
                        }
                    )

        return api_tools if api_tools else None

    def _check_if_flagged(self, result: Dict[str, Any]) -> bool:
        """
        Check if the Qualifire evaluation result indicates flagged content.

        Returns True only if there are explicitly flagged items in the evaluation results.
        A high score (close to 100) indicates GOOD content, low score indicates problems.
        """
        # Check evaluation results for any flagged items
        evaluation_results = result.get("evaluationResults", []) or []

        for eval_result in evaluation_results:
            results = eval_result.get("results", []) or []
            for r in results:
                if r.get("flagged"):
                    return True

        return False

    def _build_evaluate_payload(
        self,
        api_messages: List[Dict[str, Any]],
        output: Optional[str],
        assertions: Optional[List[str]],
        available_tools: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Build payload dictionary for the /api/evaluation/evaluate endpoint."""
        payload: Dict[str, Any] = {"messages": api_messages}

        if output is not None:
            payload["output"] = output

        # Add enabled checks
        if self.prompt_injections:
            payload["prompt_injections"] = True
        if self.hallucinations_check:
            payload["hallucinations_check"] = True
        if self.grounding_check:
            payload["grounding_check"] = True
        if self.pii_check:
            payload["pii_check"] = True
        if self.content_moderation_check:
            payload["content_moderation_check"] = True
        if self.tool_selection_quality_check:
            # Only enable tool_selection_quality_check if available_tools is provided
            if available_tools:
                payload["tool_selection_quality_check"] = True
                payload["available_tools"] = available_tools
            else:
                verbose_proxy_logger.debug(
                    "Qualifire Guardrail: tool_selection_quality_check enabled but no available_tools provided, skipping this check"
                )
        if assertions:
            payload["assertions"] = assertions

        return payload

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

        # Prepare headers
        headers = {
            "X-Qualifire-API-Key": self.qualifire_api_key or "",
            "Content-Type": "application/json",
        }

        try:
            # Convert messages to API format
            api_messages = self._convert_messages_to_api_format(messages)

            # Use invoke endpoint if evaluation_id is provided
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

                payload = {
                    "evaluation_id": evaluation_id,
                    "input": input_text,
                    "output": output or "",
                    "messages": api_messages,
                }

                # Convert tools if provided
                api_tools = self._convert_tools_to_api_format(available_tools)
                if api_tools:
                    payload["available_tools"] = api_tools

                url = f"{self.qualifire_api_base}/api/evaluation/invoke"
            else:
                # Use evaluate endpoint with individual checks
                api_tools = self._convert_tools_to_api_format(available_tools)
                payload = self._build_evaluate_payload(
                    api_messages=api_messages,
                    output=output,
                    assertions=assertions,
                    available_tools=api_tools,
                )
                url = f"{self.qualifire_api_base}/api/evaluation/evaluate"

            verbose_proxy_logger.debug(f"Qualifire Guardrail: Making request to {url}")

            # Make the API request
            response = await self.async_handler.post(
                url=url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            # Extract response info for logging
            qualifire_response = {
                "score": result.get("score"),
                "status": result.get("status"),
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

    @log_guardrail_information
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
