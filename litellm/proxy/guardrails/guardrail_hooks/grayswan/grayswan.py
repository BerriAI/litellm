"""Gray Swan Cygnal guardrail integration."""

import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    ModifyResponseException,
    log_guardrail_information,
)
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.litellm_core_utils.safe_json_loads import safe_json_loads
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

GRAYSWAN_BLOCK_ERROR_MSG = "Blocked by Gray Swan Guardrail"


class GraySwanGuardrailMissingSecrets(Exception):
    """Raised when the Gray Swan API key is missing."""


class GraySwanGuardrailAPIError(Exception):
    """Raised when the Gray Swan API returns an error."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class GraySwanGuardrail(CustomGuardrail):
    """
    Guardrail that calls Gray Swan's Cygnal monitoring endpoint.

    Uses the unified guardrail system via `apply_guardrail` method,
    which automatically works with all LiteLLM endpoints:
    - OpenAI Chat Completions
    - OpenAI Responses API
    - OpenAI Text Completions
    - Anthropic Messages
    - Image Generation
    - And more...

    see: https://docs.grayswan.ai/cygnal/monitor-requests
    """

    SUPPORTED_ON_FLAGGED_ACTIONS = {"block", "monitor", "passthrough"}
    DEFAULT_ON_FLAGGED_ACTION = "monitor"
    BASE_API_URL = "https://api.grayswan.ai"
    MONITOR_PATH = "/cygnal/monitor"
    SUPPORTED_REASONING_MODES = {"off", "hybrid", "thinking"}

    def __init__(
        self,
        guardrail_name: Optional[str] = "grayswan",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        on_flagged_action: Optional[str] = None,
        violation_threshold: Optional[float] = None,
        reasoning_mode: Optional[str] = None,
        categories: Optional[Dict[str, str]] = None,
        policy_id: Optional[str] = None,
        streaming_end_of_stream_only: bool = False,
        streaming_sampling_rate: int = 5,
        fail_open: Optional[bool] = True,
        guardrail_timeout: Optional[float] = 30.0,
        **kwargs: Any,
    ) -> None:
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )

        api_key_value = api_key or os.getenv("GRAYSWAN_API_KEY")
        if not api_key_value:
            raise GraySwanGuardrailMissingSecrets(
                "Gray Swan API key missing. Set `GRAYSWAN_API_KEY` or pass `api_key`."
            )
        self.api_key: str = api_key_value

        base = api_base or os.getenv("GRAYSWAN_API_BASE") or self.BASE_API_URL
        self.api_base = base.rstrip("/")
        self.monitor_url = f"{self.api_base}{self.MONITOR_PATH}"

        action = on_flagged_action
        if action and action.lower() in self.SUPPORTED_ON_FLAGGED_ACTIONS:
            self.on_flagged_action = action.lower()
        else:
            if action:
                verbose_proxy_logger.warning(
                    "Gray Swan Guardrail: Unsupported on_flagged_action '%s', defaulting to '%s'.",
                    action,
                    self.DEFAULT_ON_FLAGGED_ACTION,
                )
            self.on_flagged_action = self.DEFAULT_ON_FLAGGED_ACTION

        self.violation_threshold = self._resolve_threshold(violation_threshold)
        self.reasoning_mode = self._resolve_reasoning_mode(reasoning_mode)
        self.categories = categories
        self.policy_id = policy_id
        self.fail_open = True if fail_open is None else bool(fail_open)
        self.guardrail_timeout = (
            30.0 if guardrail_timeout is None else float(guardrail_timeout)
        )

        # Streaming configuration
        self.streaming_end_of_stream_only = streaming_end_of_stream_only
        self.streaming_sampling_rate = streaming_sampling_rate

        verbose_proxy_logger.debug(
            "GraySwan __init__: streaming_end_of_stream_only=%s, streaming_sampling_rate=%s",
            streaming_end_of_stream_only,
            streaming_sampling_rate,
        )

        supported_event_hooks = [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.during_call,
            GuardrailEventHooks.post_call,
        ]

        super().__init__(
            guardrail_name=guardrail_name,
            supported_event_hooks=supported_event_hooks,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Debug override to trace post_call issues
    # ------------------------------------------------------------------

    def should_run_guardrail(self, data, event_type) -> bool:
        """Override to add debug logging."""
        result = super().should_run_guardrail(data, event_type)
        # Check if apply_guardrail is in __dict__
        has_apply_guardrail = "apply_guardrail" in type(self).__dict__
        verbose_proxy_logger.debug(
            "GraySwan DEBUG: should_run_guardrail event_type=%s, result=%s, event_hook=%s, has_apply_guardrail=%s, class=%s",
            event_type,
            result,
            self.event_hook,
            has_apply_guardrail,
            type(self).__name__,
        )
        return result

    # ------------------------------------------------------------------
    # Unified Guardrail Interface (works with ALL endpoints automatically)
    # ------------------------------------------------------------------

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        """
        Apply Gray Swan guardrail to extracted text content.

        This method is called by the unified guardrail system which handles
        extracting text from any request format (OpenAI, Anthropic, etc.).

        Args:
            inputs: Dictionary containing:
                - texts: List of texts to scan
                - images: Optional list of images (not currently used by GraySwan)
                - tool_calls: Optional list of tool calls (not currently used)
            request_data: The original request data
            input_type: "request" for pre-call, "response" for post-call
            logging_obj: Optional logging object

        Returns:
            GenericGuardrailAPIInputs - texts may be replaced with violation message in passthrough mode

        Raises:
            HTTPException: If content is blocked (block mode)
            Exception: If guardrail check fails
        """
        # DEBUG: Log when apply_guardrail is called
        verbose_proxy_logger.debug(
            "GraySwan DEBUG: apply_guardrail called with input_type=%s, texts=%s",
            input_type,
            inputs.get("texts", [])[:100] if inputs.get("texts") else "NONE",
        )

        texts = inputs.get("texts", [])
        if not texts:
            verbose_proxy_logger.debug("Gray Swan Guardrail: No texts to scan")
            return inputs

        verbose_proxy_logger.debug(
            "Gray Swan Guardrail: Scanning %d text(s) for %s",
            len(texts),
            input_type,
        )

        # Convert texts to messages format for GraySwan API
        # Use "user" role for request content, "assistant" for response content
        role = "assistant" if input_type == "response" else "user"
        messages = [{"role": role, "content": text} for text in texts]

        # Get dynamic params from request metadata
        dynamic_body = (
            self.get_guardrail_dynamic_request_body_params(request_data) or {}
        )
        if dynamic_body:
            verbose_proxy_logger.debug(
                "Gray Swan Guardrail: dynamic extra_body=%s", safe_dumps(dynamic_body)
            )

        # Prepare and send payload
        payload = self._prepare_payload(messages, dynamic_body, request_data)
        if payload is None:
            return inputs

        start_time = time.time()
        try:
            response_json = await self._call_grayswan_api(payload)
            is_output = input_type == "response"
            result = self._process_response_internal(
                response_json=response_json,
                request_data=request_data,
                inputs=inputs,
                is_output=is_output,
            )
            return result
        except Exception as exc:
            if self._is_grayswan_exception(exc):
                raise
            end_time = time.time()
            status_code = getattr(exc, "status_code", None) or getattr(
                exc, "exception_status_code", None
            )
            self._log_guardrail_failure(
                exc=exc,
                request_data=request_data or {},
                start_time=start_time,
                end_time=end_time,
                status_code=status_code,
            )
            if self.fail_open:
                verbose_proxy_logger.warning(
                    "Gray Swan Guardrail: fail_open=True. Allowing request to proceed despite error: %s",
                    exc,
                )
                return inputs
            if isinstance(exc, GraySwanGuardrailAPIError):
                raise exc
            raise GraySwanGuardrailAPIError(str(exc), status_code=status_code) from exc

    def _is_grayswan_exception(self, exc: Exception) -> bool:
        # Guardrail decision (passthrough) should always propagate,
        # regardless of fail_open.
        if isinstance(exc, ModifyResponseException):
            return True
        detail = getattr(exc, "detail", None)
        if isinstance(detail, dict):
            return detail.get("error") == GRAYSWAN_BLOCK_ERROR_MSG
        return False

    # ------------------------------------------------------------------
    # Legacy Test Interface (for backward compatibility)
    # ------------------------------------------------------------------

    async def run_grayswan_guardrail(self, payload: dict) -> Dict[str, Any]:
        """
        Run the GraySwan guardrail on a payload.

        This is a legacy method for testing purposes.

        Args:
            payload: The payload to scan

        Returns:
            Dict containing the GraySwan API response
        """
        response_json = await self._call_grayswan_api(payload)
        # Call the legacy response processor (for test compatibility)
        self._process_grayswan_response(response_json)
        return response_json

    def _process_grayswan_response(
        self,
        response_json: dict,
        data: Optional[dict] = None,
        hook_type: Optional[GuardrailEventHooks] = None,
    ) -> None:
        """
        Legacy method for processing GraySwan API responses.

        This method is maintained for backward compatibility with existing tests.
        It handles the test scenarios where responses need to be processed with
        knowledge of the request context (pre/during/post call hooks).

        Args:
            response_json: Response from GraySwan API
            data: Optional request data (for passthrough exceptions)
            hook_type: Optional GuardrailEventHooks for determining behavior
        """
        violation_score = float(response_json.get("violation", 0.0) or 0.0)
        violated_rules = response_json.get("violated_rules", [])
        mutation_detected = response_json.get("mutation")
        ipi_detected = response_json.get("ipi")

        flagged = violation_score >= self.violation_threshold
        if not flagged:
            verbose_proxy_logger.debug(
                "Gray Swan Guardrail: content passed (score=%s, threshold=%s)",
                violation_score,
                self.violation_threshold,
            )
            return

        verbose_proxy_logger.warning(
            "Gray Swan Guardrail: violation score %.3f exceeds threshold %.3f",
            violation_score,
            self.violation_threshold,
        )

        detection_info = {
            "guardrail": "grayswan",
            "flagged": True,
            "violation_score": violation_score,
            "violated_rules": violated_rules,
            "mutation": mutation_detected,
            "ipi": ipi_detected,
        }

        # Determine if this is input (pre-call/during-call) or output (post-call)
        if hook_type is not None:
            is_input = hook_type in [
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.during_call,
            ]
        else:
            is_input = True

        if self.on_flagged_action == "block":
            violation_location = "output" if (not is_input) else "input"
            raise HTTPException(
                status_code=400,
                detail={
                    "error": GRAYSWAN_BLOCK_ERROR_MSG,
                    "violation_location": violation_location,
                    "violation": violation_score,
                    "violated_rules": violated_rules,
                    "mutation": mutation_detected,
                    "ipi": ipi_detected,
                },
            )
        elif self.on_flagged_action == "passthrough":
            # For passthrough mode, we need to handle violations
            detections = [detection_info]
            violation_message = self._format_violation_message(
                detections, is_output=not is_input
            )
            verbose_proxy_logger.info(
                "Gray Swan Guardrail: Passthrough mode - handling violation"
            )

            # If hook_type is provided and in pre/during call, raise exception
            if hook_type in [
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.during_call,
            ]:
                # Raise ModifyResponseException to short-circuit LLM call
                if data is None:
                    data = {}
                self.raise_passthrough_exception(
                    violation_message=violation_message,
                    request_data=data,
                    detection_info=detection_info,
                )
            elif hook_type == GuardrailEventHooks.post_call:
                # For post-call, store detection info in metadata
                if data is None:
                    data = {}
                if "metadata" not in data:
                    data["metadata"] = {}
                if "guardrail_detections" not in data["metadata"]:
                    data["metadata"]["guardrail_detections"] = []
                data["metadata"]["guardrail_detections"].append(detection_info)

    # ------------------------------------------------------------------
    # Core GraySwan API interaction
    # ------------------------------------------------------------------

    async def _call_grayswan_api(self, payload: dict) -> Dict[str, Any]:
        """Call the GraySwan monitoring API."""
        headers = self._prepare_headers()

        try:
            response = await self.async_handler.post(
                url=self.monitor_url,
                headers=headers,
                json=payload,
                timeout=self.guardrail_timeout,
            )
            response.raise_for_status()
            result = response.json()
            verbose_proxy_logger.debug(
                "Gray Swan Guardrail: monitor response %s", safe_dumps(result)
            )
            return result
        except Exception as exc:
            status_code = getattr(exc, "status_code", None) or getattr(
                exc, "exception_status_code", None
            )
            raise GraySwanGuardrailAPIError(str(exc), status_code=status_code) from exc

    def _process_response_internal(
        self,
        response_json: Dict[str, Any],
        request_data: dict,
        inputs: GenericGuardrailAPIInputs,
        is_output: bool,
    ) -> GenericGuardrailAPIInputs:
        """
        Process GraySwan API response and handle violations.

        Args:
            response_json: Response from GraySwan API
            request_data: Original request data
            inputs: The inputs being scanned
            is_output: True if scanning model output, False for input

        Returns:
            GenericGuardrailAPIInputs - possibly modified with violation message

        Raises:
            HTTPException: If content is blocked (block mode)
        """
        violation_score = float(response_json.get("violation", 0.0) or 0.0)
        violated_rules = response_json.get("violated_rule_descriptions", [])
        mutation_detected = response_json.get("mutation")
        ipi_detected = response_json.get("ipi")

        flagged = violation_score >= self.violation_threshold
        if not flagged:
            verbose_proxy_logger.debug(
                "Gray Swan Guardrail: content passed (score=%s, threshold=%s)",
                violation_score,
                self.violation_threshold,
            )
            return inputs

        verbose_proxy_logger.warning(
            "Gray Swan Guardrail: violation score %.3f exceeds threshold %.3f",
            violation_score,
            self.violation_threshold,
        )

        detection_info = {
            "guardrail": "grayswan",
            "flagged": True,
            "violation_score": violation_score,
            "violated_rules": violated_rules,
            "mutation": mutation_detected,
            "ipi": ipi_detected,
        }

        if self.on_flagged_action == "block":
            violation_location = "output" if is_output else "input"
            raise HTTPException(
                status_code=400,
                detail={
                    "error": GRAYSWAN_BLOCK_ERROR_MSG,
                    "violation_location": violation_location,
                    "violation": violation_score,
                    "violated_rules": violated_rules,
                    "mutation": mutation_detected,
                    "ipi": ipi_detected,
                },
            )
        elif self.on_flagged_action == "monitor":
            verbose_proxy_logger.info(
                "Gray Swan Guardrail: Monitoring mode - allowing flagged content"
            )
            return inputs
        elif self.on_flagged_action == "passthrough":
            # Replace content with violation message
            violation_message = self._format_violation_message(
                detection_info, is_output=is_output
            )
            verbose_proxy_logger.info(
                "Gray Swan Guardrail: Passthrough mode - replacing content with violation message"
            )

            if not is_output:
                # For pre-call (request), raise exception to short-circuit LLM call
                # and return synthetic response with violation message
                self.raise_passthrough_exception(
                    violation_message=violation_message,
                    request_data=request_data,
                    detection_info=detection_info,
                )

            # For post-call (response), replace texts and let unified system apply them
            inputs["texts"] = [violation_message]
            return inputs

        return inputs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "grayswan-api-key": self.api_key,
        }

    def _prepare_payload(
        self, messages: List[Dict[str, str]], dynamic_body: dict, request_data: dict
    ) -> Optional[Dict[str, Any]]:
        payload: Dict[str, Any] = {"messages": messages}

        categories = dynamic_body.get("categories") or self.categories
        if categories:
            payload["categories"] = categories

        policy_id = dynamic_body.get("policy_id") or self.policy_id
        if policy_id:
            payload["policy_id"] = policy_id

        reasoning_mode = dynamic_body.get("reasoning_mode") or self.reasoning_mode
        if reasoning_mode:
            payload["reasoning_mode"] = reasoning_mode

        # Pass through arbitrary metadata when provided via dynamic extra_body.
        if "metadata" in dynamic_body:
            payload["metadata"] = dynamic_body["metadata"]

        litellm_metadata = request_data.get("litellm_metadata")
        if isinstance(litellm_metadata, dict) and litellm_metadata:
            cleaned_litellm_metadata = dict(litellm_metadata)
            # cleaned_litellm_metadata.pop("user_api_key_auth", None)
            sanitized = safe_json_loads(
                safe_dumps(cleaned_litellm_metadata), default={}
            )
            if isinstance(sanitized, dict) and sanitized:
                payload["litellm_metadata"] = sanitized

        return payload

    def _format_violation_message(
        self, detection_info: Any, is_output: bool = False
    ) -> str:
        """
        Format detection info into a user-friendly violation message.

        Args:
            detection_info: Can be either:
                - A single dict with violation_score, violated_rules, mutation, ipi keys
                - A list of such dicts (legacy format)
            is_output: True if violation is in model output, False if in input

        Returns:
            Formatted violation message string
        """
        # Handle legacy format where detection_info is a list
        if isinstance(detection_info, list) and len(detection_info) > 0:
            detection_info = detection_info[0]

        # Extract fields from detection_info dict
        detection_dict: dict = (
            detection_info if isinstance(detection_info, dict) else {}
        )
        violation_score = detection_dict.get("violation_score", 0.0)
        violated_rules = detection_dict.get("violated_rules", [])
        mutation = detection_dict.get("mutation", False)
        ipi = detection_dict.get("ipi", False)

        violation_location = "the model response" if is_output else "input query"

        message_parts = [
            f"Sorry I can't help with that. According to the Gray Swan Cygnal Guardrail, "
            f"the {violation_location} has a violation score of {violation_score:.2f}.",
        ]

        if violated_rules:
            formatted_rules = self._format_violated_rules(violated_rules)
            if formatted_rules:
                message_parts.append(
                    f"It was violating the rule(s): {formatted_rules}."
                )

        if mutation:
            message_parts.append(
                "Mutation effort to make the harmful intention disguised was DETECTED."
            )

        if ipi:
            message_parts.append("Indirect Prompt Injection was DETECTED.")

        return "\n".join(message_parts)

    def _format_violated_rules(self, violated_rules: List) -> str:
        """Format violated rules list into a readable string."""
        formatted: List[str] = []
        for rule in violated_rules:
            if isinstance(rule, dict):
                # New format: {'rule': 6, 'name': 'Illegal Activities...', 'description': '...'}
                rule_num = rule.get("rule", "")
                rule_name = rule.get("name", "")
                rule_desc = rule.get("description", "")
                if rule_num and rule_name:
                    if rule_desc:
                        formatted.append(f"#{rule_num} {rule_name}: {rule_desc}")
                    else:
                        formatted.append(f"#{rule_num} {rule_name}")
                elif rule_name:
                    formatted.append(rule_name)
                else:
                    formatted.append(str(rule))
            else:
                # Legacy format: simple value
                formatted.append(str(rule))

        return ", ".join(formatted)

    def _resolve_threshold(self, value: Optional[float]) -> float:
        if value is not None:
            return float(value)
        env_val = os.getenv("GRAYSWAN_VIOLATION_THRESHOLD")
        if env_val:
            try:
                return float(env_val)
            except ValueError:
                pass
        return 0.5

    def _resolve_reasoning_mode(self, value: Optional[str]) -> Optional[str]:
        if value and value.lower() in self.SUPPORTED_REASONING_MODES:
            return value.lower()
        env_val = os.getenv("GRAYSWAN_REASONING_MODE")
        if env_val and env_val.lower() in self.SUPPORTED_REASONING_MODES:
            return env_val.lower()
        return None

    def _log_guardrail_failure(
        self,
        exc: Exception,
        request_data: dict,
        start_time: float,
        end_time: float,
        status_code: Optional[int] = None,
    ) -> None:
        """Log guardrail failure and attach standard logging metadata."""
        try:
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_json_response=str(exc),
                request_data=request_data,
                guardrail_status="guardrail_failed_to_respond",
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                guardrail_provider="grayswan",
            )
        except Exception:
            verbose_proxy_logger.exception(
                "Gray Swan Guardrail: failed to log guardrail failure for error: %s",
                exc,
            )
        verbose_proxy_logger.error(
            "Gray Swan Guardrail: API request failed%s: %s",
            f" (status_code={status_code})" if status_code else "",
            exc,
        )
