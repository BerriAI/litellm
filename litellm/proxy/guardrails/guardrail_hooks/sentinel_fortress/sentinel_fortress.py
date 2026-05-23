"""
Sentinel LLM Fortress V2 Guardrail Implementation.

This module provides integration with DeepKeep's Sentinel LLM Fortress V2
for comprehensive guardrail functionality including:
- PII Detection and Masking (via Presidio integration)
- Prompt Injection Detection
- Jailbreak Detection
- Topic Detection
- Custom detector support

For more information, see: https://deepkeep.ai/
"""

import asyncio
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import GuardrailEventHooks, LitellmParams
from litellm.types.utils import GenericGuardrailAPIInputs, GuardrailStatus

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class SentinelFortressMissingDependency(Exception):
    """Raised when sentinel-llm-fortress-v2 package is not installed."""


class SentinelFortressConfigError(Exception):
    """Raised when there's a configuration error with Sentinel Fortress."""


class SentinelFortressGuardrail(CustomGuardrail):
    """
    Guardrail that integrates with DeepKeep's Sentinel LLM Fortress V2.

    This guardrail provides comprehensive protection capabilities including:
    - PII Detection and Masking
    - Prompt Injection Detection
    - Jailbreak Detection
    - Topic Detection
    - Custom detector support

    Uses the unified guardrail system via `apply_guardrail` method,
    which automatically works with all LiteLLM endpoints.

    Configuration example in config.yaml:
    ```yaml
    guardrails:
      - guardrail_name: "sentinel-guard"
        litellm_params:
          guardrail: sentinel_fortress
          mode: "pre_call"
          default_on: true
          enabled_detectors:
            - pii
            - prompt_injection
          pii_entities_config:
            EMAIL_ADDRESS: "MASK"
            CREDIT_CARD: "BLOCK"
          on_flagged_action: "block"  # or "mask" or "monitor"
    ```
    """

    SUPPORTED_ON_FLAGGED_ACTIONS = {"block", "mask", "monitor", "passthrough"}
    DEFAULT_ON_FLAGGED_ACTION = "block"

    # Available detector types in Sentinel Fortress V2
    AVAILABLE_DETECTORS = {
        "pii",
        "prompt_injection",
        "jailbreak",
        "topic",
        "toxicity",
        "code_scanner",
        "secrets",
    }

    def __init__(
        self,
        guardrail_name: Optional[str] = "sentinel_fortress",
        enabled_detectors: Optional[List[str]] = None,
        detector_config: Optional[Dict[str, Any]] = None,
        on_flagged_action: Optional[str] = None,
        pii_entities_config: Optional[Dict[str, str]] = None,
        mask_request_content: bool = False,
        mask_response_content: bool = False,
        presidio_language: str = "en",
        sentinel_config_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Sentinel Fortress V2 Guardrail.

        Args:
            guardrail_name: Name identifier for this guardrail instance
            enabled_detectors: List of detector types to enable (e.g., ["pii", "prompt_injection"])
            detector_config: Configuration dictionary for individual detectors
            on_flagged_action: Action when content is flagged ("block", "mask", "monitor", "passthrough")
            pii_entities_config: PII entity configuration (entity -> action mapping)
            mask_request_content: Whether to mask content in requests
            mask_response_content: Whether to mask content in responses
            presidio_language: Language code for PII detection (default: "en")
            sentinel_config_path: Optional path to Sentinel configuration file
            **kwargs: Additional arguments passed to CustomGuardrail
        """
        # Validate and set on_flagged_action
        action = on_flagged_action
        if action and action.lower() in self.SUPPORTED_ON_FLAGGED_ACTIONS:
            self.on_flagged_action = action.lower()
        else:
            if action:
                verbose_proxy_logger.warning(
                    "Sentinel Fortress: Unsupported on_flagged_action '%s', defaulting to '%s'.",
                    action,
                    self.DEFAULT_ON_FLAGGED_ACTION,
                )
            self.on_flagged_action = self.DEFAULT_ON_FLAGGED_ACTION

        # Store configuration
        self.enabled_detectors = enabled_detectors or ["pii"]
        self.detector_config = detector_config or {}
        self.pii_entities_config = pii_entities_config or {}
        self.mask_request_content = mask_request_content
        self.mask_response_content = mask_response_content
        self.presidio_language = presidio_language
        self.sentinel_config_path = sentinel_config_path

        # Initialize detector instances (lazy loaded)
        self._detectors: Dict[str, Any] = {}
        self._fortress_initialized = False

        # Define supported event hooks
        supported_event_hooks = [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.during_call,
            GuardrailEventHooks.post_call,
            GuardrailEventHooks.logging_only,
        ]

        super().__init__(
            guardrail_name=guardrail_name,
            supported_event_hooks=supported_event_hooks,
            mask_request_content=mask_request_content,
            mask_response_content=mask_response_content,
            **kwargs,
        )

        verbose_proxy_logger.debug(
            "Sentinel Fortress initialized: detectors=%s, on_flagged_action=%s",
            self.enabled_detectors,
            self.on_flagged_action,
        )

    def _ensure_fortress_initialized(self) -> None:
        """
        Lazy initialization of Sentinel Fortress V2 components.
        This allows the guardrail to be configured even if sentinel is not installed,
        and only fails when actually trying to use it.
        """
        if self._fortress_initialized:
            return

        try:
            # Try to import sentinel fortress v2
            from sentinel.llm.fortress.v2 import (
                FortressManager,
            )

            # Initialize the fortress manager with configuration
            config = {
                "enabled_detectors": self.enabled_detectors,
                "detector_config": self.detector_config,
                "language": self.presidio_language,
            }

            if self.sentinel_config_path:
                config["config_path"] = self.sentinel_config_path

            self._fortress_manager = FortressManager(**config)
            self._fortress_initialized = True

            verbose_proxy_logger.info(
                "Sentinel Fortress V2 initialized successfully with detectors: %s",
                self.enabled_detectors,
            )

        except ImportError as e:
            verbose_proxy_logger.warning(
                "sentinel-llm-fortress-v2 package not installed. "
                "Using fallback mode with limited functionality. Error: %s",
                str(e),
            )
            # Set a flag to use fallback mode
            self._use_fallback_mode = True
            self._fortress_initialized = True

        except Exception as e:
            verbose_proxy_logger.error(
                "Failed to initialize Sentinel Fortress V2: %s", str(e)
            )
            raise SentinelFortressConfigError(
                f"Failed to initialize Sentinel Fortress: {str(e)}"
            ) from e

    def _initialize_pii_detector(self) -> Any:
        """Initialize the PII detector from Sentinel Fortress V2."""
        try:
            from sentinel.llm.fortress.v2.pii import PIIDetector

            return PIIDetector(
                language=self.presidio_language,
                entities_config=self.pii_entities_config,
            )
        except ImportError:
            verbose_proxy_logger.debug("PII detector not available, using fallback")
            return None

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        """
        Apply Sentinel Fortress V2 guardrail to extracted text content.

        This method is called by the unified guardrail system which handles
        extracting text from any request format (OpenAI, Anthropic, etc.).

        Args:
            inputs: Dictionary containing texts, images, and tool_calls
            request_data: The original request data
            input_type: "request" for pre-call, "response" for post-call
            logging_obj: Optional logging object

        Returns:
            GenericGuardrailAPIInputs - possibly modified inputs

        Raises:
            HTTPException: If content is blocked
        """
        start_time = datetime.now()
        status: GuardrailStatus = "success"
        detection_results: List[Dict[str, Any]] = []
        masked_entity_count: Dict[str, int] = {}

        try:
            texts = inputs.get("texts", [])
            if not texts:
                verbose_proxy_logger.debug("Sentinel Fortress: No texts to scan")
                return inputs

            verbose_proxy_logger.debug(
                "Sentinel Fortress: Scanning %d text(s) for %s",
                len(texts),
                input_type,
            )

            # Ensure fortress is initialized
            self._ensure_fortress_initialized()

            # Process each text
            processed_texts = []
            all_detections = []

            for text in texts:
                result = await self._process_text(
                    text=text,
                    input_type=input_type,
                    request_data=request_data,
                )
                processed_texts.append(result["text"])
                all_detections.extend(result.get("detections", []))

                # Track masked entities
                for detection in result.get("detections", []):
                    entity_type = detection.get("entity_type", "unknown")
                    masked_entity_count[entity_type] = (
                        masked_entity_count.get(entity_type, 0) + 1
                    )

            detection_results = all_detections

            # Check if we need to take action based on detections
            if all_detections:
                flagged = self._should_flag_content(all_detections)

                if flagged:
                    return self._handle_flagged_content(
                        inputs=inputs,
                        processed_texts=processed_texts,
                        detections=all_detections,
                        request_data=request_data,
                        input_type=input_type,
                    )

            # Return processed texts
            inputs["texts"] = processed_texts
            return inputs

        except HTTPException:
            status = "guardrail_failed_to_respond"
            raise
        except Exception as e:
            status = "guardrail_failed_to_respond"
            verbose_proxy_logger.exception(
                "Sentinel Fortress: Error processing content: %s", str(e)
            )
            raise
        finally:
            # Log guardrail information
            end_time = datetime.now()
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_provider="sentinel_fortress",
                guardrail_json_response=detection_results,
                request_data=request_data,
                guardrail_status=status,
                start_time=start_time.timestamp(),
                end_time=end_time.timestamp(),
                duration=(end_time - start_time).total_seconds(),
                masked_entity_count=masked_entity_count,
            )

    async def _process_text(
        self,
        text: str,
        input_type: Literal["request", "response"],
        request_data: dict,
    ) -> Dict[str, Any]:
        """
        Process a single text through enabled detectors.

        Args:
            text: The text to process
            input_type: Whether this is request or response content
            request_data: The original request data

        Returns:
            Dictionary with processed text and detections
        """
        result: Dict[str, Any] = {
            "text": text,
            "detections": [],
        }

        if not text or len(text.strip()) == 0:
            return result

        # Check if using fallback mode
        if getattr(self, "_use_fallback_mode", False):
            return await self._process_text_fallback(text, input_type)

        try:
            # Use Sentinel Fortress V2 for processing
            fortress_result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._fortress_manager.analyze(
                    text=text,
                    detectors=self.enabled_detectors,
                    context={"input_type": input_type},
                ),
            )

            # Extract detections
            detections = []
            for detector_name, detector_result in fortress_result.items():
                if detector_result.get("detected", False):
                    for entity in detector_result.get("entities", []):
                        detections.append(
                            {
                                "detector": detector_name,
                                "entity_type": entity.get("type", "unknown"),
                                "start": entity.get("start", 0),
                                "end": entity.get("end", len(text)),
                                "score": entity.get("score", 1.0),
                                "text": entity.get("text", ""),
                                "action": self._get_action_for_entity(
                                    entity.get("type"), detector_name
                                ),
                            }
                        )

            result["detections"] = detections

            # Apply masking if needed
            if self.on_flagged_action == "mask" and detections:
                result["text"] = self._apply_masking(text, detections)

            return result

        except Exception as e:
            verbose_proxy_logger.error(
                "Sentinel Fortress: Error in _process_text: %s", str(e)
            )
            # Return original text on error
            return result

    async def _process_text_fallback(
        self,
        text: str,
        input_type: Literal["request", "response"],
    ) -> Dict[str, Any]:
        """
        Fallback processing when Sentinel Fortress V2 is not available.
        Uses basic pattern matching for common PII types.
        """
        import re

        result: Dict[str, Any] = {
            "text": text,
            "detections": [],
        }

        # Basic PII patterns for fallback mode
        patterns = {
            "EMAIL_ADDRESS": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "PHONE_NUMBER": r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
            "CREDIT_CARD": r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",
            "IP_ADDRESS": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
            "US_SSN": r"\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b",
        }

        detections = []
        for entity_type, pattern in patterns.items():
            if entity_type in self.pii_entities_config or not self.pii_entities_config:
                for match in re.finditer(pattern, text):
                    detections.append(
                        {
                            "detector": "pii_fallback",
                            "entity_type": entity_type,
                            "start": match.start(),
                            "end": match.end(),
                            "score": 0.9,
                            "text": match.group(),
                            "action": self._get_action_for_entity(entity_type, "pii"),
                        }
                    )

        result["detections"] = detections

        # Apply masking if needed
        if self.on_flagged_action == "mask" and detections:
            result["text"] = self._apply_masking(text, detections)

        return result

    def _get_action_for_entity(self, entity_type: Optional[str], detector: str) -> str:
        """Get the action to take for a specific entity type."""
        if entity_type and entity_type in self.pii_entities_config:
            return self.pii_entities_config[entity_type]

        # Default action based on on_flagged_action
        if self.on_flagged_action == "mask":
            return "MASK"
        elif self.on_flagged_action == "block":
            return "BLOCK"
        return "MONITOR"

    def _should_flag_content(self, detections: List[Dict[str, Any]]) -> bool:
        """Determine if content should be flagged based on detections."""
        if not detections:
            return False

        # Check if any detection requires blocking
        for detection in detections:
            action = detection.get("action", "").upper()
            if action == "BLOCK":
                return True

        # In monitor mode, we don't flag
        if self.on_flagged_action == "monitor":
            return False

        # For other actions, flag if there are any detections
        return len(detections) > 0

    def _apply_masking(self, text: str, detections: List[Dict[str, Any]]) -> str:
        """Apply masking to detected entities in the text."""
        # Sort detections by position (reverse order to preserve positions)
        sorted_detections = sorted(
            detections,
            key=lambda x: x.get("start", 0),
            reverse=True,
        )

        masked_text = text
        for detection in sorted_detections:
            action = detection.get("action", "").upper()
            if action == "MASK" or (
                action != "BLOCK" and self.on_flagged_action == "mask"
            ):
                start = detection.get("start", 0)
                end = detection.get("end", len(text))
                entity_type = detection.get("entity_type", "REDACTED")
                mask = f"<{entity_type}>"
                masked_text = masked_text[:start] + mask + masked_text[end:]

        return masked_text

    def _handle_flagged_content(
        self,
        inputs: GenericGuardrailAPIInputs,
        processed_texts: List[str],
        detections: List[Dict[str, Any]],
        request_data: dict,
        input_type: Literal["request", "response"],
    ) -> GenericGuardrailAPIInputs:
        """Handle content that has been flagged by detectors."""

        # Check for blocking entities
        blocked_entities = [
            d for d in detections if d.get("action", "").upper() == "BLOCK"
        ]

        if self.on_flagged_action == "block" or blocked_entities:
            violation_location = "output" if input_type == "response" else "input"
            entity_types = list(
                set(
                    d.get("entity_type", "unknown")
                    for d in blocked_entities or detections
                )
            )

            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Blocked by Sentinel Fortress Guardrail",
                    "violation_location": violation_location,
                    "detected_entities": entity_types,
                    "detection_count": len(detections),
                    "guardrail": self.guardrail_name,
                },
            )

        elif self.on_flagged_action == "mask":
            inputs["texts"] = processed_texts
            return inputs

        elif self.on_flagged_action == "passthrough":
            # Create violation message for passthrough
            violation_message = self._format_violation_message(detections, input_type)

            if input_type == "request":
                # Short-circuit LLM call and return violation message
                self.raise_passthrough_exception(
                    violation_message=violation_message,
                    request_data=request_data,
                    detection_info={"detections": detections},
                )
            else:
                # For response, replace with violation message
                inputs["texts"] = [violation_message]

        else:  # monitor
            verbose_proxy_logger.info(
                "Sentinel Fortress: Monitoring mode - allowing flagged content"
            )
            inputs["texts"] = processed_texts

        return inputs

    def _format_violation_message(
        self,
        detections: List[Dict[str, Any]],
        input_type: Literal["request", "response"],
    ) -> str:
        """Format a user-friendly violation message."""
        location = "response" if input_type == "response" else "request"
        entity_types = list(set(d.get("entity_type", "unknown") for d in detections))

        message = (
            f"I'm sorry, but I cannot process this {location} as it contains "
            f"sensitive content detected by the Sentinel Fortress Guardrail. "
            f"Detected entity types: {', '.join(entity_types)}."
        )

        return self.render_violation_message(
            default=message,
            context={
                "location": location,
                "entity_types": entity_types,
                "detection_count": len(detections),
            },
        )

    # Legacy hook methods for backward compatibility
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Pre-call hook for request filtering.
        This is called before the LLM API call.
        """
        verbose_proxy_logger.debug(
            "Sentinel Fortress: async_pre_call_hook called for call_type=%s",
            call_type,
        )

        messages = data.get("messages")
        if not messages:
            return data

        # Extract texts from messages
        texts = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        texts.append(item["text"])

        if not texts:
            return data

        # Process through apply_guardrail
        inputs: GenericGuardrailAPIInputs = {"texts": texts}
        processed = await self.apply_guardrail(
            inputs=inputs,
            request_data=data,
            input_type="request",
        )

        # Update messages with processed texts
        processed_texts = processed.get("texts", texts)
        text_idx = 0
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str) and text_idx < len(processed_texts):
                msg["content"] = processed_texts[text_idx]
                text_idx += 1
            elif isinstance(content, list):
                for item in content:
                    if (
                        isinstance(item, dict)
                        and "text" in item
                        and text_idx < len(processed_texts)
                    ):
                        item["text"] = processed_texts[text_idx]
                        text_idx += 1

        return data

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
    ) -> Any:
        """
        Post-call hook for response filtering.
        This is called after a successful LLM API call.
        """
        from litellm.utils import ModelResponse

        verbose_proxy_logger.debug(
            "Sentinel Fortress: async_post_call_success_hook called"
        )

        if not isinstance(response, ModelResponse):
            return response

        # Extract texts from response
        texts = []
        for choice in response.choices:
            if hasattr(choice, "message") and choice.message:
                content = choice.message.content
                if isinstance(content, str):
                    texts.append(content)

        if not texts:
            return response

        # Process through apply_guardrail
        inputs: GenericGuardrailAPIInputs = {"texts": texts}
        processed = await self.apply_guardrail(
            inputs=inputs,
            request_data=data,
            input_type="response",
        )

        # Update response with processed texts
        processed_texts = processed.get("texts", texts)
        text_idx = 0
        for choice in response.choices:
            if (
                hasattr(choice, "message")
                and choice.message
                and text_idx < len(processed_texts)
            ):
                choice.message.content = processed_texts[text_idx]
                text_idx += 1

        return response

    def update_in_memory_litellm_params(self, litellm_params: LitellmParams) -> None:
        """Update guardrail parameters in memory."""
        super().update_in_memory_litellm_params(litellm_params)

        # Update Sentinel-specific params
        if (
            hasattr(litellm_params, "pii_entities_config")
            and litellm_params.pii_entities_config
        ):
            self.pii_entities_config = dict(litellm_params.pii_entities_config)  # type: ignore[arg-type]

        if (
            hasattr(litellm_params, "enabled_detectors")
            and litellm_params.enabled_detectors
        ):
            self.enabled_detectors = litellm_params.enabled_detectors
            # Reset fortress to reload with new detectors
            self._fortress_initialized = False
