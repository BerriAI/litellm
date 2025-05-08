from typing import Dict, List, Literal, Optional, Union

from litellm._logging import verbose_logger
from litellm.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.types.proxy.guardrails.guardrail_hooks.litellm_pii import (
    PiiAction,
    PiiEntityType,
)


class LitellmPIIGuardrail(CustomGuardrail):
    """
    LiteLLM Native PII Guardrail that uses Presidio to detect and mask or block PII in the input text.
    """

    def __init__(
        self,
        entities_config: Optional[Dict[PiiEntityType, PiiAction]] = None,
        language: str = "en",
        **kwargs,
    ):
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine

        super().__init__(**kwargs)
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.entities_config = entities_config or {}
        self.language = language

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
    ) -> Union[Exception, str, dict, None]:
        # Check the input messages for PII
        if call_type in ["completion", "text_completion"] and "messages" in data:
            messages = data.get("messages", [])
            for message in messages:
                if "content" in message and isinstance(message["content"], str):
                    result = self.analyze_and_handle_pii(message["content"])
                    if isinstance(result, Exception):
                        return result
                    elif result is not None:
                        message["content"] = result

        return data

    def presidio_analyzer_analyze(self, text: str):
        """
        Analyze text using Presidio Analyzer
        """
        # Build entities list from config
        entities = list(self.entities_config.keys()) if self.entities_config else None
        entities_str: List[str] = []
        # Convert Enum values to strings if entities is not None
        if entities:
            entities_str = [entity.value for entity in entities]

        # Call analyzer to get results
        results = self.analyzer.analyze(
            text=text, entities=entities_str, language=self.language
        )
        return results

    def analyze_and_handle_pii(self, text: str):
        """
        Analyze text for PII and apply the configured action (MASK or BLOCK)

        Returns:
        - None if no PII found
        - Exception if PII found and action is BLOCK
        - Modified text if PII found and action is MASK
        """
        from presidio_anonymizer.entities import RecognizerResult

        results = self.presidio_analyzer_analyze(text)

        verbose_logger.debug(f"Presidio Analyzer Results: {results}")

        if not results:
            return None

        # Check if any detected entity should trigger a BLOCK
        for result in results:
            entity_type = result.entity_type
            if (
                entity_type in self.entities_config
                and self.entities_config[entity_type] == PiiAction.BLOCK
            ):
                raise Exception(
                    f"PII detected: {entity_type}. Request blocked according to guardrail policy."
                )

        # If we've reached here, we need to mask any entities configured with MASK
        mask_entities = [
            RecognizerResult(
                entity_type=r.entity_type, start=r.start, end=r.end, score=r.score
            )
            for r in results
            if r.entity_type in self.entities_config
            and self.entities_config[r.entity_type] == PiiAction.MASK
        ]

        verbose_logger.info(f"Mask Entities: {mask_entities}")

        if mask_entities:
            anonymized_text = self.anonymizer.anonymize(
                text=text, analyzer_results=mask_entities
            ).text

            return anonymized_text

        return None
