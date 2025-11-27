"""
Pass-Through Endpoint Message Handler for Unified Guardrails

This module provides a handler for passthrough endpoint requests.
It uses the field targeting configuration from the passthrough config
to extract specific fields for guardrail processing.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.proxy._types import (
    PassThroughGuardrailsConfig,
    PassThroughGuardrailSettings,
)

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail

# Key used to store passthrough guardrails config in request data
PASSTHROUGH_GUARDRAILS_CONFIG_KEY = "passthrough_guardrails_config"


def get_passthrough_guardrails_config(data: dict) -> Optional[dict]:
    """
    Get the passthrough guardrails config from request data.

    Checks both metadata and root level for the config.
    """
    # Check in metadata first (preferred location)
    metadata = data.get("metadata", {})
    if isinstance(metadata, dict) and PASSTHROUGH_GUARDRAILS_CONFIG_KEY in metadata:
        return metadata[PASSTHROUGH_GUARDRAILS_CONFIG_KEY]

    # Fallback to root level with underscore prefix (legacy)
    return data.get(f"_{PASSTHROUGH_GUARDRAILS_CONFIG_KEY}")


def set_passthrough_guardrails_config(
    data: dict, config: Optional[PassThroughGuardrailsConfig]
) -> None:
    """
    Set the passthrough guardrails config in request data metadata.
    """
    if config is None:
        return
    if "metadata" not in data:
        data["metadata"] = {}
    data["metadata"][PASSTHROUGH_GUARDRAILS_CONFIG_KEY] = config


class PassThroughEndpointHandler(BaseTranslation):
    """
    Handler for processing passthrough endpoint requests with guardrails.

    Uses the passthrough_guardrails_config stored in request metadata
    to determine which fields to extract for guardrail processing.
    """

    def _get_guardrail_settings(
        self,
        data: dict,
        guardrail_name: Optional[str],
    ) -> Optional[PassThroughGuardrailSettings]:
        """
        Get the guardrail settings for a specific guardrail from passthrough config.
        """
        from litellm.proxy.pass_through_endpoints.passthrough_guardrails import (
            PassthroughGuardrailHandler,
        )

        passthrough_config = get_passthrough_guardrails_config(data)
        if not passthrough_config or not guardrail_name:
            return None

        return PassthroughGuardrailHandler.get_settings(
            passthrough_config, guardrail_name
        )

    def _extract_text_for_guardrail(
        self,
        data: dict,
        field_expressions: Optional[List[str]],
    ) -> str:
        """
        Extract text from data for guardrail processing.

        If field_expressions provided, extracts only those fields.
        Otherwise, returns the full payload as JSON.
        """
        from litellm.proxy.pass_through_endpoints.jsonpath_extractor import (
            JsonPathExtractor,
        )

        if field_expressions:
            text = JsonPathExtractor.extract_fields(
                data=data,
                jsonpath_expressions=field_expressions,
            )
            verbose_proxy_logger.debug(
                "PassThroughEndpointHandler: Extracted targeted fields: %s",
                text[:200] if text else None,
            )
            return text

        # Use entire payload, excluding internal fields
        from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

        payload_to_check = {
            k: v
            for k, v in data.items()
            if not k.startswith("_")
            and k != "metadata"
            and k != PASSTHROUGH_GUARDRAILS_CONFIG_KEY
        }
        verbose_proxy_logger.debug(
            "PassThroughEndpointHandler: Using full payload for guardrail"
        )
        return safe_dumps(payload_to_check)

    async def process_input_messages(
        self,
        data: dict,
        guardrail_to_apply: "CustomGuardrail",
    ) -> Any:
        """
        Process input by applying guardrails to targeted fields or full payload.
        """
        guardrail_name = guardrail_to_apply.guardrail_name
        verbose_proxy_logger.debug(
            "PassThroughEndpointHandler: Processing input for guardrail=%s",
            guardrail_name,
        )

        # Get field targeting settings for this guardrail
        settings = self._get_guardrail_settings(data, guardrail_name)
        field_expressions = settings.request_fields if settings else None

        # Extract text to check
        text_to_check = self._extract_text_for_guardrail(data, field_expressions)

        if not text_to_check:
            verbose_proxy_logger.debug(
                "PassThroughEndpointHandler: No text to check, skipping guardrail"
            )
            return data

        # Apply guardrail
        await guardrail_to_apply.apply_guardrail(
            text=text_to_check,
            request_data=data,
        )

        return data

    async def process_output_response(
        self,
        response: Any,
        guardrail_to_apply: "CustomGuardrail",
    ) -> Any:
        """
        Process output response by applying guardrails to targeted fields.
        """
        if not isinstance(response, dict):
            verbose_proxy_logger.debug(
                "PassThroughEndpointHandler: Response is not a dict, skipping"
            )
            return response

        guardrail_name = guardrail_to_apply.guardrail_name
        verbose_proxy_logger.debug(
            "PassThroughEndpointHandler: Processing output for guardrail=%s",
            guardrail_name,
        )

        # Get field targeting settings for this guardrail
        settings = self._get_guardrail_settings(response, guardrail_name)
        field_expressions = settings.response_fields if settings else None

        # Extract text to check
        text_to_check = self._extract_text_for_guardrail(response, field_expressions)

        if not text_to_check:
            return response

        # Apply guardrail
        await guardrail_to_apply.apply_guardrail(
            text=text_to_check,
            request_data=response,
        )

        return response

