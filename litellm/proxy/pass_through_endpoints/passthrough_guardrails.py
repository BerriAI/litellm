"""
Passthrough Guardrails Helper Module

Handles guardrail execution for passthrough endpoints with:
- Opt-in model (guardrails only run when explicitly configured)
- Field-level targeting using JSONPath expressions
- Automatic inheritance from org/team/key levels when enabled
"""

from typing import Any, Dict, List, Optional, Union

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    PassThroughGuardrailsConfig,
    PassThroughGuardrailSettings,
    UserAPIKeyAuth,
)
from litellm.proxy.pass_through_endpoints.jsonpath_extractor import JsonPathExtractor

# Type for raw guardrails config input (before normalization)
# Can be a list of names or a dict with settings
PassThroughGuardrailsConfigInput = Union[
    List[str],  # Simple list: ["guard-1", "guard-2"]
    PassThroughGuardrailsConfig,  # Dict: {"guard-1": {"request_fields": [...]}}
]


class PassthroughGuardrailHandler:
    """
    Handles guardrail execution for passthrough endpoints.
    
    Passthrough endpoints use an opt-in model for guardrails:
    - Guardrails only run when explicitly configured on the endpoint
    - Supports field-level targeting using JSONPath expressions
    - Automatically inherits org/team/key level guardrails when enabled
    
    Guardrails can be specified as:
    - List format (simple): ["guardrail-1", "guardrail-2"]
    - Dict format (with settings): {"guardrail-1": {"request_fields": ["query"]}}
    """

    @staticmethod
    def normalize_config(
        guardrails_config: Optional[PassThroughGuardrailsConfigInput],
    ) -> Optional[PassThroughGuardrailsConfig]:
        """
        Normalize guardrails config to dict format.
        
        Accepts:
        - List of guardrail names: ["g1", "g2"] -> {"g1": None, "g2": None}
        - Dict with settings: {"g1": {"request_fields": [...]}}
        - None: returns None
        """
        if guardrails_config is None:
            return None
        
        # Already a dict - return as-is
        if isinstance(guardrails_config, dict):
            return guardrails_config
        
        # List of guardrail names - convert to dict
        if isinstance(guardrails_config, list):
            return {name: None for name in guardrails_config}
        
        verbose_proxy_logger.debug(
            "Passthrough guardrails config is not a dict or list, got: %s",
            type(guardrails_config),
        )
        return None

    @staticmethod
    def is_enabled(
        guardrails_config: Optional[PassThroughGuardrailsConfigInput],
    ) -> bool:
        """
        Check if guardrails are enabled for a passthrough endpoint.
        
        Passthrough endpoints are opt-in only - guardrails only run when 
        the guardrails config is set with at least one guardrail.
        """
        normalized = PassthroughGuardrailHandler.normalize_config(guardrails_config)
        if normalized is None:
            return False
        return len(normalized) > 0

    @staticmethod
    def get_guardrail_names(
        guardrails_config: Optional[PassThroughGuardrailsConfigInput],
    ) -> List[str]:
        """Get the list of guardrail names configured for a passthrough endpoint."""
        normalized = PassthroughGuardrailHandler.normalize_config(guardrails_config)
        if normalized is None:
            return []
        return list(normalized.keys())

    @staticmethod
    def get_settings(
        guardrails_config: Optional[PassThroughGuardrailsConfigInput],
        guardrail_name: str,
    ) -> Optional[PassThroughGuardrailSettings]:
        """Get settings for a specific guardrail from the passthrough config."""
        normalized = PassthroughGuardrailHandler.normalize_config(guardrails_config)
        if normalized is None:
            return None
        
        settings = normalized.get(guardrail_name)
        if settings is None:
            return None
        
        if isinstance(settings, dict):
            return PassThroughGuardrailSettings(**settings)
        
        return settings

    @staticmethod
    def prepare_input(
        request_data: dict,
        guardrail_settings: Optional[PassThroughGuardrailSettings],
    ) -> str:
        """
        Prepare input text for guardrail execution based on field targeting settings.
        
        If request_fields is specified, extracts only those fields.
        Otherwise, uses the entire request payload as text.
        """
        if guardrail_settings is None or guardrail_settings.request_fields is None:
            from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
            return safe_dumps(request_data)
        
        return JsonPathExtractor.extract_fields(
            data=request_data,
            jsonpath_expressions=guardrail_settings.request_fields,
        )

    @staticmethod
    def prepare_output(
        response_data: dict,
        guardrail_settings: Optional[PassThroughGuardrailSettings],
    ) -> str:
        """
        Prepare output text for guardrail execution based on field targeting settings.
        
        If response_fields is specified, extracts only those fields.
        Otherwise, uses the entire response payload as text.
        """
        if guardrail_settings is None or guardrail_settings.response_fields is None:
            from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
            return safe_dumps(response_data)
        
        return JsonPathExtractor.extract_fields(
            data=response_data,
            jsonpath_expressions=guardrail_settings.response_fields,
        )

    @staticmethod
    async def execute(
        request_data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        guardrails_config: Optional[PassThroughGuardrailsConfig],
        event_type: str = "pre_call",
    ) -> dict:
        """
        Execute guardrails for a passthrough endpoint.
        
        This is the main entry point for passthrough guardrail execution.
        
        Args:
            request_data: The request payload
            user_api_key_dict: User API key authentication info
            guardrails_config: Passthrough-specific guardrails configuration
            event_type: "pre_call" for request, "post_call" for response
        
        Returns:
            The potentially modified request_data
        
        Raises:
            HTTPException if a guardrail blocks the request
        """
        if not PassthroughGuardrailHandler.is_enabled(guardrails_config):
            verbose_proxy_logger.debug(
                "Passthrough guardrails not enabled, skipping guardrail execution"
            )
            return request_data
        
        guardrail_names = PassthroughGuardrailHandler.get_guardrail_names(
            guardrails_config
        )
        verbose_proxy_logger.debug(
            "Executing passthrough guardrails: %s", guardrail_names
        )
        
        # Add to request metadata so guardrails know which to run
        from litellm.proxy.pass_through_endpoints.passthrough_context import (
            set_passthrough_guardrails_config,
        )

        if "metadata" not in request_data:
            request_data["metadata"] = {}
        
        # Set guardrails in metadata using dict format for compatibility
        request_data["metadata"]["guardrails"] = {
            name: True for name in guardrail_names
        }
        
        # Store passthrough guardrails config in request-scoped context
        set_passthrough_guardrails_config(guardrails_config)
        
        return request_data

    @staticmethod
    def collect_guardrails(
        user_api_key_dict: UserAPIKeyAuth,
        passthrough_guardrails_config: Optional[PassThroughGuardrailsConfigInput],
    ) -> Optional[Dict[str, bool]]:
        """
        Collect guardrails for a passthrough endpoint.

        Passthrough endpoints are opt-in only for guardrails. Guardrails only run when
        the guardrails config is set with at least one guardrail.

        Accepts both list and dict formats:
        - List: ["guardrail-1", "guardrail-2"]
        - Dict: {"guardrail-1": {"request_fields": [...]}}

        When enabled, this function collects:
        - Passthrough-specific guardrails from the config
        - Org/team/key level guardrails (automatic inheritance when passthrough is enabled)

        Args:
            user_api_key_dict: User API key authentication info
            passthrough_guardrails_config: List or Dict of guardrail names/settings

        Returns:
            Dict of guardrail names to run (format: {guardrail_name: True}), or None
        """
        from litellm.proxy.litellm_pre_call_utils import (
            _add_guardrails_from_key_or_team_metadata,
        )

        # Normalize config to dict format (handles both list and dict)
        normalized_config = PassthroughGuardrailHandler.normalize_config(
            passthrough_guardrails_config
        )

        if normalized_config is None:
            verbose_proxy_logger.debug(
                "Passthrough guardrails not configured, skipping guardrail collection"
            )
            return None

        if len(normalized_config) == 0:
            verbose_proxy_logger.debug(
                "Passthrough guardrails config is empty, skipping"
            )
            return None

        # Passthrough is enabled - collect guardrails
        guardrails_to_run: Dict[str, bool] = {}

        # Add passthrough-specific guardrails
        for guardrail_name in normalized_config.keys():
            guardrails_to_run[guardrail_name] = True
            verbose_proxy_logger.debug(
                "Added passthrough-specific guardrail: %s", guardrail_name
            )

        # Add org/team/key level guardrails using shared helper
        temp_data: Dict[str, Any] = {"metadata": {}}
        _add_guardrails_from_key_or_team_metadata(
            key_metadata=user_api_key_dict.metadata,
            team_metadata=user_api_key_dict.team_metadata,
            data=temp_data,
            metadata_variable_name="metadata",
        )

        # Merge inherited guardrails into guardrails_to_run
        inherited_guardrails = temp_data["metadata"].get("guardrails", [])
        for guardrail_name in inherited_guardrails:
            if guardrail_name not in guardrails_to_run:
                guardrails_to_run[guardrail_name] = True
                verbose_proxy_logger.debug(
                    "Added inherited guardrail (key/team level): %s", guardrail_name
                )

        verbose_proxy_logger.debug(
            "Collected guardrails for passthrough endpoint: %s",
            list(guardrails_to_run.keys()),
        )

        return guardrails_to_run if guardrails_to_run else None

    @staticmethod
    def get_field_targeted_text(
        data: dict,
        guardrail_name: str,
        is_request: bool = True,
    ) -> Optional[str]:
        """
        Get the text to check for a guardrail, respecting field targeting settings.
        
        Called by guardrail hooks to get the appropriate text based on
        passthrough field targeting configuration.
        
        Args:
            data: The request/response data dict
            guardrail_name: Name of the guardrail being executed
            is_request: True for request (pre_call), False for response (post_call)
        
        Returns:
            The text to check, or None to use default behavior
        """
        from litellm.proxy.pass_through_endpoints.passthrough_context import (
            get_passthrough_guardrails_config,
        )

        passthrough_config = get_passthrough_guardrails_config()
        if passthrough_config is None:
            return None
        
        settings = PassthroughGuardrailHandler.get_settings(
            passthrough_config, guardrail_name
        )
        if settings is None:
            return None
        
        if is_request:
            if settings.request_fields:
                return JsonPathExtractor.extract_fields(data, settings.request_fields)
        else:
            if settings.response_fields:
                return JsonPathExtractor.extract_fields(data, settings.response_fields)
        
        return None
