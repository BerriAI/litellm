"""
Auth Filter Guardrail Hook

This module provides a guardrail for custom authentication enrichment and validation.
The auth_filter runs AFTER standard authentication, receiving the validated
UserAPIKeyAuth object and allowing custom code to enrich or validate it.

Configuration:
    guardrails:
      - guardrail_name: "my-auth-filter"
        litellm_params:
          guardrail: "auth_filter"
          mode: "post_auth_check"
          custom_code: |
            async def auth_filter(request, api_key, user_api_key_auth):
                # Custom validation logic
                org_id = user_api_key_auth.organization_id
                if org_id == "restricted":
                    return block("Access denied")
                return allow()

The custom code has access to HTTP primitives for external API calls:
- http_get(url, headers=None, timeout=30)
- http_post(url, body=None, headers=None, timeout=30)
- http_request(method, url, body=None, headers=None, timeout=30)

And other primitives from the custom_code sandbox:
- regex_match, regex_replace, json_parse, json_stringify, etc.
"""

from typing import Any, Dict

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .auth_filter_guardrail import AuthFilterGuardrail


def initialize_guardrail(litellm_params: Any, guardrail: Dict[str, Any]) -> AuthFilterGuardrail:
    """
    Initialize and register the auth filter guardrail.

    This function is called by the guardrail registry during startup or when
    a new auth_filter guardrail is created.

    Args:
        litellm_params: Configuration parameters (includes custom_code, mode, etc.)
        guardrail: The guardrail configuration dict

    Returns:
        AuthFilterGuardrail instance

    Raises:
        CustomCodeCompilationError: If the custom code fails to compile
    """
    guardrail_name = guardrail.get("guardrail_name", "auth_filter")
    custom_code = getattr(litellm_params, "custom_code", "")

    if not custom_code:
        raise ValueError("auth_filter guardrail requires 'custom_code' parameter")

    instance = AuthFilterGuardrail(
        guardrail_name=guardrail_name,
        custom_code=custom_code,
        event_hook=litellm_params.mode,
        default_on=getattr(litellm_params, "default_on", True),
    )

    # Register with LiteLLM callback manager for hot-reload support
    import litellm

    litellm.logging_callback_manager.add_litellm_callback(instance)

    return instance


# Register this guardrail with the system
guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.AUTH_FILTER.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.AUTH_FILTER.value: AuthFilterGuardrail,
}


__all__ = [
    "AuthFilterGuardrail",
    "initialize_guardrail",
    "guardrail_initializer_registry",
    "guardrail_class_registry",
]
