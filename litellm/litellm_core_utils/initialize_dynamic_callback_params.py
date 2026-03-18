from typing import Any, Dict, Optional
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import StandardCallbackDynamicParams

# Hardcoded list of supported callback params to avoid runtime inspection issues with TypedDict
_supported_callback_params = [
    "langfuse_public_key",
    "langfuse_secret",
    "langfuse_secret_key",
    "langfuse_host",
    "langfuse_prompt_version",
    "gcs_bucket_name",
    "gcs_path_service_account",
    "langsmith_api_key",
    "langsmith_project",
    "langsmith_base_url",
    "langsmith_sampling_rate",
    "langsmith_tenant_id",
    "humanloop_api_key",
    "arize_api_key",
    "arize_space_key",
    "arize_space_id",
    "posthog_api_key",
    "posthog_host",
    "braintrust_api_key",
    "braintrust_project",
    "braintrust_host",
    "slack_webhook_url",
    "lunary_public_key",
    "turn_off_message_logging",
]

# Sensitive params that should be redacted from metadata when returning key info
SENSITIVE_CALLBACK_PARAMS = {
    "langfuse_public_key",
    "langfuse_secret",
    "langfuse_secret_key",
    "langfuse_host",
    "gcs_path_service_account",
    "langsmith_api_key",
    "humanloop_api_key",
    "arize_api_key",
    "arize_space_key",
    "posthog_api_key",
    "braintrust_api_key",
    "slack_webhook_url",
    "lunary_public_key",
}

REDACTED_VALUE = "***REDACTED***"


def sanitize_metadata_for_key_info(metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Sanitize metadata by redacting sensitive callback credentials before returning in key info.

    This prevents API keys and secrets from being exposed when calling /key/info endpoint.

    Args:
        metadata: The metadata dictionary to sanitize

    Returns:
        A copy of the metadata with sensitive fields redacted
    """
    if metadata is None:
        return None

    sanitized = dict(metadata)
    for param in SENSITIVE_CALLBACK_PARAMS:
        if param in sanitized:
            sanitized[param] = REDACTED_VALUE

    return sanitized


def initialize_standard_callback_dynamic_params(
    kwargs: Optional[Dict] = None,
) -> StandardCallbackDynamicParams:
    """
    Initialize the standard callback dynamic params from the kwargs

    checks if langfuse_secret_key, gcs_bucket_name in kwargs and sets the corresponding attributes in StandardCallbackDynamicParams
    """

    standard_callback_dynamic_params = StandardCallbackDynamicParams()
    if kwargs:
        # 1. Check top-level kwargs
        for param in _supported_callback_params:
            if param in kwargs:
                _param_value = kwargs.get(param)
                if (
                    _param_value is not None
                    and isinstance(_param_value, str)
                    and "os.environ/" in _param_value
                ):
                    _param_value = get_secret_str(secret_name=_param_value)
                standard_callback_dynamic_params[param] = _param_value  # type: ignore

        # 2. Fallback: check "metadata" or "litellm_params" -> "metadata"
        metadata = (kwargs.get("metadata") or {}).copy()
        litellm_params = kwargs.get("litellm_params") or {}
        if isinstance(litellm_params, dict):
            metadata.update(litellm_params.get("metadata") or {})

        if isinstance(metadata, dict):
            for param in _supported_callback_params:
                if param not in standard_callback_dynamic_params and param in metadata:
                    _param_value = metadata.get(param)
                    if (
                        _param_value is not None
                        and isinstance(_param_value, str)
                        and "os.environ/" in _param_value
                    ):
                        _param_value = get_secret_str(secret_name=_param_value)
                    standard_callback_dynamic_params[param] = _param_value  # type: ignore

    return standard_callback_dynamic_params
