from typing import Dict, Optional

from litellm._logging import verbose_logger
from litellm.types.utils import StandardCallbackDynamicParams


def _is_env_reference(value: object) -> bool:
    return isinstance(value, str) and "os.environ/" in value

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
                if _is_env_reference(_param_value):
                    verbose_logger.warning(
                        "Dropping callback param '%s': os.environ/ references "
                        "in request-supplied parameters are not resolved. "
                        "Configure this value server-side instead.",
                        param,
                    )
                    continue
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
                    if _is_env_reference(_param_value):
                        verbose_logger.warning(
                            "Dropping callback param '%s' from metadata: "
                            "os.environ/ references in request-supplied "
                            "parameters are not resolved. Configure this "
                            "value server-side instead.",
                            param,
                        )
                        continue
                    standard_callback_dynamic_params[param] = _param_value  # type: ignore

    return standard_callback_dynamic_params
