from typing import Dict, Optional

from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import StandardCallbackDynamicParams

# Callback config params - never send to external loggers (credential leak)
_SUPPORTED_CALLBACK_PARAMS_FROZEN = frozenset(
    [
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
        "litellm_logging_obj",
        "environment_variables",
    ]
)

# List form for iteration (excludes litellm_logging_obj, environment_variables - config only)
_supported_callback_params = [
    p for p in _SUPPORTED_CALLBACK_PARAMS_FROZEN if p not in ("litellm_logging_obj", "environment_variables")
]


def scrub_callback_config_params_from_dict(data: Dict) -> Dict:
    if not data:
        return data
    keys_to_remove = [
        k for k in data
        if k in _SUPPORTED_CALLBACK_PARAMS_FROZEN
        or k.endswith("_secret_key")
        or k.endswith("_secret")
        or k.endswith("_api_key")
    ]
    result = {k: v for k, v in data.items() if k not in keys_to_remove}
    return result


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
