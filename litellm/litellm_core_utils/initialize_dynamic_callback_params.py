from typing import Dict, Optional

from litellm.types.utils import StandardCallbackDynamicParams


def _is_env_reference(value: object) -> bool:
    return isinstance(value, str) and "os.environ/" in value


def _raise_env_reference_error(param: str, *, source: str) -> None:
    raise ValueError(
        f"Callback param '{param}' (from {source}) contains an 'os.environ/' "
        "reference. Environment references in request-supplied parameters are "
        "no longer resolved server-side for security reasons.\n"
        "To resolve:\n"
        "  1. Remove the 'os.environ/' reference from your request body / "
        "metadata.\n"
        "  2. Either (a) configure this callback value in your proxy "
        "config.yaml under 'litellm_settings' / 'general_settings', or "
        "(b) pass the resolved secret value directly in the request.\n"
        "See https://docs.litellm.ai/docs/proxy/logging for server-side "
        "callback configuration."
    )


def validate_no_callback_env_reference(
    param: str, value: object, *, source: str
) -> None:
    if _is_env_reference(value):
        _raise_env_reference_error(param, source=source)


# Hardcoded list of supported callback params to avoid runtime inspection issues with TypedDict
_supported_callback_params = [
    "langfuse_public_key",
    "langfuse_secret",
    "langfuse_secret_key",
    "langfuse_host",
    "langfuse_prompt_version",
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
]

_request_blocked_callback_params = {
    "gcs_bucket_name",
    "gcs_path_service_account",
}


def initialize_standard_callback_dynamic_params(
    kwargs: Optional[Dict] = None,
) -> StandardCallbackDynamicParams:
    """
    Initialize the standard callback dynamic params from the kwargs

    checks supported request callback params in kwargs and sets the corresponding attributes in StandardCallbackDynamicParams
    """

    standard_callback_dynamic_params = StandardCallbackDynamicParams()
    if kwargs:
        # 1. Check top-level kwargs
        for param in _supported_callback_params:
            if param in _request_blocked_callback_params:
                continue
            if param in kwargs:
                _param_value = kwargs.get(param)
                validate_no_callback_env_reference(
                    param, _param_value, source="request body"
                )
                standard_callback_dynamic_params[param] = _param_value  # type: ignore

        # 2. Fallback: check "metadata" or "litellm_params" -> "metadata"
        metadata = (kwargs.get("metadata") or {}).copy()
        litellm_params = kwargs.get("litellm_params") or {}
        if isinstance(litellm_params, dict):
            metadata.update(litellm_params.get("metadata") or {})

        if isinstance(metadata, dict):
            for param in _supported_callback_params:
                if param in _request_blocked_callback_params:
                    continue
                if param not in standard_callback_dynamic_params and param in metadata:
                    _param_value = metadata.get(param)
                    validate_no_callback_env_reference(
                        param, _param_value, source="metadata"
                    )
                    standard_callback_dynamic_params[param] = _param_value  # type: ignore

    return standard_callback_dynamic_params
