import re
from collections.abc import Iterator, Mapping
from typing import Any, Final

from litellm.types.utils import TRUSTED_CALLBACK_VARS_FIELD, StandardCallbackDynamicParams

_CLIENT_CALLBACK_METADATA_SLOTS: Final[tuple[str, ...]] = ("litellm_metadata", "metadata")


def iter_client_callback_metadata_dicts(
    kwargs: dict[str, Any],
) -> Iterator[tuple[str, dict[str, Any]]]:
    litellm_params: Final = kwargs.get("litellm_params")
    if isinstance(litellm_params, dict):
        nested: Final = litellm_params.get("metadata")
        if isinstance(nested, dict):
            yield "litellm_params.metadata", nested
    for key in _CLIENT_CALLBACK_METADATA_SLOTS:
        candidate = kwargs.get(key)
        if isinstance(candidate, dict):
            yield key, candidate


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


def validate_no_callback_env_reference(param: str, value: object, *, source: str) -> None:
    if _is_env_reference(value):
        _raise_env_reference_error(param, source=source)


# Langfuse rejects events whose environment does not match this pattern
# (lowercase alphanumerics, hyphens, underscores; no "langfuse" prefix).
# Validating here fails fast at config/init time instead of silently
# dropping every trace server-side.
LANGFUSE_ENVIRONMENT_PATTERN: Final = r"^(?!langfuse)[a-z0-9-_]+$"


def validate_langfuse_environment_value(value: str) -> None:
    if not re.match(LANGFUSE_ENVIRONMENT_PATTERN, value):
        raise ValueError(
            f"Invalid langfuse_environment {value!r}: must be lowercase "
            "alphanumerics/hyphens/underscores and must not start with "
            f"'langfuse' (pattern {LANGFUSE_ENVIRONMENT_PATTERN})"
        )


# Hardcoded list of supported callback params to avoid runtime inspection issues with TypedDict
_supported_callback_params: Final[tuple[str, ...]] = (
    "langfuse_public_key",
    "langfuse_secret",
    "langfuse_secret_key",
    "langfuse_host",
    "langfuse_environment",
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
    "dd_api_key",
    "dd_site",
    "dd_agent_host",
    "dd_agent_port",
    "newrelic_api_key",
    "newrelic_region",
    "turn_off_message_logging",
)

_request_blocked_callback_params: Final = frozenset(
    {
        "gcs_bucket_name",
        "gcs_path_service_account",
        "dd_api_key",
        "dd_site",
        "dd_agent_host",
        "dd_agent_port",
        "newrelic_api_key",
        "newrelic_region",
    }
)

# Request-blocked params that must still reach ``standard_callback_dynamic_params``
# when the proxy itself stamped them from admin-configured team/key callback
# settings (the trusted-vars channel). The OTel per-tenant tracer routing reads
# ``standard_callback_dynamic_params``, so without this overlay a blocked param
# could never drive routing at all.
_trusted_overlay_callback_params: Final = frozenset(
    {
        "newrelic_api_key",
        "newrelic_region",
    }
)


def get_trusted_callback_params(kwargs: Mapping[str, Any] | None) -> tuple[tuple[str, str], ...]:
    """
    Read callback params the proxy itself stamped from admin-configured team/key callback settings.

    Request-body values never reach this field: the proxy strips it from client input before
    setting it, so callbacks can consume credentials and destinations here without re-validating.

    Returned as pairs rather than a mapping because the caller keeps this on the Logging object,
    which the proxy deep-copies; a mappingproxy is not copyable and a dict would be mutable.
    """
    trusted_vars: Final = kwargs.get(TRUSTED_CALLBACK_VARS_FIELD) if kwargs else None
    if not isinstance(trusted_vars, Mapping):
        return ()
    return tuple((key, str(value)) for key, value in trusted_vars.items() if isinstance(key, str))


def initialize_standard_callback_dynamic_params(
    kwargs: dict | None = None,
) -> StandardCallbackDynamicParams:
    """
    Initialize the standard callback dynamic params from the kwargs

    checks supported request callback params in kwargs and sets the corresponding attributes in StandardCallbackDynamicParams
    """

    standard_callback_dynamic_params: Final = StandardCallbackDynamicParams()
    if kwargs:
        # 1. Check top-level kwargs
        for param in _supported_callback_params:
            if param in _request_blocked_callback_params:
                continue
            if param in kwargs:
                _param_value = kwargs.get(param)
                validate_no_callback_env_reference(param, _param_value, source="request body")
                standard_callback_dynamic_params[param] = (  # pyright: ignore[reportGeneralTypeIssues]  # several supported params predate their StandardCallbackDynamicParams fields
                    _param_value
                )

        for slot_label, metadata in iter_client_callback_metadata_dicts(kwargs):
            for param in _supported_callback_params:
                if param in _request_blocked_callback_params:
                    continue
                if param not in standard_callback_dynamic_params and param in metadata:
                    _param_value = metadata.get(param)
                    validate_no_callback_env_reference(param, _param_value, source=slot_label)
                    standard_callback_dynamic_params[param] = (  # pyright: ignore[reportGeneralTypeIssues]  # several supported params predate their StandardCallbackDynamicParams fields
                        _param_value
                    )

        for param, trusted_value in get_trusted_callback_params(kwargs):
            if param in _trusted_overlay_callback_params:
                standard_callback_dynamic_params[param] = trusted_value

    return standard_callback_dynamic_params
