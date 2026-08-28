"""Save-time validation of team/key logging configs the runtime cannot honor.

Team callbacks arrive as a single ``AddTeamCallback``, key callbacks arrive as a
``logging`` list inside the key metadata, so both shapes funnel into the same
per-integration checks here.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

_NEWRELIC_CALLBACK: Final = "newrelic"
_NEWRELIC_VAR_PREFIX: Final = "newrelic_"


def callback_config_error(callback_name: str | None, callback_vars: Mapping[str, str] | None) -> str | None:
    if not callback_vars:
        return None
    env_error: Final = _langfuse_environment_error(callback_vars)
    if env_error is not None:
        return env_error
    if callback_name != _NEWRELIC_CALLBACK:
        return None
    return _newrelic_config_error(callback_vars)


def _langfuse_environment_error(callback_vars: Mapping[str, str]) -> str | None:
    """Reject langfuse_environment values Langfuse ingestion would drop.

    Accepting an invalid value here would 200 the config write and then
    silently lose every trace for that key/team at request time.
    """
    value: Final = callback_vars.get("langfuse_environment")
    if value is None:
        return None
    from litellm.litellm_core_utils.initialize_dynamic_callback_params import (
        validate_langfuse_environment_value,
    )

    try:
        validate_langfuse_environment_value(value)
    except ValueError as e:
        return str(e)
    return None


def logging_metadata_config_error(metadata: Mapping[str, object] | None) -> str | None:
    """Validate every ``logging`` entry of a team/key metadata payload."""
    if not metadata:
        return None
    entries: Final = metadata.get("logging")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return None
    return next(
        (error for error in (_logging_entry_error(entry) for entry in entries) if error is not None),
        None,
    )


def _logging_entry_error(entry: object) -> str | None:
    if not isinstance(entry, Mapping):
        return None
    callback_name: Final = entry.get("callback_name")
    callback_vars: Final = entry.get("callback_vars")
    if not isinstance(callback_name, str) or not isinstance(callback_vars, Mapping):
        return None
    return callback_config_error(
        callback_name,
        MappingProxyType({str(key): str(value) for key, value in callback_vars.items()}),
    )


def _newrelic_config_error(callback_vars: Mapping[str, str]) -> str | None:
    """Per-team New Relic routing runs on the OTel v2 path only.

    Accepting the config with the flag off would silently ship the team's traffic
    through the operator's env-configured agent instead of the team's account. A
    region outside the fixed table, or a region without a key, would likewise be
    accepted and then silently ignored or misrouted at request time.
    """
    if not any(key.startswith(_NEWRELIC_VAR_PREFIX) for key in callback_vars):
        return None

    from litellm.integrations.otel.model.config import is_otel_v2_enabled
    from litellm.integrations.otel.presets.newrelic import NEWRELIC_OTLP_ENDPOINT_BY_REGION

    if not is_otel_v2_enabled():
        return "Per-team New Relic routing requires the proxy to run with LITELLM_OTEL_V2=true."

    region: Final = callback_vars.get("newrelic_region")
    if region is not None and region.lower() not in NEWRELIC_OTLP_ENDPOINT_BY_REGION:
        return (
            f"Unknown newrelic_region {region!r}. "
            f"Supported regions: {', '.join(sorted(NEWRELIC_OTLP_ENDPOINT_BY_REGION))}."
        )

    # ``callback_vars`` values are str()-coerced upstream, so a JSON ``null`` key
    # arrives as the literal ``"None"``; treat that and the empty string as absent.
    api_key: Final = callback_vars.get("newrelic_api_key")
    if region is not None and (not api_key or api_key == "None"):
        return "newrelic_region requires newrelic_api_key; the region rides the team's own key."
    return None
