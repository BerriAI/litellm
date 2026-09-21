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


# Which credential family a dynamic variable belongs to. The families are the
# integrations that share one account: every langfuse_* variable configures the
# same Langfuse project whether it rides the classic callback or the OTel one,
# and every dd_* variable configures the same Datadog account.
_VAR_FAMILIES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "arize_": "Arize",
        "dd_": "Datadog",
        "gcs_": "GCS",
        "humanloop_": "Humanloop",
        "langfuse_": "Langfuse",
        "langsmith_": "LangSmith",
        "newrelic_": "New Relic",
        "posthog_": "PostHog",
        "wandb_": "Weights & Biases",
        "weave_": "Weights & Biases",
    }
)


def _family_of(var: str) -> str | None:
    """The credential family ``var`` configures, or ``None`` if it configures none.

    ``turn_off_message_logging`` and friends belong to no backend, so they carry
    no credentials anyone could redirect.
    """
    return next((family for prefix, family in _VAR_FAMILIES.items() if var.startswith(prefix)), None)


def cross_entry_family_error(
    callback_vars: Mapping[str, str] | None,
    stored_vars_by_entry: Sequence[Mapping[str, str]],
) -> str | None:
    """Reject an entry that changes what a family another entry holds resolves to.

    Every stored entry's variables are flattened into one dict before a request
    reads them, and the flattened dict is what the exporter authenticates and
    addresses with. So an entry naming only a destination is enough to redirect
    credentials that were written somewhere else: a host on a second entry pairs
    with the key from the first, and the request carries that key to the new
    host.

    Two rules together keep the flattened dict out of the caller's hands. A
    variable the family already configures has to keep the value it has, so
    nothing already in use can be moved. A variable the family does not yet
    configure may only carry a value the family already holds, which is what lets
    the same credential go in under its other spelling (``langfuse_secret`` and
    ``langfuse_secret_key`` are one key) without anything here having to list the
    spellings. Between them, no value the caller chose can enter the family, and
    repeating the family as it stands is still allowed -- that is how one
    integration gets registered for both the success and the failure event.

    A team admin who does want to move a family deletes the entry holding it
    first, which reveals nothing.

    Only the writers this endpoint newly admits are held to this, because a proxy
    admin already holds every credential the proxy has.

    ``stored_vars_by_entry`` has to arrive decrypted; the credential values are
    encrypted at rest and ciphertext never equals the plaintext coming in.
    """
    if not callback_vars:
        return None
    stored_by_var: Final = {
        var: value for entry in stored_vars_by_entry for var, value in entry.items() if _family_of(var) is not None
    }
    family_values: Final = frozenset(
        (family, value)
        for entry in stored_vars_by_entry
        for var, value in entry.items()
        if (family := _family_of(var)) is not None
    )
    held_families: Final = frozenset(family for family, _ in family_values)
    return next(
        (
            f"{family} is already configured by another callback entry on this team. "
            f"Remove that entry before setting {var} here."
            for var, value, family in ((v, callback_vars[v], _family_of(v)) for v in callback_vars)
            if family in held_families
            and (stored_by_var[var] != value if var in stored_by_var else (family, value) not in family_values)
        ),
        None,
    )


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
