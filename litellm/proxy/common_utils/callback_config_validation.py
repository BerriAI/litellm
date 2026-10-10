"""Save-time validation of team/key logging configs the runtime cannot honor.

Team callbacks arrive as a single ``AddTeamCallback``, key callbacks arrive as a
``logging`` list inside the key metadata, so both shapes funnel into the same
per-integration checks here.
"""

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

from litellm.types.utils import CAPTURE_MESSAGE_CONTENT_VAR

_NEWRELIC_CALLBACK: Final = "newrelic"
_NEWRELIC_VAR_PREFIX: Final = "newrelic_"
_LANGFUSE_OTEL_CALLBACK: Final = "langfuse_otel"
_LANGFUSE_SPAN_SCOPE_VAR: Final = "langfuse_span_scope"
_ARIZE_CALLBACK: Final = "arize"
_ARIZE_OTLP_PROTOCOL_VAR: Final = "arize_otlp_protocol"
_ARIZE_SAMPLING_RATE_VARS: Final[frozenset[str]] = frozenset(
    {"arize_success_sampling_rate", "arize_error_sampling_rate"}
)


def callback_config_error(
    callback_name: str | None, callback_vars: Mapping[str, str] | None, callback_type: str | None = None
) -> str | None:
    if not callback_vars:
        return None
    arize_error: Final = _arize_sampling_rate_error(callback_name, callback_vars) or _arize_otlp_protocol_error(
        callback_name, callback_vars, callback_type
    )
    if arize_error is not None:
        return arize_error
    langfuse_error: Final = _langfuse_environment_error(callback_vars) or _langfuse_span_scope_error(
        callback_name, callback_vars
    )
    if langfuse_error is not None:
        return langfuse_error
    capture_error: Final = _capture_message_content_error(callback_name, callback_vars, callback_type)
    if capture_error is not None:
        return capture_error
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


def _langfuse_span_scope_error(callback_name: str | None, callback_vars: Mapping[str, str]) -> str | None:
    value: Final = callback_vars.get(_LANGFUSE_SPAN_SCOPE_VAR)
    if value is None:
        return None
    if callback_name != _LANGFUSE_OTEL_CALLBACK:
        return (
            f"{_LANGFUSE_SPAN_SCOPE_VAR} applies to the {_LANGFUSE_OTEL_CALLBACK} callback only, not {callback_name!r}"
        )
    from litellm.litellm_core_utils.initialize_dynamic_callback_params import (
        validate_langfuse_span_scope_value,
    )

    try:
        validate_langfuse_span_scope_value(value)
    except ValueError as e:
        return str(e)
    return None


def _capture_message_content_error(
    callback_name: str | None, callback_vars: Mapping[str, str], callback_type: str | None
) -> str | None:
    value: Final = callback_vars.get(CAPTURE_MESSAGE_CONTENT_VAR)
    if value is None:
        return None
    from litellm.integrations.otel.model.config import is_otel_v2_enabled
    from litellm.integrations.otel.presets.destinations import destination_capable_backends
    from litellm.litellm_core_utils.initialize_dynamic_callback_params import (
        validate_capture_message_content_value,
    )

    supported: Final = sorted(destination_capable_backends())
    if callback_name not in supported:
        return f"{CAPTURE_MESSAGE_CONTENT_VAR} applies to the OTel v2 callbacks {supported} only, not {callback_name!r}"
    if not is_otel_v2_enabled():
        return f"Per-destination {CAPTURE_MESSAGE_CONTENT_VAR} requires the proxy to run with LITELLM_OTEL_V2=true."
    try:
        validate_capture_message_content_value(value)
    except ValueError as e:
        return str(e)
    if callback_type == "failure":
        return f"{CAPTURE_MESSAGE_CONTENT_VAR} needs callback_type 'success' or 'success_and_failure'"
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

_FAMILY_OPTION_VARS: Final[frozenset[str]] = frozenset(
    {_LANGFUSE_SPAN_SCOPE_VAR, _ARIZE_OTLP_PROTOCOL_VAR, *_ARIZE_SAMPLING_RATE_VARS}
)


def _family_of(var: str) -> str | None:
    """The credential family ``var`` configures, or ``None`` if it configures none.

    ``turn_off_message_logging`` and friends belong to no backend, so they carry
    no credentials anyone could redirect. ``langfuse_span_scope`` shares the Langfuse
    prefix but is a fixed enum choosing what the family exports, not where to.
    """
    if var in _FAMILY_OPTION_VARS:
        return None
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


def conflicting_span_scope_error(
    callback_vars: Mapping[str, str] | None,
    stored_vars_by_entry: Sequence[Mapping[str, str]],
) -> str | None:
    incoming: Final = None if callback_vars is None else callback_vars.get(_LANGFUSE_SPAN_SCOPE_VAR)
    if incoming is None:
        return None
    return next(
        (
            f"{_LANGFUSE_SPAN_SCOPE_VAR} is already set to {stored!r} by another callback entry. "
            f"Every entry shares one scope: remove that entry or send the same value."
            for entry in stored_vars_by_entry
            if (stored := entry.get(_LANGFUSE_SPAN_SCOPE_VAR)) not in (None, incoming)
        ),
        None,
    )


def conflicting_capture_error(
    callback_name: str,
    callback_vars: Mapping[str, str] | None,
    stored_entries: Sequence[tuple[str | None, Mapping[str, str]]],
) -> str | None:
    incoming: Final = None if callback_vars is None else callback_vars.get(CAPTURE_MESSAGE_CONTENT_VAR)
    if incoming is None:
        return None
    return next(
        (
            f"{CAPTURE_MESSAGE_CONTENT_VAR} is already set to {stored!r} by another {callback_name} entry. "
            f"Every {callback_name} entry shares one value: remove that entry or send the same value."
            for name, entry_vars in stored_entries
            if name == callback_name and (stored := entry_vars.get(CAPTURE_MESSAGE_CONTENT_VAR)) not in (None, incoming)
        ),
        None,
    )


def logging_metadata_config_error(metadata: Mapping[str, object] | None) -> str | None:
    """Validate every ``logging`` entry of a team/key metadata payload."""
    if not metadata:
        return None
    entries: Final = _object_sequence(metadata.get("logging"))
    if entries is None:
        return None
    entry_vars: Final = tuple(_entry_callback_vars(entry) for entry in entries)
    named_vars: Final = tuple(zip((_entry_callback_name(entry) for entry in entries), entry_vars))
    return next(
        (
            error
            for error in (
                *(_logging_entry_error(entry) for entry in entries),
                *(conflicting_span_scope_error(entry_vars[i], entry_vars[:i]) for i in range(len(entry_vars))),
                *(_conflicting_entry_capture_error(named_vars[i], named_vars[:i]) for i in range(len(named_vars))),
            )
            if error is not None
        ),
        None,
    )


def stored_capture_entries(logging_entries: object) -> tuple[tuple[str | None, Mapping[str, str]], ...]:
    """The callback name and vars of every stored ``logging`` entry, for ``conflicting_capture_error``."""
    entries: Final = _object_sequence(logging_entries) or ()
    return tuple((_entry_callback_name(entry), _entry_callback_vars(entry)) for entry in entries)


def _object_sequence(value: object) -> Sequence[object] | None:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else None


def _entry_callback_name(entry: object) -> str | None:
    match entry:
        case {"callback_name": str() as callback_name}:
            return callback_name
        case _:
            return None


def _conflicting_entry_capture_error(
    entry: tuple[str | None, Mapping[str, str]], earlier: Sequence[tuple[str | None, Mapping[str, str]]]
) -> str | None:
    callback_name, callback_vars = entry
    return None if callback_name is None else conflicting_capture_error(callback_name, callback_vars, earlier)


def _entry_callback_vars(entry: object) -> Mapping[str, str]:
    callback_vars: Final = entry.get("callback_vars") if isinstance(entry, Mapping) else None
    if not isinstance(callback_vars, Mapping):
        return MappingProxyType({})
    return MappingProxyType({str(key): str(value) for key, value in callback_vars.items()})


def _logging_entry_error(entry: object) -> str | None:
    if not isinstance(entry, Mapping):
        return None
    callback_name: Final = entry.get("callback_name")
    if not isinstance(callback_name, str) or not isinstance(entry.get("callback_vars"), Mapping):
        return None
    callback_type_raw: Final = entry.get("callback_type")
    return callback_config_error(
        callback_name, _entry_callback_vars(entry), callback_type_raw if isinstance(callback_type_raw, str) else None
    )


def _arize_sampling_rate_error(callback_name: str | None, callback_vars: Mapping[str, str]) -> str | None:
    for var in sorted(_ARIZE_SAMPLING_RATE_VARS):
        value = callback_vars.get(var)
        if value is None or value in ("", "None"):
            continue
        if callback_name != _ARIZE_CALLBACK:
            return f"{var} applies to the {_ARIZE_CALLBACK} callback only, not {callback_name!r}"
        try:
            rate = float(value)
        except (TypeError, ValueError):
            return f"{var} must be a number between 0.0 and 1.0 (inclusive), got {value!r}"
        if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
            return f"{var} must be a number between 0.0 and 1.0 (inclusive), got {value!r}"
    return None


def _arize_otlp_protocol_error(
    callback_name: str | None, callback_vars: Mapping[str, str], callback_type: str | None = None
) -> str | None:
    value: Final = callback_vars.get(_ARIZE_OTLP_PROTOCOL_VAR)
    if value is None:
        return None
    if callback_name != _ARIZE_CALLBACK:
        return f"{_ARIZE_OTLP_PROTOCOL_VAR} applies to the {_ARIZE_CALLBACK} callback only, not {callback_name!r}"
    from litellm.integrations.otel.model.config import is_otel_v2_enabled
    from litellm.litellm_core_utils.initialize_dynamic_callback_params import (
        validate_arize_otlp_protocol_value,
    )

    try:
        validate_arize_otlp_protocol_value(value)
    except ValueError as e:
        return str(e)
    if callback_type == "failure":
        return f"{_ARIZE_OTLP_PROTOCOL_VAR} needs callback_type 'success' or 'success_and_failure'; failure-only Arize callbacks export over the proxy's own Arize transport"
    if not is_otel_v2_enabled():
        return "Per-team Arize transport selection requires the proxy to run with LITELLM_OTEL_V2=true."
    return None


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
