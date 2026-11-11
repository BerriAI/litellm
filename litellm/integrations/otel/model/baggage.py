"""Baggage promotion: request-identity values carried across child spans.

A bounded set of identity values is written into OpenTelemetry Baggage on the
LLM-call span so that child spans (guardrail, service) inherit them.
``providers.LiteLLMBaggageSpanProcessor`` reads Baggage at span start and stamps
the allowlisted keys onto every span.

This module is the single place baggage is defined: ``_PROMOTABLE`` maps each
promotable attribute key to how its value is read, and the ``*_KEYS`` defaults
select what is promoted unless the config overrides them. ``TEAM_METADATA``'s
extractor filters the team's free-form metadata to the sub-keys an operator
allowlists via ``baggage_team_metadata_keys`` (default none), so the blob is
never promoted whole.
"""

import json
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final

from litellm.integrations.otel.model.metadata import (
    REQUESTER_METADATA_PATH,
    RequestIdentity,
    allowlisted_metadata,
)
from litellm.integrations.otel.model.semconv import GenAI, LiteLLM

# Attribute key -> value extractor over (identity, request_model,
# team_metadata_keys). The single definition of what may be promoted and under
# which key. Only the ``TEAM_METADATA`` extractor consults team_metadata_keys
# (to filter the team's metadata to an allowlist); the rest ignore it.
_PROMOTABLE: Final[dict[str, Callable[[RequestIdentity, str | None, tuple[str, ...]], str | None]]] = {
    LiteLLM.TEAM_ID: lambda identity, model, team_metadata_keys: identity.team_id,
    LiteLLM.TEAM_ALIAS: lambda identity, model, team_metadata_keys: identity.team_alias,
    LiteLLM.TEAM_METADATA: lambda identity, model, team_metadata_keys: _filtered_team_metadata_json(
        identity.team_metadata, team_metadata_keys
    ),
    LiteLLM.KEY_HASH: lambda identity, model, team_metadata_keys: identity.key_hash,
    LiteLLM.END_USER: lambda identity, model, team_metadata_keys: identity.end_user,
    GenAI.REQUEST_MODEL: lambda identity, model, team_metadata_keys: model,
    LiteLLM.PROVIDER_MODEL: lambda identity, model, team_metadata_keys: identity.provider_model,
}

# Keys promoted by default (a subset of ``_PROMOTABLE``). ``END_USER`` is
# promotable but off by default — it identifies an individual user, so stamping
# it onto every span is opt-in via ``config.baggage_promoted_keys``.
BAGGAGE_PROMOTED_KEYS: Final[tuple[str, ...]] = (
    LiteLLM.TEAM_ID,
    LiteLLM.TEAM_ALIAS,
    LiteLLM.TEAM_METADATA,
    LiteLLM.KEY_HASH,
    GenAI.REQUEST_MODEL,
    LiteLLM.PROVIDER_MODEL,
)

# Metadata sub-keys eligible for promotion under the ``litellm.metadata.*``
# namespace. The full metadata blob is never promoted; only this allowlist is.
DEFAULT_BAGGAGE_METADATA_KEYS: Final[tuple[str, ...]] = (
    "user_api_key_org_id",
    "user_api_key_user_id",
    "user_api_key_alias",
    "user_api_key_end_user_id",
    "requester_ip_address",
)

# Sub-keys of the team's free-form metadata eligible for promotion under
# ``litellm.team.metadata``. Empty by default: a team's metadata can hold
# arbitrary operator data, so none of it is promoted until each key is
# explicitly allowlisted via ``config.baggage_team_metadata_keys``.
DEFAULT_BAGGAGE_TEAM_METADATA_KEYS: Final[tuple[str, ...]] = ()

# A team or key OTel v2 destination can add caller metadata keys on top of the
# global ``baggage_metadata_keys``, for its own destination only. Each key names
# a ``requester_metadata.<key>`` path, so only what the caller sent can be read,
# and a key whose exported name is a proxy-owned identity field is refused. The
# proxy copies the request headers into ``requester_metadata.headers``, credentials
# masked but anything a gateway in front of it added left as sent, so exporting a
# header stays an operator decision made in the global list.
MAX_DESTINATION_METADATA_KEYS: Final = 20
_RESERVED_METADATA_PREFIX: Final = "user_api_key_"
_REQUEST_HEADERS_KEY: Final = "headers"


def destination_metadata_keys_error(value: str) -> str | None:
    """Why ``value`` is not a usable destination ``baggage_metadata_keys`` list, or ``None``."""
    keys: Final = _listed_keys(value)
    named: Final = tuple((key, key.removeprefix(REQUESTER_METADATA_PATH)) for key in keys)
    if any(name == key or not name for key, name in named):
        return f"Invalid baggage_metadata_keys {value!r}: use a comma-separated list of {REQUESTER_METADATA_PATH}<key> entries"
    reserved: Final = tuple(key for key, name in named if _is_reserved_metadata_name(name))
    if reserved:
        return (
            f"Invalid baggage_metadata_keys {value!r}: {', '.join(reserved)} would overwrite a proxy identity attribute"
        )
    headers: Final = tuple(key for key, name in named if _is_request_header(name))
    if headers:
        return (
            f"Invalid baggage_metadata_keys {value!r}: {', '.join(headers)} would export request headers, "
            "which only the proxy-wide LITELLM_OTEL_BAGGAGE_METADATA_KEYS can do"
        )
    if len(frozenset(keys)) > MAX_DESTINATION_METADATA_KEYS:
        return f"Invalid baggage_metadata_keys {value!r}: at most {MAX_DESTINATION_METADATA_KEYS} keys"
    return None


def parse_destination_metadata_keys(value: object) -> tuple[str, ...] | None:
    """A destination's ``baggage_metadata_keys`` as a key tuple, ``None`` when unset or invalid."""
    if not isinstance(value, str) or destination_metadata_keys_error(value) is not None:
        return None
    return tuple(dict.fromkeys(_listed_keys(value)))


def _listed_keys(value: str) -> tuple[str, ...]:
    return tuple(key.strip() for key in value.split(","))


def _is_reserved_metadata_name(name: str) -> bool:
    return name in DEFAULT_BAGGAGE_METADATA_KEYS or name.startswith(_RESERVED_METADATA_PREFIX)


def _is_request_header(name: str) -> bool:
    return name.partition(".")[0].strip().lower() == _REQUEST_HEADERS_KEY


def promoted_baggage(
    identity: RequestIdentity,
    request_model: str | None,
    promoted_keys: tuple[str, ...],
    metadata_keys: tuple[str, ...] = DEFAULT_BAGGAGE_METADATA_KEYS,
    team_metadata_keys: tuple[str, ...] = DEFAULT_BAGGAGE_TEAM_METADATA_KEYS,
) -> dict[str, str]:
    """Identity values to write into Baggage, filtered to ``promoted_keys``.

    ``promoted_keys`` selects from ``_PROMOTABLE``; ``metadata_keys`` selects
    sub-keys of ``identity.metadata`` to promote under ``litellm.metadata.*``;
    ``team_metadata_keys`` selects sub-keys of the team's metadata to promote
    under ``litellm.team.metadata``. Empty values are dropped.
    """
    identity_values: Final = {
        key: value
        for key, extract in _PROMOTABLE.items()
        if key in promoted_keys and (value := extract(identity, request_model, team_metadata_keys))
    }
    return {**identity_values, **promoted_metadata(identity.metadata, metadata_keys)}


def promoted_metadata(metadata: Mapping[str, str], metadata_keys: tuple[str, ...]) -> Mapping[str, str]:
    """Allowlisted entries of a flattened metadata mapping under ``litellm.metadata.*``."""
    return MappingProxyType(
        {
            f"{LiteLLM.METADATA_PREFIX}{meta_key}": value
            for meta_key, value in allowlisted_metadata(metadata, metadata_keys).items()
        }
    )


def _filtered_team_metadata_json(
    metadata: Mapping[str, object] | None,
    allowed_keys: tuple[str, ...],
) -> str | None:
    """JSON-serialize only the allowlisted sub-keys of a team's metadata.

    Returns ``None`` when nothing is allowlisted or no allowlisted key is
    present, so the empty case is dropped rather than promoting ``"{}"``. Keys
    are sorted for a stable, diff-friendly value.
    """
    if not isinstance(metadata, Mapping) or not allowed_keys:
        return None
    filtered: Final = {key: metadata[key] for key in allowed_keys if key in metadata}
    if not filtered:
        return None
    return json.dumps(filtered, default=str, sort_keys=True)
