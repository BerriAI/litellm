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
from typing import Final

from litellm.integrations.otel.model.metadata import RequestIdentity
from litellm.integrations.otel.model.semconv import GenAI, LiteLLM

# Attribute key -> value extractor over (identity, request_model,
# team_metadata_keys). The single definition of what may be promoted and under
# which key. Only the ``TEAM_METADATA`` extractor consults team_metadata_keys
# (to filter the team's metadata to an allowlist); the rest ignore it.
_PROMOTABLE: Final[
    dict[str, Callable[[RequestIdentity, str | None, tuple[str, ...]], str | None]]
] = {
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
    out: dict[str, str] = {}
    for key, extract in _PROMOTABLE.items():
        if key in promoted_keys:
            value = extract(identity, request_model, team_metadata_keys)
            if value:
                out[key] = value
    for meta_key in metadata_keys:
        value = identity.metadata.get(meta_key)
        if value:
            out[f"{LiteLLM.METADATA_PREFIX}{meta_key}"] = value
    return out


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
    filtered = {key: metadata[key] for key in allowed_keys if key in metadata}
    if not filtered:
        return None
    return json.dumps(filtered, default=str, sort_keys=True)
