"""Baggage promotion: request-identity values carried across child spans.

A bounded set of identity values is written into OpenTelemetry Baggage on the
LLM-call span so that child spans (guardrail, service) inherit them.
``providers.LiteLLMBaggageSpanProcessor`` reads Baggage at span start and stamps
the allowlisted keys onto every span.

This module is the single place baggage is defined: ``_PROMOTABLE`` maps each
promotable attribute key to how its value is read, and the two ``*_KEYS``
defaults select what is promoted unless the config overrides them.
"""

from collections.abc import Callable
from typing import Final

from litellm.integrations.otel.model.metadata import RequestIdentity
from litellm.integrations.otel.model.semconv import GenAI, LiteLLM

# Attribute key -> value extractor over (identity, request_model). The single
# definition of what may be promoted and under which key.
_PROMOTABLE: Final[dict[str, Callable[[RequestIdentity, str | None], str | None]]] = {
    LiteLLM.TEAM_ID: lambda identity, model: identity.team_id,
    LiteLLM.TEAM_ALIAS: lambda identity, model: identity.team_alias,
    LiteLLM.TEAM_METADATA: lambda identity, model: identity.team_metadata,
    LiteLLM.KEY_HASH: lambda identity, model: identity.key_hash,
    LiteLLM.END_USER: lambda identity, model: identity.end_user,
    GenAI.REQUEST_MODEL: lambda identity, model: model,
    LiteLLM.PROVIDER_MODEL: lambda identity, model: identity.provider_model,
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


def promoted_baggage(
    identity: RequestIdentity,
    request_model: str | None,
    promoted_keys: tuple[str, ...],
    metadata_keys: tuple[str, ...] = DEFAULT_BAGGAGE_METADATA_KEYS,
) -> dict[str, str]:
    """Identity values to write into Baggage, filtered to ``promoted_keys``.

    ``promoted_keys`` selects from ``_PROMOTABLE``; ``metadata_keys`` selects
    sub-keys of ``identity.metadata`` to promote under ``litellm.metadata.*``.
    Empty values are dropped.
    """
    out: dict[str, str] = {}
    for key, extract in _PROMOTABLE.items():
        if key in promoted_keys:
            value = extract(identity, request_model)
            if value:
                out[key] = value
    for meta_key in metadata_keys:
        value = identity.metadata.get(meta_key)
        if value:
            out[f"{LiteLLM.METADATA_PREFIX}{meta_key}"] = value
    return out
