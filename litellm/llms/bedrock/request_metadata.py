"""
Resolve AWS Bedrock ``requestMetadata`` from LiteLLM proxy identity and caller metadata.

Bedrock attaches request metadata to CloudTrail records and to the dimension AWS Cost
Explorer groups on, so everything here is opt-in: nothing is forwarded unless the operator
sets ``litellm.bedrock_request_metadata_fields`` (``litellm_settings`` on the proxy).

Two properties are load-bearing for that billing record and are asserted by the tests:
proxy identity is resolved first so it can never be evicted by caller-supplied pairs, and the
whole ``user_api_key_`` prefix is reserved so a caller cannot write a proxy-authoritative
looking key. Values that break Bedrock's constraints are dropped rather than sanitised or
rejected, because an operator flipping this setting on must not turn a working request into a
400 and a silently rewritten attribution key is worse than an absent one.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Final

import litellm

BEDROCK_REQUEST_METADATA_HEADER: Final = "X-Amzn-Bedrock-Request-Metadata"
BEDROCK_REQUEST_METADATA_MAX_PAIRS: Final = 16
BEDROCK_REQUEST_METADATA_IDENTITY_PREFIX: Final = "user_api_key_"
BEDROCK_REQUEST_METADATA_CLIENT_FIELD: Final = "spend_logs_metadata"

_METADATA_PARAM_NAMES: Final[tuple[str, ...]] = ("metadata", "litellm_metadata")
_KEY_PATTERN: Final = re.compile(r"^[a-zA-Z0-9\s:_@$#=/+,.-]{1,256}$")
_VALUE_PATTERN: Final = re.compile(r"^[a-zA-Z0-9\s:_@$#=/+,.-]{0,256}$")
_OWNED_HEADER_NAMES: Final[frozenset[str]] = frozenset((BEDROCK_REQUEST_METADATA_HEADER.lower(),))


def _is_forwardable(key: str, value: str) -> bool:
    return _KEY_PATTERN.match(key) is not None and _VALUE_PATTERN.match(value) is not None


def _text_pairs(source: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(source, Mapping):
        return ()
    return tuple((key, value) for key, value in source.items() if isinstance(key, str) and isinstance(value, str))


def _allowed_fields() -> tuple[str, ...]:
    """
    The operator allow-list, deduplicated so a field repeated in config cannot consume a second
    reserved slot and shrink the client budget for nothing. First occurrence wins, which keeps
    the operator's declared precedence intact.
    """
    configured: Final[object] = litellm.bedrock_request_metadata_fields
    if not isinstance(configured, (list, tuple)):
        return ()
    fields: Final = tuple(str(field) for field in configured)
    return tuple(field for index, field in enumerate(fields) if field not in fields[:index])


def _metadata_sources(litellm_params: Mapping[str, object] | None) -> tuple[Mapping[str, object], ...]:
    """``metadata`` on /v1/chat/completions, ``litellm_metadata`` on the LITELLM_METADATA_ROUTES."""
    if litellm_params is None:
        return ()
    return tuple(
        source
        for name in _METADATA_PARAM_NAMES
        for source in (litellm_params.get(name),)
        if isinstance(source, Mapping)
    )


def _identity_pairs(
    sources: tuple[Mapping[str, object], ...],
    allowed_fields: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (field, value)
        for field in allowed_fields
        if field.startswith(BEDROCK_REQUEST_METADATA_IDENTITY_PREFIX)
        for value in (_first_text(sources, field),)
        if value is not None and _is_forwardable(field, value)
    )[:BEDROCK_REQUEST_METADATA_MAX_PAIRS]


def _first_text(sources: tuple[Mapping[str, object], ...], field: str) -> str | None:
    return next((value for source in sources if isinstance(value := source.get(field), str)), None)


def _client_pairs(
    sources: tuple[Mapping[str, object], ...],
    allowed_fields: tuple[str, ...],
    caller_metadata: object,
    budget: int,
) -> tuple[tuple[str, str], ...]:
    spend_logs_pairs: Final = (
        tuple(pair for source in sources for pair in _text_pairs(source.get(BEDROCK_REQUEST_METADATA_CLIENT_FIELD)))
        if BEDROCK_REQUEST_METADATA_CLIENT_FIELD in allowed_fields
        else ()
    )
    candidates: Final = tuple(
        (key, value)
        for key, value in (*_text_pairs(caller_metadata), *spend_logs_pairs)
        if not key.startswith(BEDROCK_REQUEST_METADATA_IDENTITY_PREFIX) and _is_forwardable(key, value)
    )
    return tuple(
        pair
        for index, pair in enumerate(candidates)
        if pair[0] not in tuple(earlier for earlier, _ in candidates[:index])
    )[:budget]


def resolve_bedrock_request_metadata(
    litellm_params: Mapping[str, object] | None,
    caller_metadata: object = None,
) -> dict[str, str] | None:
    """
    Resolve the ``requestMetadata`` pairs to send to Bedrock, or ``None`` when the feature is
    off or nothing survives Bedrock's constraints. The result is a plain dict because it is
    written straight onto the Converse body, which Bedrock types as ``dict[str, str]``.

    ``caller_metadata`` is any ``requestMetadata`` the caller passed explicitly. It has already
    been validated (and rejected with a 400) by the Converse transformation, so it is only
    filtered here for the reserved identity prefix and the remaining slot budget.
    """
    allowed_fields: Final = _allowed_fields()
    if not allowed_fields:
        return None
    sources: Final = _metadata_sources(litellm_params)
    identity: Final = _identity_pairs(sources, allowed_fields)
    client: Final = _client_pairs(
        sources=sources,
        allowed_fields=allowed_fields,
        caller_metadata=caller_metadata,
        budget=BEDROCK_REQUEST_METADATA_MAX_PAIRS - len(identity),
    )
    resolved: Final = {key: value for key, value in (*identity, *client)}
    return resolved or None


def bedrock_request_metadata_is_owned() -> bool:
    """
    Whether the proxy OWNS the request-metadata field and header name for this request.

    Ownership follows the operator's opt-in alone, never whether anything resolved, because a
    caller can suppress the resolver by omitting the allow-listed fields or by sending values
    that all fail Bedrock's rules. Owned-but-empty has to mean "absent on the wire" rather than
    "fall back to whatever the caller supplied", or the reserved-prefix guarantee is bypassable
    by anyone who can make the resolver produce nothing.
    """
    return bool(_allowed_fields())


def bedrock_request_metadata_headers(
    litellm_params: Mapping[str, object] | None,
) -> tuple[frozenset[str], tuple[tuple[str, str], ...]]:
    """
    The signed ``X-Amzn-Bedrock-Request-Metadata`` header for the Invoke paths, which have no
    body field for request metadata.

    Returns the header names the proxy OWNS and, separately, the pairs to send. Ownership is
    reported whenever forwarding is enabled, including when nothing resolves, because a caller
    can suppress the resolver (omit the allow-listed fields, or send values that all fail
    Bedrock's rules) and an owned-but-empty result must still evict the caller's header rather
    than fall back to it.
    """
    if not bedrock_request_metadata_is_owned():
        return frozenset(), ()
    resolved: Final = resolve_bedrock_request_metadata(litellm_params)
    if resolved is None:
        return _OWNED_HEADER_NAMES, ()
    return _OWNED_HEADER_NAMES, ((BEDROCK_REQUEST_METADATA_HEADER, json.dumps(resolved, separators=(",", ":"))),)


def merge_bedrock_invoke_headers(
    headers: dict[str, str],
    caller_owned: tuple[tuple[str, str], ...],
    proxy_owned: tuple[tuple[str, str], ...],
    proxy_owned_names: frozenset[str],
) -> dict[str, str]:
    """
    Merge the ``X-Amzn-*`` headers the Invoke paths derive from params.

    ``caller_owned`` (the guardrail headers) defers to a header the caller already set, which is
    the long-standing behaviour for those. ``proxy_owned_names`` are dropped from the caller's
    headers unconditionally and re-supplied only from ``proxy_owned``, because those names carry
    proxy-authenticated identity into an AWS billing record that the caller must not be able to
    write. Names are compared case-insensitively so a caller cannot leave a second spelling in
    the dict and let the transport pick the winner.
    """
    if not caller_owned and not proxy_owned and not proxy_owned_names:
        return headers
    existing_names: Final = frozenset(name.lower() for name in headers)
    return {
        name: value
        for name, value in (
            *((n, v) for n, v in headers.items() if n.lower() not in proxy_owned_names),
            *((n, v) for n, v in caller_owned if n.lower() not in existing_names),
            *proxy_owned,
        )
    }
