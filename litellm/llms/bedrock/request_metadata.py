"""
Resolve AWS Bedrock ``requestMetadata`` from LiteLLM proxy identity and caller metadata.

Bedrock attaches request metadata to CloudTrail records and to the dimension AWS Cost
Explorer groups on, so everything here is opt-in: nothing is forwarded unless the operator
sets ``litellm.bedrock_request_metadata_fields`` (``litellm_settings`` on the proxy).

Two properties are load-bearing for that billing record and are asserted by the tests:
proxy identity is resolved first so it can never be evicted by caller-supplied pairs, and the
whole ``user_api_key_`` prefix is reserved so a caller cannot write a proxy-authoritative
looking key. Identity and ``spend_logs_metadata`` values that break Bedrock's constraints are
dropped rather than sanitised or rejected, because an operator flipping this setting on must not
turn a working request into a 400 and a silently rewritten attribution key is worse than an
absent one. For the same reason an Invoke header that would pass Bedrock's 8,500-character limit
keeps only the leading pairs that fit. A caller's own malformed ``requestMetadata`` is rejected with
a 400 unless ``drop_params`` is set, in which case its bad pairs are dropped the same way, with or
without the setting.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sized
from dataclasses import dataclass
from typing import Final, Literal

import litellm

BEDROCK_REQUEST_METADATA_HEADER: Final = "X-Amzn-Bedrock-Request-Metadata"
BEDROCK_REQUEST_METADATA_MAX_PAIRS: Final = 16
BEDROCK_REQUEST_METADATA_IDENTITY_PREFIX: Final = "user_api_key_"
BEDROCK_REQUEST_METADATA_CLIENT_FIELD: Final = "spend_logs_metadata"
BEDROCK_REQUEST_METADATA_HEADER_MAX_CHARS: Final = 8500

_METADATA_PARAM_NAMES: Final[tuple[str, ...]] = ("metadata", "litellm_metadata")
_ALLOWED_CHARACTERS: Final = r"[a-zA-Z0-9\s:_@$#=/+,.-]"
_KEY_PATTERN: Final = re.compile(rf"{_ALLOWED_CHARACTERS}{{1,256}}", re.ASCII)
_VALUE_PATTERN: Final = re.compile(rf"{_ALLOWED_CHARACTERS}{{0,256}}", re.ASCII)
_OWNED_HEADER_NAMES: Final[frozenset[str]] = frozenset((BEDROCK_REQUEST_METADATA_HEADER.lower(),))


@dataclass(frozen=True, slots=True)
class BedrockRequestMetadataViolation:
    kind: Literal["type", "constraint"]
    message: str


def _is_forwardable(key: str, value: str) -> bool:
    return _KEY_PATTERN.fullmatch(key) is not None and _VALUE_PATTERN.fullmatch(value) is not None


def _is_reserved(key: str) -> bool:
    return key.strip().lower().startswith(BEDROCK_REQUEST_METADATA_IDENTITY_PREFIX)


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
        if not _is_reserved(key) and _is_forwardable(key, value)
    )
    first_index: Final = {key: index for index, (key, _) in reversed(tuple(enumerate(candidates)))}
    return tuple(pair for index, pair in enumerate(candidates) if first_index[pair[0]] == index)[:budget]


def _forwardable_caller_pairs(caller_metadata: object) -> tuple[tuple[str, str], ...]:
    return tuple(pair for pair in _text_pairs(caller_metadata) if _is_forwardable(*pair))[
        :BEDROCK_REQUEST_METADATA_MAX_PAIRS
    ]


def resolve_bedrock_request_metadata(
    litellm_params: Mapping[str, object] | None,
    caller_metadata: object = None,
) -> dict[str, str] | None:
    """
    Resolve the ``requestMetadata`` pairs to send to Bedrock, or ``None`` when nothing survives
    Bedrock's constraints. The result is a plain dict because it is written straight onto the
    Converse body, which Bedrock types as ``dict[str, str]``.

    ``caller_metadata`` is any ``requestMetadata`` the caller passed explicitly. Callers reject a
    malformed one with a 400 unless ``drop_params`` is set, so here its pairs are only filtered for
    Bedrock's rules, the reserved identity prefix, and the remaining slot budget. With the feature
    off only Bedrock's rules apply, since no identity is written for a caller to forge.
    """
    allowed_fields: Final = _allowed_fields()
    if not allowed_fields:
        return dict(_forwardable_caller_pairs(caller_metadata)) or None
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


def _pair_violation(key: str, value: str) -> BedrockRequestMetadataViolation | None:
    if _KEY_PATTERN.fullmatch(key) is None:
        return BedrockRequestMetadataViolation(
            "constraint", f"requestMetadata key {key!r} must be 1-256 characters of {_ALLOWED_CHARACTERS}"
        )
    if _VALUE_PATTERN.fullmatch(value) is None:
        return BedrockRequestMetadataViolation(
            "constraint", f"requestMetadata value for key {key!r} must be 0-256 characters of {_ALLOWED_CHARACTERS}"
        )
    return None


def bedrock_request_metadata_violation(caller_metadata: object) -> BedrockRequestMetadataViolation | None:
    """Why a caller's ``requestMetadata`` breaks Bedrock's rules, in the order Bedrock checks them: a
    wrong JSON type before the pair count and character rules. ``None`` means absent or valid."""
    text_pairs: Final = _text_pairs(caller_metadata)
    entry_count: Final = len(caller_metadata) if isinstance(caller_metadata, Sized) else 0
    if caller_metadata is None:
        return None
    if not isinstance(caller_metadata, Mapping):
        return BedrockRequestMetadataViolation("type", "requestMetadata must be a dictionary")
    if len(text_pairs) != entry_count:
        return BedrockRequestMetadataViolation("type", "requestMetadata keys and values must be strings")
    if len(text_pairs) > BEDROCK_REQUEST_METADATA_MAX_PAIRS:
        return BedrockRequestMetadataViolation(
            "constraint", f"requestMetadata can contain a maximum of {BEDROCK_REQUEST_METADATA_MAX_PAIRS} items"
        )
    return next((violation for key, value in text_pairs if (violation := _pair_violation(key, value))), None)


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


def caller_request_metadata_violations_are_dropped(drop_params: bool) -> bool:
    return drop_params or litellm.drop_params is True


def bedrock_request_metadata_is_rewritten(drop_params: bool) -> bool:
    return bedrock_request_metadata_is_owned() or caller_request_metadata_violations_are_dropped(drop_params)


def _header_value(pairs: Mapping[str, str]) -> str:
    return json.dumps(pairs, separators=(",", ":"))


def _fits_header(pairs: Mapping[str, str]) -> bool:
    return len(_header_value(pairs)) <= BEDROCK_REQUEST_METADATA_HEADER_MAX_CHARS


def _longest_fitting_prefix(resolved: Mapping[str, str]) -> dict[str, str]:
    pairs: Final = tuple(resolved.items())
    return next(
        (dict(pairs[:count]) for count in range(len(pairs), 0, -1) if _fits_header(dict(pairs[:count]))),
        {},
    )


def bedrock_request_metadata_headers(
    litellm_params: Mapping[str, object] | None,
) -> tuple[frozenset[str], tuple[tuple[str, str], ...]]:
    """
    The signed ``X-Amzn-Bedrock-Request-Metadata`` header for an Invoke request whose body carries
    no ``requestMetadata`` of its own.

    Returns the header names the proxy OWNS and, separately, the pairs to send. Ownership is
    reported whenever forwarding is enabled, including when nothing resolves, because a caller
    can suppress the resolver (omit the allow-listed fields, or send values that all fail
    Bedrock's rules) and an owned-but-empty result must still evict the caller's header rather
    than fall back to it. Only the leading pairs that fit Bedrock's header limit are sent, so
    identity always survives and long ``spend_logs_metadata`` pairs drop off the end.
    """
    if not bedrock_request_metadata_is_owned():
        return frozenset(), ()
    resolved: Final = resolve_bedrock_request_metadata(litellm_params)
    fitting: Final = {} if resolved is None else _longest_fitting_prefix(resolved)
    if not fitting:
        return _OWNED_HEADER_NAMES, ()
    return _OWNED_HEADER_NAMES, ((BEDROCK_REQUEST_METADATA_HEADER, _header_value(fitting)),)


def merge_bedrock_invoke_headers(
    headers: dict[str, str],
    caller_owned: tuple[tuple[str, str], ...],
    proxy_owned: tuple[tuple[str, str], ...],
    proxy_owned_names: frozenset[str],
) -> dict[str, str]:
    """
    Merge the ``X-Amzn-*`` headers the Invoke paths derive from params.

    ``caller_owned`` defers to a header the caller already set. ``proxy_owned_names`` are dropped
    from the caller's headers unconditionally and re-supplied only from ``proxy_owned``, for names
    the caller must not be able to write, such as proxy-authenticated identity headed for an AWS
    billing record or a deployment's guardrail. Names are compared case-insensitively so a caller
    cannot leave a second spelling in the dict and let the transport pick the winner.
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
