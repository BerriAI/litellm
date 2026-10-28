"""
Vertex AI forwards request ``labels`` to Cloud Billing, so proxy identity becomes labels only when the operator sets
``litellm.vertex_request_label_fields``. Identity resolves first and the ``user_api_key_`` prefix is reserved, so a
caller can neither evict nor forge it. Identity and ``spend_logs_metadata`` text is lowercased, every character Google's
label rules disallow becomes ``_``, and it is cut to 63 characters, so an alias like ``AI Platform`` still bills under a
compliant ``ai_platform`` label. Caller labels pass through verbatim, as they do with the setting off.
"""

from __future__ import annotations

import base64
import json
import unicodedata
from collections.abc import Mapping
from itertools import chain
from typing import Final, TypeAlias

from pydantic import TypeAdapter

import litellm

VERTEX_REQUEST_LABELS_MAX: Final = 64
VERTEX_PARTNER_REQUEST_LABELS_MAX: Final = 32
VERTEX_REQUEST_LABEL_MAX_CHARS: Final = 63
VERTEX_REQUEST_LABEL_IDENTITY_PREFIX: Final = "user_api_key_"
VERTEX_REQUEST_LABEL_CLIENT_FIELD: Final = "spend_logs_metadata"
VERTEX_LABELS_HEADER: Final = "X-Vertex-AI-Labels"

_METADATA_PARAM_NAMES: Final = ("metadata", "litellm_metadata")
_KEY_START_CATEGORIES: Final = frozenset(("Ll", "Lo"))
_LABEL_CHARACTER_CATEGORIES: Final = frozenset(("Ll", "Lo", "Nd", "Nl", "No"))
_ANY_MAPPING: Final = TypeAdapter(Mapping[object, object])
_ANY_SEQUENCE: Final = TypeAdapter(tuple[object, ...])

Label: TypeAlias = tuple[str, str]


def vertex_request_label_fields() -> tuple[str, ...]:
    configured: Final[object] = litellm.vertex_request_label_fields
    if not isinstance(configured, (list, tuple)):
        return ()
    return tuple(field for field in _ANY_SEQUENCE.validate_python(configured) if isinstance(field, str))


def _is_label_character(character: str) -> bool:
    return character in "_-" or unicodedata.category(character) in _LABEL_CHARACTER_CATEGORIES


def _normalized(text: str) -> str:
    compliant: Final = (character if _is_label_character(character) else "_" for character in text.lower())
    return "".join(compliant)[:VERTEX_REQUEST_LABEL_MAX_CHARS]


def _normalized_label(key: str, value: str) -> Label | None:
    normalized_key: Final = _normalized(key)
    if not normalized_key or unicodedata.category(normalized_key[0]) not in _KEY_START_CATEGORIES:
        return None
    return normalized_key, _normalized(value)


def _is_reserved(key: str) -> bool:
    return key.strip().lower().startswith(VERTEX_REQUEST_LABEL_IDENTITY_PREFIX)


def _metadata_sources(litellm_params: Mapping[str, object]) -> tuple[Mapping[object, object], ...]:
    candidates: Final = (litellm_params.get(name) for name in _METADATA_PARAM_NAMES)
    return tuple(_ANY_MAPPING.validate_python(source) for source in candidates if isinstance(source, Mapping))


def _text_labels(source: object) -> tuple[Label, ...]:
    if not isinstance(source, Mapping):
        return ()
    pairs: Final = _ANY_MAPPING.validate_python(source).items()
    return tuple((key, value) for key, value in pairs if isinstance(key, str) and isinstance(value, str))


def _first_text(sources: tuple[Mapping[object, object], ...], field: str) -> str | None:
    return next((value for source in sources if isinstance(value := source.get(field), str)), None)


def _identity_label(sources: tuple[Mapping[object, object], ...], field: str) -> Label | None:
    value: Final = _first_text(sources, field)
    if value is None or not _is_reserved(field):
        return None
    return _normalized_label(field, value)


def _identity_labels(
    sources: tuple[Mapping[object, object], ...], allowed_fields: tuple[str, ...]
) -> tuple[Label, ...]:
    candidates: Final = (_identity_label(sources, field) for field in allowed_fields)
    return tuple(label for label in candidates if label is not None)


def _spend_logs_labels(sources: tuple[Mapping[object, object], ...]) -> tuple[Label, ...]:
    pairs: Final = chain.from_iterable(
        _text_labels(source.get(VERTEX_REQUEST_LABEL_CLIENT_FIELD)) for source in sources
    )
    candidates: Final = (_normalized_label(key, value) for key, value in pairs)
    return tuple(label for label in candidates if label is not None)


def _first_occurrences(labels: tuple[Label, ...], max_labels: int) -> tuple[Label, ...]:
    first_value: Final = dict(reversed(labels))
    keys: Final = tuple(dict.fromkeys(key for key, _ in labels))
    return tuple((key, first_value[key]) for key in keys[:max_labels])


def resolve_vertex_request_labels(
    allowed_fields: tuple[str, ...],
    caller_labels: object,
    litellm_params: Mapping[str, object],
    max_labels: int,
) -> tuple[Label, ...]:
    """
    The labels to send for a Vertex request once ``allowed_fields`` (``vertex_request_label_fields()``) is non-empty.
    Identity comes first, then the caller's string labels minus the reserved prefix, then ``spend_logs_metadata``
    pairs when allowed, each key keeping its first value, up to ``max_labels``.
    """
    sources: Final = _metadata_sources(litellm_params)
    spend_logs: Final = _spend_logs_labels(sources) if VERTEX_REQUEST_LABEL_CLIENT_FIELD in allowed_fields else ()
    client: Final = tuple(label for label in (*_text_labels(caller_labels), *spend_logs) if not _is_reserved(label[0]))
    return _first_occurrences((*_identity_labels(sources, allowed_fields), *client), max_labels)


def is_vertex_labels_header(name: str) -> bool:
    return name.lower() == VERTEX_LABELS_HEADER.lower()


def _decoded_labels_header(encoded: object) -> object:
    if not isinstance(encoded, str):
        return None
    try:
        return _ANY_MAPPING.validate_json(base64.b64decode(encoded, validate=True))
    except ValueError:
        return None


def vertex_partner_labels_header(
    allowed_fields: tuple[str, ...],
    caller_labels: object,
    outbound_headers: Mapping[str, object],
    litellm_params: Mapping[str, object],
) -> str | None:
    """
    The ``X-Vertex-AI-Labels`` value for a partner model call once ``allowed_fields`` is non-empty. A labels header the
    caller already set stands in for ``caller_labels``, as it does with the setting off, and the result replaces it.
    """
    caller_header: Final = next(
        (value for name, value in outbound_headers.items() if is_vertex_labels_header(name)), None
    )
    resolved: Final = resolve_vertex_request_labels(
        allowed_fields,
        caller_labels if caller_header is None else _decoded_labels_header(caller_header),
        litellm_params,
        VERTEX_PARTNER_REQUEST_LABELS_MAX,
    )
    return base64.b64encode(json.dumps(dict(resolved)).encode()).decode() if resolved else None
