from collections.abc import Collection, Mapping, Sequence
from itertools import chain
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter, ValidationError

_ISOLATED_METADATA_KEYS: Final = frozenset(("metadata", "litellm_metadata"))
_MAPPING_ADAPTER: Final[TypeAdapter[Mapping[object, object]]] = TypeAdapter(Mapping[object, object])


def isolate_agentic_metadata(value: object) -> object:
    """Return a shallow dict copy of metadata without per-request ``_otel_internal`` state."""
    raw: Final[object] = value
    if not isinstance(value, Mapping):
        return raw
    try:
        mapped: Final[Mapping[object, object]] = _MAPPING_ADAPTER.validate_python(value)
    except ValidationError:
        return raw
    return {k: v for k, v in mapped.items() if k != "_otel_internal"}


def build_agentic_followup_kwargs(
    *,
    request_kwargs: Mapping[str, object],
    patch_kwargs: Mapping[str, object],
    request_params: Collection[str],
    depth: int,
    max_loops: int,
    fingerprints: Sequence[str],
    fingerprint: str,
) -> Mapping[str, object]:
    """Kwargs for an agentic follow-up call: the request's kwargs overlaid by the plan's, never repeating a key already sent as a request param"""
    seen: Final = [*fingerprints, fingerprint]
    return MappingProxyType(
        {
            key: isolate_agentic_metadata(value) if key in _ISOLATED_METADATA_KEYS else value
            for key, value in chain(
                ((k, v) for k, v in request_kwargs.items() if k not in request_params),
                ((k, v) for k, v in patch_kwargs.items() if k not in request_params),
                (
                    ("_agentic_loop_depth", depth + 1),
                    ("max_agentic_loops", max_loops),
                    ("_agentic_loop_fingerprints", seen),
                ),
            )
        }
    )
