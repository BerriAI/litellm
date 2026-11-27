from collections.abc import Collection, Mapping, Sequence
from itertools import chain
from types import MappingProxyType
from typing import Final

from litellm.litellm_core_utils.core_helpers import as_str_mapping


def _isolated_followup_value(key: str, value: object) -> object:
    # Copy metadata dicts without OTel's per-request dedupe state so a nested
    # follow-up never mutates or suppresses the parent call's spans.
    if key in ("metadata", "litellm_metadata") and (mapping := as_str_mapping(value)) is not None:
        return {k: v for k, v in mapping.items() if k != "_otel_internal"}
    return value


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
            key: _isolated_followup_value(key, value)
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
