from collections.abc import Collection, Mapping, Sequence
from itertools import chain
from types import MappingProxyType
from typing import Final


def _clean_followup_value(key: str, val: object) -> object:
    if key == "metadata" and isinstance(val, Mapping):
        return {k: v for k, v in val.items() if k != "_otel_internal"}
    if key == "litellm_params" and isinstance(val, Mapping):
        clean_params = dict(val)
        if "metadata" in clean_params and isinstance(clean_params["metadata"], Mapping):
            clean_params["metadata"] = {k: v for k, v in clean_params["metadata"].items() if k != "_otel_internal"}
        return clean_params
    return val


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
            key: _clean_followup_value(key, value)
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
