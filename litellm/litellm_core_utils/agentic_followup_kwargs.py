from collections.abc import Collection, Mapping, Sequence
from itertools import chain
from types import MappingProxyType
from typing import Final


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
    seen: Final = [*fingerprints, fingerprint]  # mutable-ok: the chat loop's settings reader only accepts a list
    return MappingProxyType(
        {
            key: value
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
