# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Numbers a router deployment cannot be served with, refused the same way by both router cores.

A whole number wider than 64 bits (a `ptu_shares` of `1e20`) does not fit the router's JSON
deployment record, and an infinite or NaN cost would read back as null there, so the call
would silently be priced from the model cost map. Both are refused when the deployment is
built or added, with the error naming the field.
"""

import math
import re
from collections.abc import Iterator, Mapping
from typing import Final

from litellm.litellm_core_utils.core_helpers import as_str_mapping

# The whole numbers a deployment record holds (serde_json's i64 and u64).
SMALLEST_NUMBER: Final = -(2**63)
LARGEST_NUMBER: Final = 2**64 - 1
# A pricing field: `input_cost_per_token`, `cache_read_input_token_cost`, ...
_COST_FIELD: Final = re.compile(r"cost_per|_cost$")


def _reads_as_non_finite(value: object) -> bool:
    """Whether a cost, as written, is an infinity or a NaN (``float()`` reads ``"inf"`` and ``"1e400"``)."""
    if isinstance(value, float):
        return not math.isfinite(value)
    if not isinstance(value, str):
        return False
    try:
        return not math.isfinite(float(value))
    except ValueError:
        return False


def _is_cost(key: object) -> bool:
    return isinstance(key, str) and _COST_FIELD.search(key) is not None


def _number_leaves(value: object) -> Iterator[tuple[str, object, bool]]:
    """Every value under ``value`` that is no container, in order: its path, and whether it sits in a
    cost field. Each container is walked once, so an entry that holds itself ends."""
    pending: Final[list[tuple[object, str, bool]]] = [(value, "", False)]  # mutable-ok: the walk's work list
    walked: Final[set[int]] = set()  # mutable-ok: the ids walked so far
    while pending:
        current, path, cost = pending.pop()
        if not isinstance(current, Mapping | list | tuple):
            yield path, current, cost
        elif id(current) not in walked:
            walked.add(id(current))
            pending.extend(
                reversed(
                    [(item, f"{path}.{key}".removeprefix("."), cost or _is_cost(key)) for key, item in current.items()]
                    if isinstance(current, Mapping)
                    else [(item, f"{path}[{index}]", cost) for index, item in enumerate(current)]
                )
            )


def _refusal(entry: Mapping[str, object], reason: str) -> ValueError:
    return ValueError(f"Deployment {entry.get('model_name')!r} cannot be served: {reason}")


def refuse_wide_numbers(entry: object) -> None:
    """Refuse a deployment holding a whole number wider than 64 bits, by name."""
    deployment: Final = as_str_mapping(entry)
    if deployment is None:
        return
    for path, value, _ in _number_leaves(deployment):
        if isinstance(value, int) and not SMALLEST_NUMBER <= value <= LARGEST_NUMBER:
            raise _refusal(
                deployment,
                f"{path} is {value}, outside what the router can hold "
                f"({SMALLEST_NUMBER} to {LARGEST_NUMBER}). Use a smaller value.",
            )


def refuse_unservable_numbers(entry: object) -> None:
    """``refuse_wide_numbers``, and a deployment with an infinite or NaN cost."""
    refuse_wide_numbers(entry)
    deployment: Final = as_str_mapping(entry)
    if deployment is None:
        return
    for path, value, cost in _number_leaves(deployment):
        if cost and _reads_as_non_finite(value):
            raise _refusal(
                deployment,
                f"{path} is {value!r}, and the router only prices with finite costs. "
                "Set a finite cost, or remove it to use the model cost map's price.",
            )
