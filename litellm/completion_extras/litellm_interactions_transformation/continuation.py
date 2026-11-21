from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import chain
from typing import Final, TypeAlias

from pydantic import JsonValue

from litellm._logging import verbose_logger

JsonMapping: TypeAlias = Mapping[str, JsonValue]

HOP_END_STATUS: Final = "requires_continuation"
CUT_OFF_STATUSES: Final = frozenset({"incomplete", HOP_END_STATUS})
MAX_CONTINUATION_HOPS: Final = 32


def _token_of(interaction: JsonMapping) -> str | None:
    token: Final = interaction.get("continuation_token")
    return token if isinstance(token, str) and token else None


def client_continuation_token(interaction: JsonMapping) -> str | None:
    return _token_of(interaction) if interaction.get("status") in CUT_OFF_STATUSES else None


def next_hop_body(
    request_body: Mapping[str, object], interaction: JsonMapping, hops_so_far: int
) -> dict[str, object] | None:  # mutable-ok: JSON request body passed to HTTPHandler.post
    token: Final = _token_of(interaction)
    if interaction.get("status") != HOP_END_STATUS or token is None or hops_so_far >= MAX_CONTINUATION_HOPS:
        return None
    interaction_id: Final = interaction.get("id")
    return {
        **request_body,
        "continuation_token": token,
        **({"previous_interaction_id": interaction_id} if isinstance(interaction_id, str) and interaction_id else {}),
    }


def _summed(values: Sequence[JsonValue]) -> JsonValue:
    if all(isinstance(value, list) for value in values):
        return list(chain.from_iterable(value for value in values if isinstance(value, list)))
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return sum(value for value in values if isinstance(value, int))
    return values[-1]


def merge_hop_usage(usages: Sequence[JsonValue]) -> JsonValue:
    hop_usages: Final = tuple(usage for usage in usages if isinstance(usage, dict))
    if len(hop_usages) <= 1:
        return hop_usages[0] if hop_usages else None
    keys: Final = tuple(dict.fromkeys(chain.from_iterable(hop_usages)))
    summed: Final = {key: _summed(tuple(usage[key] for usage in hop_usages if key in usage)) for key in keys}
    first: Final = hop_usages[0]
    return {
        **summed,
        **({"initial_input_tokens": first["initial_input_tokens"]} if "initial_input_tokens" in first else {}),
    }


def merge_hops(hops: Sequence[JsonMapping]) -> JsonMapping:
    last: Final = hops[-1]
    if len(hops) == 1:
        return last
    hop_steps: Final = tuple(hop.get("steps") for hop in hops)
    steps: Final = list(chain.from_iterable(steps for steps in hop_steps if isinstance(steps, list)))
    return {**last, "steps": steps, "usage": merge_hop_usage(tuple(hop.get("usage") for hop in hops))}


def log_failed_hop(error: Exception | str) -> None:
    verbose_logger.warning(
        "Gemini Interactions long decode: a continuation hop failed, returning the answer so far with its continuation_token: %s",
        error,
    )
