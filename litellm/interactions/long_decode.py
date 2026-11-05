"""
Client-side continuation hops for Interactions API long decode.

A foreground Interactions request stops at Google's per-hop output cap with
``status: "requires_continuation"`` and a ``continuation_token``. Resending the
request with that token (and ``previous_interaction_id`` when the interaction is
stored) continues the same answer, and the hops merge into one interaction whose
``steps`` and ``usage`` cover the whole decode, the way Google's SDK does with
``automatic_continuation``.
"""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from itertools import chain
from types import MappingProxyType
from typing import Final, TypeAlias

import httpx
from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.exceptions import Timeout

InteractionBody: TypeAlias = Mapping[str, object]
LongDecodeHops: TypeAlias = tuple[InteractionBody, ...]
SyncHopPost: TypeAlias = Callable[[InteractionBody], httpx.Response]
AsyncHopPost: TypeAlias = Callable[[InteractionBody], Awaitable[httpx.Response]]

REQUIRES_CONTINUATION: Final = "requires_continuation"
HOP_FAILURES: Final = (httpx.HTTPError, Timeout, ValidationError, ValueError)

_BODY: Final = TypeAdapter(dict[str, object])
_USAGE: Final = TypeAdapter(dict[str, object])
_USAGE_ROWS: Final = TypeAdapter(list[dict[str, object]])
_STEPS: Final = TypeAdapter(list[object])
_INTERACTION: Final = TypeAdapter(dict[str, object])
_REQUEST_BODY: Final = TypeAdapter(dict[str, object])
_SUMMED_USAGE_KEYS: Final = (
    "total_input_tokens",
    "total_cached_tokens",
    "total_output_tokens",
    "total_tool_use_tokens",
    "total_reasoning_tokens",
    "total_thought_tokens",
    "total_tokens",
)
_ROW_USAGE_KEYS: Final[Mapping[str, tuple[str, str]]] = MappingProxyType(
    {
        "input_tokens_by_modality": ("modality", "tokens"),
        "cached_tokens_by_modality": ("modality", "tokens"),
        "output_tokens_by_modality": ("modality", "tokens"),
        "tool_use_tokens_by_modality": ("modality", "tokens"),
        "grounding_tool_count": ("type", "count"),
    }
)


def parse_body(response: httpx.Response) -> InteractionBody:
    return _BODY.validate_python(response.json())


def validated_request_body(data: object) -> Mapping[str, object]:
    """The provider request body as a typed mapping, so hop requests can be derived from it."""
    return _REQUEST_BODY.validate_python(data)


def interaction_of(chunk: Mapping[str, object]) -> InteractionBody | None:
    """The interaction a lifecycle stream event (interaction.created, interaction.completed) carries."""
    try:
        return _INTERACTION.validate_python(chunk.get("interaction")) if chunk.get("interaction") is not None else None
    except ValidationError:
        return None


def continuation_token_of(body: InteractionBody) -> str | None:
    token: Final = body.get("continuation_token")
    if body.get("status") != REQUIRES_CONTINUATION or not isinstance(token, str) or not token:
        return None
    return token


def hop_request(request_body: Mapping[str, object], hop: InteractionBody, token: str) -> InteractionBody:
    """The next hop keeps the first request and adds the token; a stored hop is chained by id instead of its input."""
    if request_body.get("store") is False or not isinstance(hop.get("id"), str):
        return {**request_body, "continuation_token": token}
    return {
        **{key: value for key, value in request_body.items() if key != "input"},
        "previous_interaction_id": hop["id"],
        "continuation_token": token,
    }


def _int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _usage_rows(value: object) -> Sequence[Mapping[str, object]]:
    try:
        return _USAGE_ROWS.validate_python(value) if value is not None else []
    except ValidationError:
        return []


def _sum_rows(rows: Sequence[Mapping[str, object]], group: str, count: str) -> Sequence[Mapping[str, object]]:
    """Rows keyed by a modality or a grounding tool type add up per key, e.g. two hops with 3 and 2 searches give 5."""
    groups: Final = tuple(dict.fromkeys(row.get(group) for row in rows))
    return [{group: key, count: sum(_int(row.get(count)) for row in rows if row.get(group) == key)} for key in groups]


def usage_of(hop: InteractionBody | None) -> Mapping[str, object] | None:
    if hop is None or hop.get("usage") is None:
        return None
    try:
        return _USAGE.validate_python(hop.get("usage"))
    except ValidationError:
        return None


def sum_usage(usages: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    """Every hop re-sends the answer so far, so per-hop counts add up; the first hop's prompt size is kept."""
    first: Final = usages[0]
    summed: Final = {
        key: sum(_int(usage.get(key)) for usage in usages)
        for key in _SUMMED_USAGE_KEYS
        if any(key in u for u in usages)
    }
    by_row: Final = {
        key: _sum_rows(tuple(chain.from_iterable(_usage_rows(usage.get(key)) for usage in usages)), group, count)
        for key, (group, count) in _ROW_USAGE_KEYS.items()
        if any(key in u for u in usages)
    }
    return {
        **first,
        **summed,
        **by_row,
        "initial_input_tokens": first.get("initial_input_tokens", first.get("total_input_tokens")),
    }


def _steps_of(hop: InteractionBody, key: str) -> Sequence[object]:
    try:
        return _STEPS.validate_python(hop.get(key)) if hop.get(key) is not None else []
    except ValidationError:
        return []


def merge_hops(hops: LongDecodeHops) -> InteractionBody:
    """One interaction for the whole decode: the last hop's id and status, every hop's steps, summed usage."""
    if len(hops) == 1:
        return hops[0]
    usages: Final = tuple(usage for usage in (usage_of(hop) for hop in hops) if usage is not None)
    merged_steps: Final = {
        key: list(chain.from_iterable(_steps_of(hop, key) for hop in hops))
        for key in ("steps", "outputs")
        if any(hop.get(key) is not None for hop in hops)
    }
    return {
        **hops[-1],
        **merged_steps,
        **({"usage": sum_usage(usages)} if usages else {}),
    }


def log_failed_hop(error: Exception | str) -> None:
    verbose_logger.warning(
        "Interactions long decode: a continuation hop failed, returning the answer so far with its continuation_token: %s",
        error,
    )


def _next_hop_token(hops: LongDecodeHops) -> str | None:
    token: Final = continuation_token_of(hops[-1])
    if token is None or any(token == hop.get("continuation_token") for hop in hops[:-1]):
        return None
    return token


def collect_hops_sync(first: InteractionBody, request_body: Mapping[str, object], post: SyncHopPost) -> LongDecodeHops:
    hops: LongDecodeHops = (first,)  # rebind-ok: grows by one hop per loop pass, CI bans the recursive form
    while (token := _next_hop_token(hops)) is not None:
        try:
            body = parse_body(post(hop_request(request_body, hops[-1], token)))
        except HOP_FAILURES as e:
            log_failed_hop(e)
            return hops
        hops = (*hops, body)
    return hops


async def collect_hops_async(
    first: InteractionBody, request_body: Mapping[str, object], post: AsyncHopPost
) -> LongDecodeHops:
    hops: LongDecodeHops = (first,)  # rebind-ok: grows by one hop per loop pass, CI bans the recursive form
    while (token := _next_hop_token(hops)) is not None:
        try:
            body = parse_body(await post(hop_request(request_body, hops[-1], token)))
        except HOP_FAILURES as e:
            log_failed_hop(e)
            return hops
        hops = (*hops, body)
    return hops


def merged_response(first: httpx.Response, body: Mapping[str, object]) -> httpx.Response:
    """The merged interaction in the first hop's envelope, so the provider's transform_response reads it unchanged."""
    headers: Final = {
        key: value for key, value in first.headers.items() if key.lower() not in {"content-length", "content-encoding"}
    }
    return httpx.Response(status_code=first.status_code, headers=headers, json=dict(body), request=first.request)
