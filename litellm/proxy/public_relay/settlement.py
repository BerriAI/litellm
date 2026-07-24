from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import cast  # noqa: TID251, RUF100  # LiteLLM callbacks expose dynamic metadata payloads.

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.public_relay.money import UsageQuantity
from litellm.proxy.public_relay.repository import release_request, settle_request


async def settle_success(kwargs: dict[str, object], completion_response: object, upstream_cost: float) -> None:
    reservation = _reservation_from_kwargs(kwargs)
    if reservation is None:
        return
    usage = _usage(kwargs, completion_response, reservation.get("embedding") is True)
    if usage is None:
        await _release(reservation)
        return
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return
    await settle_request(
        prisma_client,
        _request_id(reservation),
        usage,
        _cost_micros(upstream_cost),
    )


async def settle_failure(
    request_data: dict[str, object],
    user: UserAPIKeyAuth,
    upstream_cost: float = 0.0,
) -> None:
    reservation = _object_dict(user.public_relay_reservation)
    if reservation is None:
        return
    usage = _usage(request_data, None, reservation.get("embedding") is True)
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return
    if usage is None or (usage.input_tokens == 0 and usage.output_tokens == 0):
        await release_request(prisma_client, _request_id(reservation))
        return
    await settle_request(
        prisma_client,
        _request_id(reservation),
        usage,
        _cost_micros(upstream_cost),
    )


def _reservation_from_kwargs(kwargs: dict[str, object]) -> dict[str, object] | None:
    params = _object_dict(kwargs.get("litellm_params"))
    metadata = _object_dict(params.get("metadata")) if params is not None else None
    return _object_dict(metadata.get("public_relay_reservation")) if metadata is not None else None


async def _release(reservation: dict[str, object]) -> None:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is not None:
        await release_request(prisma_client, _request_id(reservation))


def _usage(kwargs: dict[str, object], completion_response: object, embedding: bool) -> UsageQuantity | None:
    usage = kwargs.get("combined_usage_object")
    if not isinstance(usage, litellm.Usage):
        candidate = getattr(completion_response, "usage", None)
        usage = candidate if isinstance(candidate, litellm.Usage) else None
    if usage is None:
        standard = _object_dict(kwargs.get("standard_logging_object"))
        if standard is None:
            return None
        input_tokens = standard.get("prompt_tokens")
        output_tokens = standard.get("completion_tokens")
        if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
            return None
        return UsageQuantity(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            embedding=embedding,
        )
    cached_tokens = 0
    prompt_details = usage.prompt_tokens_details
    if prompt_details is not None and isinstance(prompt_details.cached_tokens, int):
        cached_tokens = prompt_details.cached_tokens
    return UsageQuantity(
        input_tokens=usage.prompt_tokens,
        cached_input_tokens=cached_tokens,
        output_tokens=usage.completion_tokens,
        embedding=embedding,
    )


def _request_id(reservation: dict[str, object]) -> str:
    value = reservation.get("request_id")
    if not isinstance(value, str):
        raise TypeError("public relay reservation request ID is missing")
    return value


def _cost_micros(cost: float) -> int:
    value = Decimal(str(max(cost, 0.0))) * Decimal(1_000_000)
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _object_dict(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    return cast(dict[str, object], value)  # cast-ok: isinstance validates the callback metadata boundary.
