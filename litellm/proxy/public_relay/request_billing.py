from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Protocol, cast  # noqa: TID251, RUF100  # LiteLLM metadata uses dynamic callback payloads.

import litellm
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.public_relay.config import PublicRelaySettings
from litellm.proxy.public_relay.db_types import PriceRow
from litellm.proxy.public_relay.repository import PUBLIC_ALLOWED_ROUTES, get_active_price, reserve_request
from litellm.proxy.utils import PrismaClient


class TokenCounter(Protocol):
    def __call__(
        self,
        *,
        model: str,
        text: str | None = None,
        messages: list[object] | None = None,
        tools: object = None,
        tool_choice: object = None,
    ) -> int: ...


def is_public_relay_key(user: UserAPIKeyAuth) -> bool:
    return _metadata(user).get("public_relay") is True


def enforce_public_route(user: UserAPIKeyAuth, route: str) -> None:
    if not is_public_relay_key(user):
        return
    normalized = route.split("?", 1)[0].rstrip("/") or "/"
    if normalized not in PUBLIC_ALLOWED_ROUTES:
        raise ProxyException(
            message="This API key is not allowed to access the requested route",
            type="invalid_request_error",
            param=None,
            code=403,
            openai_code="route_not_allowed",
        )


async def reserve_public_request(
    user: UserAPIKeyAuth,
    request_data: dict[str, object],
    route: str,
    prisma_client: PrismaClient | None,
) -> None:
    user.public_relay_reservation = None
    if not is_public_relay_key(user):
        return
    if route.rstrip("/").endswith("/models"):
        return
    value = PublicRelaySettings.from_env()
    if not value.enabled or prisma_client is None:
        raise ProxyException(
            message="Public relay is unavailable",
            type="service_unavailable",
            param=None,
            code=503,
        )
    account_id = _metadata(user).get("public_relay_account_id")
    model = request_data.get("model")
    if not isinstance(account_id, str) or not isinstance(model, str):
        raise _invalid_request("Public relay key or model is invalid")
    from litellm._uuid import uuid

    request_id = str(uuid.uuid4())
    request_data["litellm_call_id"] = request_id
    embedding = route.rstrip("/").endswith("/embeddings")
    input_tokens = _input_tokens(request_data, model)
    price = await get_active_price(prisma_client, model)
    if price is None:
        raise _invalid_request("model is not available on the public relay")
    output_tokens = 0 if embedding else _resolve_output_limit(request_data, route, price)
    try:
        result = await reserve_request(
            prisma_client,
            account_id,
            request_id,
            model,
            input_tokens,
            output_tokens,
            embedding,
            value.reservation_ttl_seconds,
            price,
        )
    except ArithmeticError as exc:
        raise ProxyException(
            message="Insufficient balance",
            type="insufficient_balance",
            param=None,
            code=402,
            openai_code="insufficient_balance",
        ) from exc
    except (LookupError, PermissionError, ValueError) as exc:
        raise _invalid_request(str(exc)) from exc
    user.public_relay_reservation = {
        "request_id": result.reservation.request_id,
        "account_id": result.reservation.account_id,
        "reservation_id": result.reservation.reservation_id,
        "embedding": embedding,
    }


def _resolve_output_limit(request_data: dict[str, object], route: str, price: PriceRow) -> int:
    for multiplier_key in ("n", "best_of"):
        multiplier = request_data.get(multiplier_key)
        if multiplier is not None and multiplier != 1:
            raise _invalid_request(f"{multiplier_key} must be 1 for public relay requests")
    output_keys = ("max_output_tokens", "max_completion_tokens", "max_tokens")
    requested = next((request_data.get(key) for key in output_keys if request_data.get(key) is not None), None)
    if requested is None:
        requested = price.default_max_output_tokens
        request_data["max_output_tokens" if route.rstrip("/").endswith("/responses") else "max_tokens"] = requested
    if isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0:
        raise _invalid_request("maximum output tokens must be a positive integer")
    if requested > price.max_output_tokens:
        raise _invalid_request("maximum output tokens exceeds the public model limit")
    return requested


def _input_tokens(request_data: Mapping[str, object], model: str) -> int:
    try:
        messages = request_data.get("messages")
        if isinstance(messages, list):
            return _token_counter()(
                model=model,
                messages=cast(  # cast-ok: isinstance validates the OpenAI messages array boundary.
                    list[object], messages
                ),
                tools=request_data.get("tools"),
                tool_choice=request_data.get("tool_choice"),
            )
        if "input" in request_data:
            return _count_value_tokens(request_data.get("input"), model)
        if "prompt" in request_data:
            return _count_value_tokens(request_data.get("prompt"), model)
    except Exception as exc:
        raise _invalid_request("Unable to count request input tokens") from exc
    return 0


def _count_value_tokens(value: object, model: str) -> int:
    if value is None:
        return 0
    if isinstance(value, list):
        return sum(
            _count_value_tokens(item, model)
            for item in cast(list[object], value)  # cast-ok: isinstance validates the request array boundary.
        )
    if isinstance(value, Mapping):
        return _token_counter()(model=model, text=json.dumps(value))
    return _token_counter()(model=model, text=str(value))


def _invalid_request(message: str) -> ProxyException:
    return ProxyException(
        message=message,
        type="invalid_request_error",
        param=None,
        code=400,
    )


def _metadata(user: UserAPIKeyAuth) -> dict[str, object]:
    return cast(dict[str, object], user.metadata)  # cast-ok: isinstance validates virtual-key metadata.


def _token_counter() -> TokenCounter:
    return cast(TokenCounter, litellm.token_counter)  # cast-ok: LiteLLM exports the token-counter protocol.
