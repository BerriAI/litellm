from datetime import datetime, timezone

import pytest

from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.public_relay.db_types import PriceRow, ReservationRow
from litellm.proxy.public_relay.repository import ReservationResult
from litellm.proxy.public_relay.request_billing import (
    _resolve_output_limit,
    enforce_public_route,
    reserve_public_request,
)


def _public_user() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="hashed-key",
        metadata={"public_relay": True, "public_relay_account_id": "account-1"},
    )


def _price() -> PriceRow:
    return PriceRow(
        price_id="price-1",
        model_name="relay-model",
        version=1,
        input_micros_per_million=1,
        cached_input_micros_per_million=None,
        output_micros_per_million=1,
        embedding_micros_per_million=None,
        default_max_output_tokens=4096,
        max_output_tokens=8192,
        enabled=True,
        effective_at=datetime.now(timezone.utc),
    )


@pytest.mark.parametrize(
    "route",
    ["/v1/models", "/v1/chat/completions", "/v1/responses", "/v1/embeddings"],
)
def test_public_routes_are_allowed(route: str) -> None:
    enforce_public_route(_public_user(), route)


def test_non_public_route_is_rejected() -> None:
    with pytest.raises(ProxyException) as exc_info:
        enforce_public_route(_public_user(), "/v1/images/generations")

    assert exc_info.value.code == "403"
    assert exc_info.value.openai_code == "route_not_allowed"


def test_default_output_limit_is_injected() -> None:
    request_data: dict = {}

    result = _resolve_output_limit(request_data, "/v1/chat/completions", _price())

    assert result == 4096
    assert request_data["max_tokens"] == 4096


def test_output_over_public_limit_is_rejected() -> None:
    with pytest.raises(ProxyException):
        _resolve_output_limit({"max_tokens": 8193}, "/v1/chat/completions", _price())


def test_multiple_choices_are_rejected() -> None:
    with pytest.raises(ProxyException):
        _resolve_output_limit({"n": 2}, "/v1/chat/completions", _price())


@pytest.mark.asyncio
async def test_public_request_id_is_server_generated(monkeypatch) -> None:
    captured_request_ids: list[str] = []

    async def active_price(_prisma, _model: str) -> PriceRow:
        return _price()

    async def reserve(_prisma, _account_id: str, request_id: str, *_args, **_kwargs) -> ReservationResult:
        captured_request_ids.append(request_id)
        return ReservationResult(
            reservation=ReservationRow(
                reservation_id="reservation-1",
                request_id=request_id,
                account_id="account-1",
                wallet_id="wallet-1",
                price_id="price-1",
                reserved_micros=1,
                input_tokens=0,
                max_output_tokens=4096,
                status="OPEN",
            ),
            price=_price(),
        )

    monkeypatch.setenv("PUBLIC_RELAY_ENABLED", "true")
    monkeypatch.setattr("litellm.proxy.public_relay.request_billing.get_active_price", active_price)
    monkeypatch.setattr("litellm.proxy.public_relay.request_billing.reserve_request", reserve)
    request_data: dict[str, object] = {"model": "relay-model", "litellm_call_id": "client-selected"}

    await reserve_public_request(_public_user(), request_data, "/v1/chat/completions", object())

    assert captured_request_ids == [request_data["litellm_call_id"]]
    assert request_data["litellm_call_id"] != "client-selected"
