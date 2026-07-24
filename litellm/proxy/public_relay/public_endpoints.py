from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast  # noqa: TID251, RUF100  # Stripe JSON requires explicit runtime boundary narrowing.

import stripe
from fastapi import APIRouter, Header, HTTPException, Request, status

from litellm.proxy.public_relay.api_types import (
    MessageResponse,
    ModelPriceResponse,
    PricingResponse,
    StatusResponse,
)
from litellm.proxy.public_relay.config import PublicRelaySettings
from litellm.proxy.public_relay.db_types import PriceRow
from litellm.proxy.public_relay.repository import (
    apply_dispute,
    attach_refund_submission,
    complete_refund,
    credit_checkout,
    database_handle,
    fail_checkout,
    fail_refund,
    get_refund_by_id,
    get_refund_by_stripe_id,
    list_active_prices,
    record_stripe_event,
)
from litellm.proxy.public_relay.runtime import database, relay_cache, settings
from litellm.proxy.public_relay.stripe_client import parse_webhook

router = APIRouter(tags=["public relay"])


@router.get("/v1/public/status", response_model=StatusResponse)
async def relay_status() -> StatusResponse:
    value = PublicRelaySettings.from_env()
    operational = value.enabled and not value.missing_runtime_configuration()
    if operational:
        try:
            relay_cache(value)
            await database_handle(database()).query_raw("SELECT 1")
        except Exception:  # noqa: BLE001  # Health status must collapse dependency failures to unavailable.
            operational = False
    return StatusResponse(enabled=value.enabled, operational=operational)


@router.get("/v1/public/pricing", response_model=PricingResponse)
async def pricing() -> PricingResponse:
    settings()
    values = await list_active_prices(database())
    return PricingResponse(models=tuple(_price_response(value) for value in values))


@router.post("/v1/public/payments/stripe/webhook", response_model=MessageResponse)
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(alias="stripe-signature"),
) -> MessageResponse:
    value = settings()
    payload = await request.body()
    try:
        parse_webhook(payload, stripe_signature, value.stripe_webhook_secret or "")
    except (ValueError, stripe.SignatureVerificationError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Stripe signature") from exc
    raw_event = _mapping(cast(object, json.loads(payload)))  # cast-ok: _mapping validates parsed webhook JSON.
    event_id = _string(raw_event, "id")
    event_type = _string(raw_event, "type")
    livemode = _boolean(raw_event, "livemode")
    event_object = _mapping(_mapping(raw_event.get("data")).get("object"))
    raw_payload = payload.decode()
    if event_type in {"checkout.session.completed", "checkout.session.async_payment_succeeded"}:
        await _process_paid_checkout(event_id, event_type, livemode, raw_payload, event_object)
    elif event_type in {"checkout.session.expired", "checkout.session.async_payment_failed"}:
        await fail_checkout(
            database(),
            event_id,
            event_type,
            livemode,
            raw_payload,
            _string(event_object, "id"),
        )
    elif event_type == "charge.dispute.created":
        await _process_dispute(event_id, event_type, livemode, raw_payload, event_object)
    elif event_type == "refund.updated":
        await _process_refund_update(event_id, event_type, livemode, raw_payload, event_object)
    else:
        await record_stripe_event(database(), event_id, event_type, livemode, raw_payload)
    return MessageResponse(message="accepted")


async def _process_paid_checkout(
    event_id: str,
    event_type: str,
    livemode: bool,
    payload: str,
    event_object: Mapping[str, object],
) -> None:
    if _string(event_object, "payment_status") != "paid":
        await record_stripe_event(database(), event_id, event_type, livemode, payload)
        return
    metadata = _mapping(event_object.get("metadata"))
    amount_total = _integer(event_object, "amount_total")
    await credit_checkout(
        database(),
        event_id,
        event_type,
        livemode,
        payload,
        _string(event_object, "id"),
        _expandable_id(event_object.get("payment_intent")),
        amount_total * 10_000,
        _string(metadata, "public_relay_account_id"),
        _string(event_object, "currency"),
        _string(metadata, "public_relay_payment_id"),
    )


async def _process_dispute(
    event_id: str,
    event_type: str,
    livemode: bool,
    payload: str,
    event_object: Mapping[str, object],
) -> None:
    currency = _string(event_object, "currency")
    if currency.lower() != "usd":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported dispute currency")
    await apply_dispute(
        database(),
        event_id,
        event_type,
        livemode,
        payload,
        _expandable_id(event_object.get("payment_intent")),
        _integer(event_object, "amount") * 10_000,
    )


async def _process_refund_update(
    event_id: str,
    event_type: str,
    livemode: bool,
    payload: str,
    event_object: Mapping[str, object],
) -> None:
    stripe_refund_id = _string(event_object, "id")
    refund = await get_refund_by_stripe_id(database(), stripe_refund_id)
    if refund is None:
        metadata = _mapping(event_object.get("metadata"))
        local_refund_id = metadata.get("public_relay_refund_id")
        if not isinstance(local_refund_id, str):
            await record_stripe_event(database(), event_id, event_type, livemode, payload)
            return
        refund = await get_refund_by_id(database(), local_refund_id)
        if refund is None:
            await record_stripe_event(database(), event_id, event_type, livemode, payload)
            return
        refund = await attach_refund_submission(database(), refund.refund_id, stripe_refund_id)
    refund_status = _string(event_object, "status")
    if refund_status == "succeeded":
        await complete_refund(database(), refund.refund_id, stripe_refund_id)
    elif refund_status in {"failed", "canceled"}:
        await fail_refund(database(), refund.refund_id, f"Stripe refund status: {refund_status}")
    await record_stripe_event(database(), event_id, event_type, livemode, payload)


def _price_response(value: PriceRow) -> ModelPriceResponse:
    return ModelPriceResponse(
        price_id=value.price_id,
        model_name=value.model_name,
        version=value.version,
        input_micros_per_million=value.input_micros_per_million,
        cached_input_micros_per_million=value.cached_input_micros_per_million,
        output_micros_per_million=value.output_micros_per_million,
        embedding_micros_per_million=value.embedding_micros_per_million,
        default_max_output_tokens=value.default_max_output_tokens,
        max_output_tokens=value.max_output_tokens,
        enabled=value.enabled,
        effective_at=value.effective_at,
    )


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Stripe event")
    return cast(Mapping[str, object], value)  # cast-ok: isinstance validates the webhook mapping boundary.


def _string(value: Mapping[str, object], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Stripe event")
    return result


def _boolean(value: Mapping[str, object], key: str) -> bool:
    result = value.get(key)
    if not isinstance(result, bool):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Stripe event")
    return result


def _integer(value: Mapping[str, object], key: str) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Stripe event")
    return result


def _expandable_id(value: object) -> str:
    if isinstance(value, str):
        return value
    return _string(_mapping(value), "id")
