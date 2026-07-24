from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast  # noqa: TID251, RUF100  # Stripe omits a precise webhook callable type.

import stripe

from litellm.proxy.public_relay.repository import CheckoutOrder


@dataclass(frozen=True, slots=True)
class CheckoutSession:
    session_id: str
    url: str


@dataclass(frozen=True, slots=True)
class StripeRefund:
    refund_id: str
    status: str


class StripeGateway(Protocol):
    async def create_checkout(self, order: CheckoutOrder, email: str) -> CheckoutSession: ...

    async def create_refund(
        self,
        refund_id: str,
        payment_intent_id: str,
        amount_micros: int,
        idempotency_key: str,
        reason: str,
    ) -> StripeRefund: ...


class WebhookConstructor(Protocol):
    def __call__(self, *, payload: bytes, sig_header: str, secret: str) -> stripe.Event: ...


@dataclass(frozen=True, slots=True)
class StripeSdkGateway:
    secret_key: str
    success_url: str
    cancel_url: str

    async def create_checkout(self, order: CheckoutOrder, email: str) -> CheckoutSession:
        client = stripe.StripeClient(self.secret_key, max_network_retries=2)
        session = await client.v1.checkout.sessions.create_async(
            {
                "mode": "payment",
                "success_url": self.success_url,
                "cancel_url": self.cancel_url,
                "customer_email": email,
                "client_reference_id": order.payment_id,
                "line_items": [
                    {
                        "quantity": 1,
                        "price_data": {
                            "currency": "usd",
                            "unit_amount": order.amount_micros // 10_000,
                            "product_data": {"name": "LiteLLM relay balance"},
                        },
                    }
                ],
                "metadata": {
                    "public_relay_payment_id": order.payment_id,
                    "public_relay_account_id": order.account_id,
                },
                "payment_intent_data": {
                    "metadata": {
                        "public_relay_payment_id": order.payment_id,
                        "public_relay_account_id": order.account_id,
                    }
                },
            },
            options={"idempotency_key": order.idempotency_key},
        )
        if session.url is None:
            raise RuntimeError("Stripe checkout URL is missing")
        return CheckoutSession(session_id=session.id, url=session.url)

    async def create_refund(
        self,
        refund_id: str,
        payment_intent_id: str,
        amount_micros: int,
        idempotency_key: str,
        reason: str,
    ) -> StripeRefund:
        client = stripe.StripeClient(self.secret_key, max_network_retries=2)
        refund = await client.v1.refunds.create_async(
            {
                "payment_intent": payment_intent_id,
                "amount": amount_micros // 10_000,
                "metadata": {"public_relay_refund_id": refund_id, "reason": reason},
            },
            options={"idempotency_key": idempotency_key},
        )
        return StripeRefund(refund_id=refund.id, status=refund.status or "pending")


def parse_webhook(payload: bytes, signature: str, secret: str) -> stripe.Event:
    constructor = cast(  # cast-ok: Stripe documents construct_event with this callable contract.
        WebhookConstructor, stripe.Webhook.construct_event
    )
    return constructor(payload=payload, sig_header=signature, secret=secret)
