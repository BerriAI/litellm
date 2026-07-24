import json
import time

import pytest
import stripe

from litellm.proxy.public_relay.stripe_client import parse_webhook


def test_parse_webhook_accepts_valid_signature() -> None:
    payload = json.dumps({"id": "evt_1", "object": "event", "type": "checkout.session.completed"}).encode()
    secret = "whsec_test"
    timestamp = int(time.time())
    signature = stripe.WebhookSignature._compute_signature(f"{timestamp}.{payload.decode()}", secret)
    header = f"t={timestamp},v1={signature}"

    event = parse_webhook(payload, header, secret)

    assert event.id == "evt_1"


def test_parse_webhook_rejects_invalid_signature() -> None:
    payload = json.dumps({"id": "evt_1", "object": "event"}).encode()

    with pytest.raises(stripe.SignatureVerificationError):
        parse_webhook(payload, "t=1,v1=invalid", "whsec_test")
