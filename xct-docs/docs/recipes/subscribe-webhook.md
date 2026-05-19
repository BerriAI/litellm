---
sidebar_position: 7
---

# Subscribe to a webhook

Get pinged when things happen.

## 1. Register

```http
POST /v1/webhooks
Authorization: Bearer <admin or scoped key>
Content-Type: application/json

{
  "target_url": "https://hooks.your-app.test/litellm",
  "events": ["capability.invoked", "budget.exhausted"],
  "app_id": "xct-chat",
  "filters": {"entity_type": "agent"}
}
→ 200 { ..., "secret": "<RETURNED ONCE>" }
```

Save the `secret` immediately. The proxy stores only `sha256(secret)`.

## 2. Verify signature on receive

```python
import hmac, hashlib

def verify(request_body: bytes, header_signature: str, secret: str) -> bool:
    expected = "sha256=" + hmac.new(secret.encode(), request_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header_signature)
```

```python
@app.post("/litellm")
async def webhook(req):
    body = await req.body()
    if not verify(body, req.headers["x-xct-signature"], WEBHOOK_SECRET):
        return Response(status_code=401)
    payload = json.loads(body)
    if payload["event"] == "capability.invoked":
        # payload["data"] has app_id / entity_type / entity_id / spend / …
        track(payload["data"])
    return Response(status_code=200)
```

## 3. Test wiring

```http
POST /v1/webhooks/{subscription_id}/test
→ Sends a synthetic webhook.test event with max_attempts=1
```

## Retry + DLQ

5 attempts at 1s, 5s, 30s, 2m, 10m backoffs. After the last failure the
payload is parked in `LiteLLM_WebhookDLQ`. **20 consecutive failures
auto-disables** the subscription — re-enable via PATCH.

## Filters

`filters` is a tiny DSL: top-level keys in the filter dict must
exact-match the same keys in the event's `data`. AND-joined.
`{"app_id": "xct-chat"}` skips events from other apps.

## Events available today

- `capability.invoked` — every spend-log write with `entity_type`
- `agent.healthcheck.failed` — wired from `/v1/agents?health_check=true`
- `budget.exhausted` — helper exists; emit-site wiring is a follow-up
- `mcp.tool.called` — helper exists; emit-site wiring is a follow-up
