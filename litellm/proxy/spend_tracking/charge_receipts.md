# Required charge receipts

Required receipts are an opt-in accounting contract for selected virtual keys. The proxy reuses its canonical cost calculation and compact spend-event projection. It commits the key's `spend` and `total_spend` increments together with an immutable receipt in PostgreSQL, then delivers that receipt through a separate worker

The normal cached-budget bookkeeping runs before the database charge, so a cold cached counter does not seed from a balance that already includes the same charge. The ordinary callback retains its other rollups and alerts while skipping the key increment and counters already applied by required capture

Configure these values separately on the gateway and delivery worker:

```sh
LITELLM_CHARGE_RECEIPTS_ENABLED=true
LITELLM_CHARGE_RECEIPTS_KEY_HASHES='["<sha256 of the application virtual key>"]'
LITELLM_CHARGE_RECEIPTS_GATEWAY_SCOPE=<consumer's stable gateway scope digest>
LITELLM_CHARGE_RECEIPTS_DESTINATION_URL=https://consumer.example/billing/receipts
LITELLM_CHARGE_RECEIPTS_DESTINATION_SECRET=<shared secret from the secret store>
```

All configured keys share one fixed consumer destination. The URL must use HTTPS, except for loopback tests. Request data cannot select a destination. Authentication and the signed billing context use the same strong shared secret, at least 32 whitespace-free ASCII characters

Apply the normal proxy schema migration before enabling the feature. It adds `LiteLLM_ChargeReceipt` without rewriting existing rows. Receipt storage is independent of optional spend logs, their cleanup, and the collector's in-memory buffer. Startup requires the receipt table, `fsync=on`, and synchronous commit

Run delivery in a separate process with its own bounded database connection configuration:

```sh
python -m litellm.proxy.collector --charge-receipts
```

The worker uses the writer database, leases bounded batches, and acknowledges only event IDs explicitly confirmed by the consumer. It retains unavailable, rejected, and unacknowledged events. Expired leases can be reclaimed by another worker. The response path never waits for the consumer or its acknowledgement

The authenticated `GET /billing/receipts/capabilities` endpoint reports whether the caller's key requires version 1 receipts and the configured gateway scope

For each inference call, the application sends `x-litellm-billing-context`, an HS256 JWT with these claims:

```json
{
  "version": 1,
  "sub": "<original admitted request ID>",
  "aud": "litellm-charge-receipts",
  "iat": 0,
  "exp": 300,
  "gateway_scope": "<configured digest>",
  "key_hash": "<authenticated virtual-key digest>",
  "method": "POST",
  "route": "/v1/responses",
  "body_sha256": "<digest of the exact request bytes>"
}
```

The timestamps above illustrate a five-minute validity interval; use the actual issuance time. Context is checked against the authenticated key and exact body, method, and path. It is excluded from provider forwarding and credential-bearing logging copies

Version 1 supports `/responses`, `/responses/compact`, and `/chat/completions`, including their `/v1` aliases. Background requests and WebSocket inference are rejected for required-receipt keys. Responses terminal envelopes and Chat Completions' final DONE wait for durable capture. Non-streaming responses wait before returning. Failures after dispatch tell compatible SDKs not to automatically retry inference

Use dedicated application keys. Required-receipt keys may read the existing information routes and call `/utils/token_counter`. All other unqualified routes, including custom passthrough routes, are rejected before dispatch

If a streaming client disconnects, the gateway keeps reading the in-flight provider call for up to `LITELLM_CHARGE_RECEIPTS_DRAIN_TIMEOUT_SECONDS`, which defaults to 120 seconds. This captures final usage during application deployments. A user pressing Stop also leaves that one call running within this bound. It does not start another inference request. After the deadline the upstream stream closes, and any amount that cannot be established remains unresolved. Give the gateway's graceful shutdown a longer deadline when draining these calls

The delivery body is an object with an `events` array of at most 32 receipts. Each receipt contains `version`, `event_id`, `request_id`, `gateway_scope`, `key_hash`, `currency`, `amount_usd`, `model`, and optional token counters. It contains no prompt, response text, or credential. Amounts are decimal strings and currency is `USD`

The consumer returns HTTP 200 with an `acknowledged` array containing only event IDs committed locally. It may return per-event `rejected` details. Duplicate event delivery must not change money twice, and a reused event ID with different contents must be rejected. Delivery acknowledgements referring to events outside the leased batch are rejected by the publisher

Deploy compatible gateway code across all replicas before applications start requiring receipts. Applications may start sending signed context while enforcement is disabled. Then start the consumer and publisher, enable gateway enforcement everywhere, and switch application accounting to receipts. Drain requests before an incompatible rollback

A committed receipt survives publisher or gateway process loss and can be delivered later. A crash before capture commits still leaves an uncertain upstream amount. The contract does not make provider invoices discoverable when usage was never reported, and it does not replay inference. PostgreSQL durability and backup/failover guarantees remain deployment requirements

If a key is removed while an admitted request is finishing, the original receipt is retained with `key_counter_updated=false`; no credential is recreated. Investigate that retained evidence against the original key scope. Existing key counters are floating-point values, while each receipt retains the canonical decimal representation of its reported charge

The small durable write has a real completion-time cost. Measure the deployed database, cached-budget work, streaming completion, and delivery contention. Time to first text, full completion, and receipt visibility are separate metrics
