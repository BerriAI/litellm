# Enable MCP Events

The gateway relays events from MCP servers that support the beta Events extension. It checks the caller's access before listing event types, polling or starting a stream. For webhooks, it verifies and stores incoming deliveries in Postgres and signs a new request to the client's HTTPS endpoint. The gateway does not produce its own events.

## Configure the gateway

Apply the subscription, delivery and object-permission migrations before enabling a gateway worker. Workers need the same database and encryption key. Configure the public HTTPS origin that the upstream can reach. Enable each participating server independently.

```yaml
general_settings:
  mcp_advertised_versions: ["2026-07-28", "2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"]
  mcp_events_enabled: true
  mcp_events_public_base_url: https://gateway.example.com
mcp_servers:
  ci:
    url: https://ci.example.com/mcp
    transport: http
    protocol_version: "2026-07-28"
    available_on_public_internet: true
    events:
      enabled: true
      allowed_events: ["build.*"]
```

Grant server access to the caller. To limit a key to a specific event, set its object permission's `mcp_event_permissions` map to `{"<server-id>":["build.failed"]}`. Each independent source of server access grants its own event set, subject to the caller's applicable restrictions. Tool permissions are separate.

## List and poll

This example uses Python and HTTPX. Set `GATEWAY` to the gateway origin and `LITELLM_API_KEY` to a virtual key that can use the server. Each request declares the protocol revision and the extension. With Events disabled the same request returns the JSON-RPC method-not-found error.

```python
import json
import os

import httpx

revision = "2026-07-28"
meta = {
    "io.modelcontextprotocol/protocolVersion": revision,
    "io.modelcontextprotocol/clientInfo": {"name": "build-listener", "version": "1"},
    "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {"io.modelcontextprotocol/events": {}}
    },
}

def call(method, params, routing_headers=None):
    headers = {
        "Authorization": "Bearer " + os.environ["LITELLM_API_KEY"],
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": revision,
        "Mcp-Method": method,
        **(routing_headers or {}),
    }
    response = httpx.post(
        os.environ["GATEWAY"].rstrip("/") + "/mcp",
        headers=headers,
        json={"jsonrpc": "2.0", "id": 1, "method": method,
              "params": {**params, "_meta": meta}},
        follow_redirects=True,
    )
    response.raise_for_status()
    body = next((line[6:] for line in response.text.splitlines()
                 if line.startswith("data: ")), response.text)
    message = json.loads(body)
    if "error" in message:
        raise RuntimeError(message["error"])
    return message["result"]

catalog = call("events/list", {})
name = "ci-build.failed"
page = call("events/poll", {"name": name, "arguments": {}, "cursor": None},
            {"Mcp-Name": name})
```

Use the names in the returned catalog. Supply any argument headers declared in the event descriptor. Process the events and persist the opaque cursor with the same caller, name and arguments. Poll immediately while `hasMore` is true, then wait `nextPollMs`. If `truncated` is true, reconcile the missing interval with the upstream application's state.

## Receive a webhook

Expose an HTTPS endpoint that verifies the Standard Webhooks signature over the raw request bytes and rejects stale timestamps. For a signed verification envelope, return JSON containing the same challenge. For a signed event, durably record its `eventId` before returning a 2xx. A delivery can be retried after a successful POST if the worker exits before recording completion.

```python
import base64
import secrets

callback = os.environ["CALLBACK_URL"]
secret = "whsec_" + base64.b64encode(secrets.token_bytes(32)).decode()
# Provision the receiving endpoint with this secret before calling subscribe.
subscription = call(
    "events/subscribe",
    {"name": name, "arguments": {},
     "delivery": {"mode": "webhook", "url": callback, "secret": secret}},
    {"Mcp-Name": name},
)
```

Refresh before `refreshBefore` when it is set. Persist the latest safe cursor and include it on refresh. A gap means the client must reconcile events from its last usable position. To end delivery, call `events/unsubscribe` with the original event, arguments and delivery URL. If the call times out or reports unconfirmed teardown, retry with the same event, arguments and delivery URL. The gateway holds delivery while it waits for the upstream acknowledgement. The upstream sees the gateway callback URL and its own signing secret.

```python
call("events/unsubscribe", {"name": name, "arguments": {}, "delivery": {"url": callback}},
     {"Mcp-Name": name})
```

Operators can inspect delivery status in the MCP Servers dashboard or page through `GET /v1/mcp/events/subscriptions`. Turning off `mcp_events_enabled` stops new admissions, inbound callbacks and new worker claims. Account for callbacks already in flight.
