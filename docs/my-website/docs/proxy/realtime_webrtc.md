# /realtime - WebRTC Support

Connect to the Realtime API via WebRTC from browser/mobile clients. LiteLLM handles auth; audio streams directly to OpenAI/Azure.

**Providers:** OpenAI · Azure

:::info **WebRTC vs WebSocket**
- **WebSocket** (`/v1/realtime`) — server-to-server
- **WebRTC** (`/v1/realtime/client_secrets` + `/v1/realtime/calls`) — browser/mobile, lower latency
:::

## How it works

LiteLLM issues tokens and relays SDP; audio never passes through the proxy.

```
Browser                  LiteLLM Proxy              OpenAI/Azure
  |                           |                          |
  |-- POST client_secrets --->|-- POST sessions -------->|
  |<-- encrypted_token -------|<-- ek_... ---------------|
  |-- POST calls [SDP+token] ->|-- POST calls ----------->|
  |<-- SDP answer ------------|<-- SDP answer -----------|
  |===== audio P2P direct ===============================>|
```

## Proxy Setup

```yaml
model_list:
  - model_name: gpt-4o-realtime
    litellm_params:
      model: openai/gpt-4o-realtime-preview-2024-12-17
      api_key: os.environ/OPENAI_API_KEY
    model_info:
      mode: realtime
```

**Azure:** `model: azure/gpt-4o-realtime-preview`, `api_key`, `api_base`.

```bash
litellm --config /path/to/config.yaml
```

## Client Usage

1. **Token** — `POST /v1/realtime/client_secrets` with LiteLLM key and `{ model }`.
2. **WebRTC** — Create `RTCPeerConnection`, add mic, data channel `oai-events`, send SDP offer to `POST /v1/realtime/calls` with `Authorization: Bearer <token>`, `Content-Type: application/sdp`.
3. **Events** — Use data channel for `session.update` and other events.

```javascript
const r = await fetch("http://proxy:4000/v1/realtime/client_secrets", {
  method: "POST",
  headers: { "Authorization": "Bearer sk-litellm-key", "Content-Type": "application/json" },
  body: JSON.stringify({ model: "gpt-4o-realtime" }),
});
const token = (await r.json()).client_secret.value;

const pc = new RTCPeerConnection();
const audio = document.createElement("audio");
audio.autoplay = true;
pc.ontrack = (e) => (audio.srcObject = e.streams[0]);
const ms = await navigator.mediaDevices.getUserMedia({ audio: true });
pc.addTrack(ms.getTracks()[0]);
const dc = pc.createDataChannel("oai-events");
const offer = await pc.createOffer();
await pc.setLocalDescription(offer);

const sdpRes = await fetch("http://proxy:4000/v1/realtime/calls", {
  method: "POST",
  headers: { "Authorization": `Bearer ${token}`, "Content-Type": "application/sdp" },
  body: offer.sdp,
});
await pc.setRemoteDescription({ type: "answer", sdp: await sdpRes.text() });

dc.send(JSON.stringify({ type: "session.update", session: { instructions: "..." } }));
```

## FAQ

- **401 Token expired** — Get a fresh token right before creating the WebRTC offer.
- **Which key for `/calls`?** — Encrypted token from `client_secrets`, not raw key.
- **Pass `model`?** — No. Token encodes routing.
- **Azure `api-version`** — Set `api_version` in `litellm_params` and correct `api_base`.
- **No audio** — Grant mic; ensure `pc.ontrack` sets autoplay audio; check firewall/WebRTC; inspect console.