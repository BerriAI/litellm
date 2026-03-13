import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Realtime API - WebRTC (Browser & Mobile)

Connect to the Realtime API via WebRTC for browser and mobile clients. LiteLLM handles authentication while audio streams **directly** to OpenAI or Azure — bypassing the proxy entirely.

**Supported Providers:** OpenAI, Azure

## WebRTC vs WebSocket

| | WebSocket `/v1/realtime` | WebRTC `/v1/realtime/client_secrets` + `/v1/realtime/calls` |
|---|---|---|
| **Best for** | Server-to-server | Browser / mobile |
| **Latency** | Standard | Lower (peer-to-peer audio) |
| **Audio routing** | Through proxy | Direct to provider |
| **Auth** | LiteLLM API key | Encrypted short-lived token |

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

## Setup

### 1. Configure `config.yaml`

<Tabs>
<TabItem value="openai" label="OpenAI">

```yaml
model_list:
  - model_name: gpt-4o-realtime
    litellm_params:
      model: openai/gpt-4o-realtime-preview-2024-12-17
      api_key: os.environ/OPENAI_API_KEY
    model_info:
      mode: realtime
```

</TabItem>
<TabItem value="azure" label="Azure">

```yaml
model_list:
  - model_name: gpt-4o-realtime
    litellm_params:
      model: azure/gpt-4o-realtime-preview
      api_key: os.environ/AZURE_API_KEY
      api_base: os.environ/AZURE_API_BASE
      api_version: "2024-10-01-preview"
    model_info:
      mode: realtime
```

</TabItem>
</Tabs>

### 2. Start the proxy

```bash
litellm --config /path/to/config.yaml
```

### 3. Connect from the browser

Three steps: get a token → set up WebRTC → send SDP offer.

```javascript
// Step 1 — Get short-lived encrypted token
const r = await fetch("http://localhost:4000/v1/realtime/client_secrets", {
  method: "POST",
  headers: {
    "Authorization": "Bearer <your-litellm-api-key>",
    "Content-Type": "application/json",
  },
  body: JSON.stringify({ model: "gpt-4o-realtime" }),
});
const token = (await r.json()).client_secret.value;

// Step 2 — Set up WebRTC peer connection
const pc = new RTCPeerConnection();

// Play back remote audio
const audio = document.createElement("audio");
audio.autoplay = true;
pc.ontrack = (e) => (audio.srcObject = e.streams[0]);

// Add microphone input
const ms = await navigator.mediaDevices.getUserMedia({ audio: true });
pc.addTrack(ms.getTracks()[0]);

// Create data channel for events
const dc = pc.createDataChannel("oai-events");

// Step 3 — Send SDP offer using the encrypted token
const offer = await pc.createOffer();
await pc.setLocalDescription(offer);

const sdpRes = await fetch("http://localhost:4000/v1/realtime/calls", {
  method: "POST",
  headers: {
    "Authorization": `Bearer ${token}`,   // use the token, not your API key
    "Content-Type": "application/sdp",
  },
  body: offer.sdp,
});

await pc.setRemoteDescription({ type: "answer", sdp: await sdpRes.text() });

// Send session config through data channel
dc.send(JSON.stringify({
  type: "session.update",
  session: { instructions: "You are a helpful assistant." },
}));
```

## API Reference

### `POST /v1/realtime/client_secrets`

Issues a short-lived encrypted token for a WebRTC session.

**Request body**

| Field | Type | Required | Description |
|---|---|---|---|
| `model` | string | Yes | Model name as configured in `config.yaml`, e.g. `gpt-4o-realtime` |

**Response**

```json
{
  "client_secret": {
    "value": "<encrypted-token>",
    "expires_at": 1700000060
  }
}
```

### `POST /v1/realtime/calls`

Accepts an SDP offer and returns an SDP answer to complete the WebRTC handshake. Use the token from `client_secrets` as the Bearer token — **not** your LiteLLM API key.

**Headers**

```
Authorization: Bearer <token-from-client_secrets>
Content-Type: application/sdp
```

**Body:** Raw SDP offer string (`pc.localDescription.sdp`)

**Response:** Raw SDP answer string

## Troubleshooting

**`401 Unauthorized` on `/v1/realtime/calls`**
The token from `client_secrets` expires quickly. Fetch a fresh token immediately before creating the WebRTC offer — don't cache it.

**No audio**
- Check that the browser has microphone permission
- Confirm `pc.ontrack` sets `audio.srcObject` before the offer
- Check browser console for WebRTC errors
- Some corporate firewalls block WebRTC — test on a different network

**Azure: wrong API version**
Set `api_version` in `litellm_params`. Supported version: `2024-10-01-preview`.

**Do I pass `model` to `/v1/realtime/calls`?**
No. The model is encoded inside the token returned by `client_secrets`.

## Related

- [Realtime API (WebSocket)](/docs/realtime) — server-to-server WebSocket realtime
- [OpenAI Realtime docs](https://platform.openai.com/docs/guides/realtime-webrtc) — upstream WebRTC reference
