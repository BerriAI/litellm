# /realtime - WebRTC Support

Use this to connect to the Realtime API via WebRTC from browser/mobile clients, with LiteLLM handling authentication and key management.

Supported Providers:
- OpenAI
- Azure

:::info
**When to use WebRTC vs WebSocket?**

- Use **WebSocket** (`/v1/realtime`) for server-to-server connections
- Use **WebRTC** (`/v1/realtime/client_secrets` + `/v1/realtime/calls`) for browser/mobile clients where lower latency matters
:::

## How it works

WebRTC keeps your provider API keys secure while allowing the browser to stream audio directly to OpenAI/Azure — without routing audio through LiteLLM.

```
Browser                  LiteLLM Proxy              OpenAI/Azure
  |                           |                          |
  |-- POST /v1/realtime/      |                          |
  |   client_secrets -------->|                          |
  |   [LiteLLM API key]       |-- POST /v1/realtime/    |
  |                           |   sessions [Real key] -->|
  |                           |<-- { ek_... } -----------|
  |                           | encrypt(ek_...)           |
  |<-- { encrypted_token } ---|                          |
  |                           |                          |
  |-- POST /v1/realtime/calls |                          |
  |   [SDP + encrypted_token]>|                          |
  |                           | decrypt → ek_...         |
  |                           |-- POST /v1/realtime/    |
  |                           |   calls [SDP + ek_...] ->|
  |                           |<-- SDP answer -----------|
  |<-- SDP answer ------------|                          |
  |                           |                          |
  |===== audio P2P direct to OpenAI/Azure =============>|
```

LiteLLM **never touches the audio stream** — it only handles token issuance and the SDP exchange. All audio flows directly browser ↔ provider.

## Proxy Usage

### Add model to config

```yaml
model_list:
  - model_name: gpt-4o-realtime
    litellm_params:
      model: openai/gpt-4o-realtime-preview-2024-12-17
      api_key: os.environ/OPENAI_API_KEY
    model_info:
      mode: realtime
```

For Azure:

```yaml
model_list:
  - model_name: gpt-4o-realtime
    litellm_params:
      model: azure/gpt-4o-realtime-preview
      api_key: os.environ/AZURE_API_KEY
      api_base: os.environ/AZURE_API_BASE
    model_info:
      mode: realtime
```

### Start proxy

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

## Client Usage

### Step 1 — Get an encrypted session token

Call `POST /v1/realtime/client_secrets` from your browser using your LiteLLM API key. LiteLLM will fetch a real ephemeral key from OpenAI, encrypt it, and return the encrypted token — so the real provider key never reaches your browser.

```javascript
const tokenResponse = await fetch("http://your-litellm-proxy:4000/v1/realtime/client_secrets", {
  method: "POST",
  headers: {
    "Authorization": "Bearer sk-litellm-your-key",
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    model: "gpt-4o-realtime",  // your model name from config
  }),
});

const { client_secret } = await tokenResponse.json();
const ENCRYPTED_TOKEN = client_secret.value; // encrypted by LiteLLM, not the real ek_...
```

### Step 2 — Establish WebRTC connection via LiteLLM

Use standard WebRTC APIs to set up the peer connection, then send your SDP offer to LiteLLM's `/v1/realtime/calls` endpoint. LiteLLM decrypts the token and forwards the SDP to OpenAI using the real ephemeral key.

```javascript
// Create a peer connection
const pc = new RTCPeerConnection();

// Set up to play remote audio from the model
const audioEl = document.createElement("audio");
audioEl.autoplay = true;
pc.ontrack = (e) => (audioEl.srcObject = e.streams[0]);

// Add local audio track for microphone input
const ms = await navigator.mediaDevices.getUserMedia({ audio: true });
pc.addTrack(ms.getTracks()[0]);

// Set up data channel for sending and receiving events
const dc = pc.createDataChannel("oai-events");

// Create SDP offer
const offer = await pc.createOffer();
await pc.setLocalDescription(offer);

// Send SDP to LiteLLM — it decrypts the token and forwards to OpenAI
const sdpResponse = await fetch("http://your-litellm-proxy:4000/v1/realtime/calls", {
  method: "POST",
  headers: {
    "Authorization": `Bearer ${ENCRYPTED_TOKEN}`,
    "Content-Type": "application/sdp",
  },
  body: offer.sdp,
});

// Set the SDP answer from OpenAI (returned via LiteLLM)
const answer = {
  type: "answer",
  sdp: await sdpResponse.text(),
};
await pc.setRemoteDescription(answer);

// Audio now flows directly browser <-> OpenAI/Azure (P2P)
```

### Step 3 — Send and receive events

Use the WebRTC data channel to send and receive session events:

```javascript
// Listen for server events
dc.addEventListener("message", (e) => {
  const event = JSON.parse(e.data);
  console.log(event);
});

// Send a client event
dc.send(JSON.stringify({
  type: "session.update",
  session: {
    instructions: "You are a helpful assistant.",
  },
}));
```