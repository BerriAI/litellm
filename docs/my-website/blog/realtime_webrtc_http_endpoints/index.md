---
slug: realtime_webrtc_http_endpoints
title: "Realtime WebRTC HTTP Endpoints"
date: 2026-03-12T10:00:00
authors:
  - name: Sameer Kankute
    title: SWE @ LiteLLM (LLM Translation)
    url: https://www.linkedin.com/in/sameer-kankute/
    image_url: https://pbs.twimg.com/profile_images/2001352686994907136/ONgNuSk5_400x400.jpg
  - name: Krrish Dholakia
    title: "CEO, LiteLLM"
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://pbs.twimg.com/profile_images/1298587542745358340/DZv3Oj-h_400x400.jpg
  - name: Ishaan Jaff
    title: "CTO, LiteLLM"
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://pbs.twimg.com/profile_images/1613813310264340481/lz54oEiB_400x400.jpg
description: "Use the LiteLLM proxy to route OpenAI-style WebRTC realtime via HTTP: client_secrets and SDP exchange."
tags: [realtime, webrtc, proxy, openai]
hide_table_of_contents: false
---

import WebRTCTester from '@site/src/components/WebRTCTester';

Connect to the Realtime API via WebRTC from browser/mobile clients. LiteLLM handles auth and key management; audio streams directly to OpenAI/Azure.

**Providers:** OpenAI · Azure OpenAI

:::info **WebRTC vs WebSocket**
- **WebSocket** (`/v1/realtime`) — server-to-server
- **WebRTC** (`/v1/realtime/client_secrets` + `/v1/realtime/calls`) — browser/mobile, lower latency
:::

## How it works

LiteLLM issues tokens and relays SDP; audio never passes through the proxy.

```
Browser                  LiteLLM Proxy              OpenAI/Azure
  |                           |                          |
  |-- POST /v1/realtime/      |                          |
  |   client_secrets -------->|-- POST sessions -------->|
  |                           |<-- { ek_... } -----------|
  |<-- { encrypted_token } ---|                          |
  |-- POST /v1/realtime/calls |-- POST calls ----------->|
  |   [SDP + token] --------->|                          |
  |<-- SDP answer ------------|<-- SDP answer -----------|
  |===== audio P2P direct to OpenAI/Azure =============>|
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

**Azure:** use `model: azure/gpt-4o-realtime-preview`, `api_key`, `api_base`.

```bash
litellm --config /path/to/config.yaml
```

## Try it live

<WebRTCTester />

## Client Usage

**1. Get token** — `POST /v1/realtime/client_secrets` with LiteLLM API key and `{ model }`.

**2. WebRTC handshake** — Create `RTCPeerConnection`, add mic track, create data channel `oai-events`, send SDP offer to `POST /v1/realtime/calls` with `Authorization: Bearer <encrypted_token>` and `Content-Type: application/sdp`.

**3. Events** — Use the data channel for `session.update` and other events.

<details>
<summary>Full code example</summary>

```javascript
// 1. Token
const r = await fetch("http://proxy:4000/v1/realtime/client_secrets", {
  method: "POST",
  headers: { "Authorization": "Bearer sk-litellm-key", "Content-Type": "application/json" },
  body: JSON.stringify({ model: "gpt-4o-realtime" }),
});
const { client_secret } = await r.json();
const token = client_secret.value;

// 2. WebRTC
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

// 3. Events
dc.send(JSON.stringify({ type: "session.update", session: { instructions: "..." } }));
```

</details>

## FAQ

**Q: What do I do if I get a 401 Token expired error?**  
A: Tokens are short-lived. Get a fresh token right before creating the WebRTC offer.

**Q: Which key should I use for `/v1/realtime/calls`?**  
A: Use the **encrypted token** from `client_secrets`, not your raw API key.

**Q: Should I pass the `model` parameter when making the call?**  
A: No, the encrypted token already encodes all routing information including model.

**Q: How do I resolve Azure `api-version` errors?**  
A: Set the correct `api_version` in `litellm_params` (or via the `AZURE_API_VERSION` environment variable), along with the right `api_base` and deployment values.

**Q: What if I get no audio?**  
A: Make sure you grant microphone permission, ensure `pc.ontrack` assigns the audio element with `autoplay` enabled, check your network/firewall for WebRTC traffic, and inspect the browser console for ICE or SDP errors.

