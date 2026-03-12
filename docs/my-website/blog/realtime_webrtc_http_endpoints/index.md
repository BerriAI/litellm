---
slug: realtime_webrtc_http_endpoints
title: "Realtime WebRTC HTTP Endpoints on LiteLLM Proxy"
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
---
id: webrtc
title: "/realtime - WebRTC Support"
sidebar_label: "/realtime WebRTC"
---

import WebRTCTester from '@site/src/components/WebRTCTester';

Use this to connect to the Realtime API via WebRTC from browser/mobile clients, with LiteLLM handling authentication and key management.

**Supported Providers:**
- OpenAI
- Azure OpenAI

:::info When to use WebRTC vs WebSocket?
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

---

## Proxy Setup

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

### Start the proxy

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

---

## Client Usage

### Step 1 — Get an encrypted session token

Call `POST /v1/realtime/client_secrets` from your browser using your LiteLLM API key. LiteLLM fetches a real ephemeral key from OpenAI, encrypts it, and returns the encrypted token — so your provider key never reaches the browser.

```javascript
const tokenResponse = await fetch("http://your-litellm-proxy:4000/v1/realtime/client_secrets", {
  method: "POST",
  headers: {
    "Authorization": "Bearer sk-litellm-your-key",
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    model: "gpt-4o-realtime",  // model_name from your config
  }),
});

const { client_secret } = await tokenResponse.json();
const ENCRYPTED_TOKEN = client_secret.value; // encrypted by LiteLLM, not the real ek_...
```

### Step 2 — Establish WebRTC connection via LiteLLM

Use the standard WebRTC APIs to set up the peer connection, then send your SDP offer to LiteLLM's `/v1/realtime/calls` endpoint. LiteLLM decrypts the token (which encodes the model) and forwards the SDP to OpenAI — no need to pass `model` again.

```javascript
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

// Send SDP to LiteLLM — model is decoded from the token, no ?model= needed
const sdpResponse = await fetch("http://your-litellm-proxy:4000/v1/realtime/calls", {
  method: "POST",
  headers: {
    "Authorization": `Bearer ${ENCRYPTED_TOKEN}`,
    "Content-Type": "application/sdp",
  },
  body: offer.sdp,
});

const answer = { type: "answer", sdp: await sdpResponse.text() };
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

---

## Try it live

Paste your LiteLLM proxy URL and API key to run a real end-to-end WebRTC session right here.

<WebRTCTester />

---

## FAQ

### Why do I get `401 Token has expired` on `/v1/realtime/calls`?

The encrypted token returned by `/v1/realtime/client_secrets` is short-lived.
Generate a fresh token right before creating your WebRTC offer, and avoid reusing old tokens across page refreshes or long idle periods.

### Do I send my LiteLLM key or provider key to `/v1/realtime/calls`?

Use the **encrypted token** from `/v1/realtime/client_secrets` as:

```http
Authorization: Bearer <encrypted_token>
```

Do not send your raw OpenAI/Azure key from the client.

### Do I need to pass `model` again on `/v1/realtime/calls`?

Usually no. The encrypted token encodes routing metadata (including model), so LiteLLM can route the SDP exchange without `?model=...`.

### Azure call failing with `api-version` errors - what should I check?

Make sure your Azure deployment config includes a valid `api_version` in `litellm_params` (or set `AZURE_API_VERSION`), plus correct `api_base` and deployment/model mapping.

### Why does the SDP request need `Content-Type: application/sdp` on the client?

Your browser sends raw SDP text to LiteLLM, so `application/sdp` is correct for the client-to-proxy request.
LiteLLM then transforms and forwards provider-specific payloads upstream.

### The browser asks for microphone permission but I hear no audio. What can I check?

- Confirm microphone permission is granted for your site.
- Ensure `pc.ontrack` sets an autoplay-enabled audio element.
- Verify your network allows WebRTC (no restrictive firewall or enterprise policy).
- Check browser console logs for ICE and SDP negotiation errors.

---
