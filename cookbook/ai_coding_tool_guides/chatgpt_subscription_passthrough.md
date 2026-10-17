# Use a personal ChatGPT subscription through LiteLLM

Compatible Codex clients can send subscription requests through an authenticated LiteLLM gateway. The client signs into ChatGPT and refreshes its own tokens. LiteLLM does not sign in, refresh tokens, or store subscription credentials.

This route supports HTTP and server-sent events with uncompressed JSON requests. Set `features.enable_request_compression = false` in Codex. Requests declaring a content encoding other than `identity` return HTTP 415 before gateway authentication or body parsing. Subscription WebSockets, voice, and gateway OAuth sign-in are not supported. It does not change the API-key-backed `/v1/responses` endpoint.

## Configure the gateway

Start a proxy with a master key and a database, then create a virtual key for each gateway user through the usual key-management workflow. No `OPENAI_API_KEY`, server-side ChatGPT login, or model deployment is required for these routes. Existing virtual-key model and route restrictions still apply to inference requests.

Set the client base URL to `https://litellm.example.com/backend-api/codex`. The client sends its ChatGPT token in `Authorization: Bearer <subscription token>` and its gateway virtual key in `x-litellm-api-key`. If the deployment uses `general_settings.litellm_key_header_name`, send `Bearer <gateway virtual key>` in that configured header instead.

LiteLLM forwards requests only to the corresponding endpoint under `https://chatgpt.com/backend-api/codex`, with redirects disabled. Gateway credentials and cookies are not forwarded. `ChatGPT-Account-ID` and client protocol headers are preserved. Missing gateway or subscription credentials are rejected before an upstream request.

## Pi

Pi 1.0.4 compresses SSE request bodies when its runtime supports zstd. Direct compressed Pi requests are not supported. Use the configuration below only behind an ingress explicitly configured to decompress requests before forwarding them to LiteLLM. TLS termination or response compression alone is insufficient. The ingress must enforce decompression limits and update the content encoding and length headers.

Sign into the built-in `openai-codex` provider through Pi's `/login`. Keep the subscription credential in Pi, not in the gateway configuration.

Add an override to Pi's `models.json`, merging it with any existing providers.

```json
{
  "providers": {
    "openai-codex": {
      "baseUrl": "https://litellm.example.com/backend-api/codex",
      "headers": {
        "x-litellm-api-key": "${LITELLM_GATEWAY_KEY}"
      }
    }
  }
}
```

Set `LITELLM_GATEWAY_KEY` in the environment of the Pi process. Do not replace Pi's subscription credential with the virtual key. Select an available model from the built-in provider with `/model`.

Set the transport in Pi's `settings.json` to HTTP/SSE.

```json
{
  "transport": "sse"
}
```

Pi retains responsibility for its normal ChatGPT login and refresh flow. Restart Pi after changing its process environment.

## Codex

Use a Codex build supporting OpenAI's [ChatGPT gateway flow](https://developers.openai.com/codex/enterprise/sign-in-with-chatgpt-through-a-gateway). Disable request compression in the client's `config.toml`.

```toml
[features]
enable_request_compression = false
```

This is a client setting, not a LiteLLM gateway setting. Codex selects request compression locally rather than negotiating support from server headers. A client sending `Content-Encoding: zstd` receives HTTP 415 unless an ingress decompresses the request first.

For a managed workspace, configure the provider in `requirements.toml`.

```toml
model_provider = "litellm_subscription"

[model_providers.litellm_subscription]
name = "LiteLLM subscription gateway"
base_url = "https://litellm.example.com/backend-api/codex"
requires_openai_auth = true
supports_websockets = false
env_http_headers = { "x-litellm-api-key" = "LITELLM_GATEWAY_KEY" }
```

Provide the gateway key in the Codex process environment and sign in with ChatGPT normally. Use HTTP/SSE rather than WebSocket-only transport. Older clients might not support these managed settings. This configuration does not install an OAuth server or move ChatGPT login into LiteLLM.

## Verify routing and usage

`GET https://litellm.example.com/backend-api/codex/models?client_version=<your-client-version>` returns the upstream Codex catalog, including its native `models` array. It is not the gateway's API-backed `/v1/models` catalog. OpenAI determines subscription model availability, while the gateway enforces its inference restrictions separately.

Send a prompt from Pi or Codex. `POST https://litellm.example.com/backend-api/codex/responses` streams the provider's SSE events without translating the response ID. Open `https://litellm.example.com/ui/?page=logs` and locate the request under the caller's gateway virtual key and user. Token counts come from the terminal Responses usage event.

Gateway cost estimates are API-equivalent estimates, not subscription invoices, remaining subscription quota, or charges from OpenAI. When a model has no local price, LiteLLM still records its reported token usage with zero estimated cost.

Subscription credentials are redacted in gateway logging and callback header copies. Keep credentials out of request bodies, URLs, screenshots, shell history, and custom application logs. Do not enable shell tracing or verbose curl output when testing with real credentials.

## Compressed requests

Native zstd request decoding is deferred. Supporting `enable_request_compression = true` requires a maintainer follow-up that declares the zstd dependency and adds bounded decoding before authentication and body parsing. Adding the dependency alone does not enable compressed requests. Other proxy routes retain their existing request-body behavior.
