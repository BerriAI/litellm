# IsMalicious MCP guardrail

Inspect normalized MCP argument strings before tool execution and text/structured-content strings before a tool result is returned. IsMalicious checks URL reputation for standalone HTTP(S) URL arguments, then scans the complete normalized text list for content injection and links

## Configuration

Obtain an API key and secret from [your account](https://ismalicious.com/app/account). In your secret manager, set `ISMALICIOUS_ENCODED_API_KEY` to the Base64 encoding of `apiKey:apiSecret`, with no newline. Do not log that value or put it directly into YAML

Merge [`config.yaml`](config.yaml) into the proxy configuration containing your MCP servers. It sets both MCP modes and `default_on: true`. MCP subcalls do not necessarily inherit a parent chat request's guardrail selection, so relying only on a `guardrails` field on the parent request does not establish MCP enforcement

```yaml
guardrails:
  - guardrail_name: ismalicious-mcp
    litellm_params:
      guardrail: ismalicious
      mode: [pre_mcp_call, post_mcp_call]
      default_on: true
      api_key: os.environ/ISMALICIOUS_ENCODED_API_KEY
```

Requests only send credentials to `https://api.ismalicious.com`, with TLS verification, no redirects, a 15-second timeout and no retries. A custom API base is refused. Incoming request headers, provider credentials, user metadata and model prompts are not added to the inspection body by this provider

## Decisions

Top-level `warn` and `block` refuse. Timeout, quota429, redirects, invalid JSON/schema/verdict, incomplete link inspection and oversized serialized UTF-8 request bodies also refuse. A valid `allow` returns the original inputs without masking or sanitization. It means the service did not block under its current rules, not that content is proven benign. An unknown inner link reputation is a valid response and can still have an allow decision

The content request serializes the complete normalized string list into `content` and uses `mode: fast`. The serialized HTTP body must fit within 1 MiB; it is never truncated. Results in `structuredContent`, including labels and numeric fields emitted by the native handler, are included when that handler exposes them. These calls use the separate scan quota, not the indicator lookup quota. URL reputation does not fetch the destination

## Scope

This provider uses LiteLLM's native MCP guardrail translation. It supports scannable text fields only. The native handler can skip a response that has no scannable text and can omit binary or multimodal blocks alongside text. This integration therefore does not enforce binary/multimodal/streaming content policy. Route only text tools through this policy and exclude unsupported tools at the gateway. Streaming LLM/tool effects, model-native search, hidden tool network calls and direct network access are outside its guarantees

The normalized list is not the complete MCP envelope: transport metadata, content-block metadata, annotations, binary fields and fields not exposed by LiteLLM's string extraction are not inspected. Scanning this list must not be described as scanning every field in the original response

The result can have existed in process memory or logging structures before the post-call inspection. Disable message/content logging and prompt storage as shown in the sample, do not install callbacks that expose raw tool output, and review your tracing configuration. Tool side effects cannot be undone. The failure message omits the raw text, URLs, credentials and upstream exception details. Normal MCP error responses may retain HTTP200 while indicating `isError`; clients must inspect the JSON-RPC/tool result rather than treating HTTP200 as an allow decision

## Validation

The mapped unit tests execute native pre/post MCP translation with an injected HTTP fixture. They check identity-preserving allowed inputs, original URL query/comma/fragment encoding, both refusal stages, structured-only results, warn, invalid verdicts, incomplete inspection, errors and UTF-8 limits without truncation. Synthetic replies validate integration policy, not the detector's accuracy

An authenticated live proxy run against the actual IsMalicious API and a real LLM provider is still required before asking for maintainer review under this repository's contribution rules. The local MCP/unit tests must not be described as that proof. No pricing, plan counts or detection-accuracy claim is embedded in this example
