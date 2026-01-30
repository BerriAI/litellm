import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Qostodian Nexus by Qohash

[Qohash](https://qohash.com/) is a pioneer of zero-copy data security, the only model designed to secure petabytes of unstructured data in large enterprises. Enterprises run dozens of AI models, copilots, and autonomous agents, all hungry for data. Qostodian Nexus is the single control layer that governs every interaction. It knows your data. It enforces your policies. It scales from prompt inspection to LLM output data governance, interrogating all agentic, human, SaaS and API interactions with one control plane and a consistent set of policies. Nexus scans prompts and responses using deterministic classification policies and LLM-as-a-judge checks, returning an explicit enforcement decision (ALLOW, LOG, REDACT or BLOCK).

:::info
Qostodian Nexus is not a public offering. To inquire about access, visit [qohash.com](https://qohash.com).
:::

## Quick Start

### 1. Deploy Qostodian Nexus

Run Qostodian Nexus as a container with your policy config mounted:

```bash
docker run --rm \
  -p 8800:8800 \
  -v $(pwd)/nexus.yaml:/etc/nexus/config.yaml \
  qohash/nexus:latest
```

Verify it's ready:

```bash
curl -i http://localhost:8800/health
# Expected: HTTP/1.1 200 OK
```

:::note
Additional deployment options are available. [Contact Qohash](https://qohash.com) for details.
:::

### 2. Configure LiteLLM Proxy (config.yaml)

**Pre-call** — block sensitive data before it reaches the model:

```yaml title="config.yaml (pre-call)"
guardrails:
  - guardrail_name: "qostodian-nexus-pre-call"
    litellm_params:
      guardrail: qostodian_nexus
      api_base: http://nexus:8800
      mode: "pre_call"
      default_on: true
```

**Post-call** — redact or block sensitive data in model output before it reaches the caller:

```yaml title="config.yaml (post-call)"
guardrails:
  - guardrail_name: "qostodian-nexus-post-call"
    litellm_params:
      guardrail: qostodian_nexus
      api_base: http://nexus:8800
      mode: "post_call"
      default_on: true
```

### 3. Start LiteLLM Gateway

```bash
litellm --config config.yaml
```

### 4. Test Requests

<Tabs>
<TabItem label="BLOCK" value="block">

Send a prompt containing a credit card number (blocked by a `BLOCK` policy):

```bash
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your_litellm_key>" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "MASTERCARD 5555555555554444 03/2027 123"}
    ],
    "guardrails": ["qostodian-nexus-pre-call"]
  }'
```

Expected: Qostodian Nexus returns `BLOCK` → LiteLLM returns an error, no provider request is made.

</TabItem>
<TabItem label="REDACT" value="redact">

**Pre-call** — sensitive substrings are masked before the prompt reaches the model:

```bash
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your_litellm_key>" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "My credit card is 5555555555554444, please summarize this."}
    ],
    "guardrails": ["qostodian-nexus-pre-call"]
  }'
```

Expected: Qostodian Nexus returns `REDACT` → LiteLLM forwards a masked prompt to the provider. Response headers include `x-qostodian-nexus-outcome-decision: REDACT`.

**Post-call** — sensitive content in the model response is masked before it reaches the caller:

```bash
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your_litellm_key>" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "Return my credit card number: 5555555555554444."}
    ],
    "guardrails": ["qostodian-nexus-post-call"]
  }'
```

Expected: Qostodian Nexus returns `REDACT` → LiteLLM returns the response with masked output.

</TabItem>
<TabItem label="LOG" value="log">

Send a prompt with low-sensitivity data that triggers a `LOG` policy (request continues):

```bash
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your_litellm_key>" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "My employee ID is 123456 (test) and my phone is 555-0100"}
    ],
    "guardrails": ["qostodian-nexus-pre-call", "qostodian-nexus-post-call"]
  }'
```

Expected: Qostodian Nexus returns `LOG` → LiteLLM forwards to the provider, response returns normally with decision headers.

</TabItem>
<TabItem label="ALLOW" value="allow">

Send a benign prompt (no sensitive data detected):

```bash
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your_litellm_key>" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "Summarize the main differences between TCP and UDP."}
    ],
    "guardrails": ["qostodian-nexus-pre-call", "qostodian-nexus-post-call"]
  }'
```

Expected: Qostodian Nexus returns `ALLOW` → LiteLLM forwards to the provider normally.

</TabItem>
</Tabs>

## Decisions

Qostodian Nexus returns one decision per request:

| Decision | Request continues? | Description |
|---|---|---|
| `ALLOW` | Yes | No policy violation detected |
| `LOG` | Yes | Violation logged; request proceeds with outcome metadata |
| `REDACT` | Yes (masked) | Sensitive substrings replaced in the payload before forwarding |
| `BLOCK` | No | Request fails; no provider call is made (for pre-call) |

## Supported Parameters

| Parameter | Type | Description |
|---|---|---|
| `guardrail` | string | Must be `qostodian_nexus` |
| `api_base` | string | Base URL of your Qostodian Nexus instance (e.g. `http://nexus:8800`) |
| `mode` | string | `pre_call` (scan prompt) or `post_call` (scan model output) |
| `default_on` | boolean | Apply this guardrail to all requests by default |

No API key is required for LiteLLM to call Qostodian Nexus. As Qostodian Nexus is designed to be deployed within your infrastructure, you must secure it using network controls.

## Request Identifiers

Qostodian Nexus requires correlation identifiers on every request. These identifiers are never used to access content, they carry only metadata that attributes detections to the right user, session, and context.

Pass them via request headers:

```bash
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your_litellm_key>" \
  -H "x-qostodian-nexus-identifiers-trace: trace-id" \
  -H "x-qostodian-nexus-identifiers-source: source-id" \
  -H "x-qostodian-nexus-identifiers-container: container-id" \
  -H "x-qostodian-nexus-identifiers-identity: identity@example.com" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "..."}
    ],
    "guardrails": ["qostodian-nexus-pre-call", "qostodian-nexus-post-call"]
  }'
```

| Identifier | Description |
|---|---|
| `trace` | Unique ID for the request or session, used for correlation across events |
| `source` | The application or integration sending the request (e.g. app ID, service name) |
| `container` | The conversation or thread context (e.g. conversation ID) |
| `identity` | The end-user identity (e.g. email or UPN), used for user-level attribution |

These fields are required in all deployment modes. Their effect depends on the operating mode:

### Qostodian Platform

[Qostodian](https://qohash.com/qostodian/) is the Qohash data security posture management (DSPM) platform. It monitors high-risk unstructured data across your organization, providing visibility into sensitive data exposure, behavioral analytics, and governance workflows. When Qostodian Nexus operates in connected or advanced mode, identifiers are forwarded to Qostodian to correlate AI detections with broader data security activity across users, sessions, and applications.

| Mode | Effect |
|---|---|
| Basic standalone | Identifiers appear in structured log output for traceability |
| Basic connected | Connects to the Qostodian Platform — identifiers are used for display and attribution |
| Advanced (platform) | Connects to the Qostodian Platform — identifiers unlock full DSPM capabilities: activity correlation, behavioral profiling, and governance workflows |

## Security Guidance

Qostodian Nexus operates on a **zero-copy, data-sovereign processing model** in all deployment modes: content is analyzed in-memory and never persisted or transmitted to Qohash. Only metadata (detection outcomes, policy decisions, identifiers) is reported — prompt and response content stays within your infrastructure at all times.

- **Use TLS** between LiteLLM and Qostodian Nexus in production environments
- **Authenticate calls** using mTLS (preferred) or bearer token
- **Deploy Qostodian Nexus in customer-controlled infrastructure** (on-premises or cloud tenant) to ensure data stays within your security boundary
