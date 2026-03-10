import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Qohash QAIGS

[Qohash](https://qohash.com/) is a data security company that helps enterprises locate, classify, and protect sensitive data — including as organizations adopt generative AI. QAIGS (Qohash AI Guardrail Server) is a deterministic guardrail service that scans prompts and model outputs for sensitive data using classification policies and LLM-as-a-judge checks, returning an explicit enforcement decision (ALLOW, REPORT, REDACT, WARN or BLOCK).

:::info
QAIGS is not a public offering. To inquire about access, visit [qohash.com](https://qohash.com).
:::

## Quick Start

### 1. Deploy QAIGS

Run QAIGS as a container with your policy config mounted:

```bash
docker run --rm \
  -p 8800:8800 \
  -v $(pwd)/qaigs.yaml:/etc/qaigs/config.yaml \
  qohash/qaigs:latest
```

Verify it's ready:

```bash
curl -i http://localhost:8800/health
# Expected: HTTP/1.1 200 OK
```

:::note
Additional deployment options are available (cloud marketplace, on-premises). [Contact Qohash](https://qohash.com) for details.
:::

### 2. Configure LiteLLM Proxy (config.yaml)

**Pre-call** — block sensitive data before it reaches the model:

```yaml title="config.yaml (pre-call)"
guardrails:
  - guardrail_name: "qohash-qaigs-pre-call"
    litellm_params:
      guardrail: qohash_qaigs
      api_key: os.environ/QAIGS_API_KEY
      api_base: http://qaigs:8800
      mode: "pre_call"
      default_on: true
```

**Post-call** — redact or block sensitive data in model output before it reaches the caller:

```yaml title="config.yaml (post-call)"
guardrails:
  - guardrail_name: "qohash-qaigs-post-call"
    litellm_params:
      guardrail: qohash_qaigs
      api_key: os.environ/QAIGS_API_KEY
      api_base: http://qaigs:8800
      mode: "post_call"
      default_on: true
```

### 3. Start LiteLLM Gateway

```bash
export QAIGS_API_KEY="<your-qaigs-api-key>"
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
    "guardrails": ["qohash-qaigs-pre-call"]
  }'
```

Expected: QAIGS returns `BLOCK` → LiteLLM returns an error, no provider request is made.

</TabItem>
<TabItem label="WARN" value="warn">

Send a prompt with data that triggers a `WARN` policy. Without `override: true`, the request is blocked:

```bash
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your_litellm_key>" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "My employee ID is 123456 and my phone is 555-0100"}
    ],
    "guardrails": ["qohash-qaigs-pre-call"]
  }'
```

To proceed after a warning, resubmit with `override: true`. Use the `guardrail_name` you set in `config.yaml` as the key:

```bash
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your_litellm_key>" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "My employee ID is 123456 and my phone is 555-0100"}
    ],
    "guardrails": [{"qohash-qaigs-pre-call": {"extra_body": {"override": true}}}]
  }'
```

Note: `WARN` is only applicable for `pre_call`.

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
    ]
  }'
```

Expected: QAIGS returns `REDACT` → LiteLLM forwards a masked prompt to the provider. Response headers include `x-qaigs-outcome-decision: REDACT`.

**Post-call** — sensitive content in the model response is masked before it reaches the caller:

```bash
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your_litellm_key>" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "Return my credit card number: 5555555555554444."}
    ]
  }'
```

Expected: QAIGS returns `REDACT` → LiteLLM returns the response with masked output.

</TabItem>
<TabItem label="REPORT" value="report">

Send a prompt with low-sensitivity data that triggers a `REPORT` policy (request continues):

```bash
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your_litellm_key>" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "My employee ID is 123456 (test) and my phone is 555-0100"}
    ]
  }'
```

Expected: QAIGS returns `REPORT` → LiteLLM forwards to the provider, response returns normally with decision headers.

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
    ]
  }'
```

Expected: QAIGS returns `ALLOW` → LiteLLM forwards to the provider normally.

</TabItem>
</Tabs>

## Decisions

QAIGS returns one decision per request:

| Decision | Request continues? | Description |
|---|---|---|
| `ALLOW` | Yes | No policy violation detected |
| `REPORT` | Yes | Violation logged; request proceeds with outcome metadata |
| `REDACT` | Yes (masked) | Sensitive substrings replaced in the payload before forwarding |
| `WARN` | Blocked until override | Request blocked; caller can resubmit with `override: true` (pre-call only) |
| `BLOCK` | No | Request fails; no provider call is made (for pre-call) |

## Supported Parameters

| Parameter | Type | Description |
|---|---|---|
| `guardrail` | string | Must be `qohash_qaigs` |
| `api_key` | string | API key for QAIGS (use `os.environ/VAR_NAME` for env vars) |
| `api_base` | string | Base URL of your QAIGS instance (e.g. `http://qaigs:8800`) |
| `mode` | string | `pre_call` (scan prompt) or `post_call` (scan model output) |
| `default_on` | boolean | Apply this guardrail to all requests by default |

## Request Identifiers

QAIGS requires correlation identifiers on every request. These identifiers are never used to access content — they carry only metadata that attributes detections to the right user, session, and context.

Pass them via `extra_body` in the `guardrails` field, using your `guardrail_name` as the key:

```bash
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your_litellm_key>" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "..."}
    ],
    "guardrails": [
      {
        "qohash-qaigs-pre-call": {
          "extra_body": {
            "identifiers": {
              "trace": "<request-or-session-id>",
              "source": "<application-id>",
              "container": "<conversation-id>",
              "identity": "<user-email-or-upn>"
            }
          }
        }
      }
    ]
  }'
```

| Identifier | Description |
|---|---|
| `trace` | Unique ID for the request or session, used for correlation across events |
| `source` | The application or integration sending the request (e.g. app ID, service name) |
| `container` | The conversation or thread context (e.g. conversation ID) |
| `identity` | The end-user identity (e.g. email or UPN), used for user-level attribution |

These fields are required in all deployment modes. Their effect depends on the operating mode:

### Qohash Qostodian

[Qostodian](https://qohash.com/qostodian/) is the Qohash data security posture management (DSPM) platform. It monitors high-risk unstructured data across your organization, providing visibility into sensitive data exposure, behavioral analytics, and governance workflows. When QAIGS operates in connected or advanced mode, identifiers are forwarded to Qostodian to correlate AI detections with broader data security activity across users, sessions, and applications.

| Mode | Effect |
|---|---|
| Basic standalone | Identifiers appear in structured log output for traceability |
| Basic connected | Connects to Qohash Qostodian — identifiers are used for display and attribution |
| Advanced (platform) | Connects to Qohash Qostodian — identifiers unlock full DSPM capabilities: activity correlation, behavioral profiling, and governance workflows |

## Security Guidance

QAIGS operates on a **zero-copy, data-sovereign processing model** in all deployment modes: content is analyzed in-memory and never persisted or transmitted to Qohash. Only metadata (detection outcomes, policy decisions, identifiers) is reported — prompt and response content stays within your infrastructure at all times.

- **Use TLS** between LiteLLM and QAIGS in production environments
- **Authenticate calls** using mTLS (preferred) or bearer token
- **Deploy QAIGS in customer-controlled infrastructure** (on-premises or cloud tenant) to ensure data stays within your security boundary
