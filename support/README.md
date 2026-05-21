# LiteLLM customer support drafting endpoint (v1)

A single HTTP endpoint that drafts a LiteLLM customer support reply from a pasted question. The drafting itself runs in a **Cursor Cloud Agent** that has both this repo and `litellm-docs` cloned, with the support rule and skill applied. A human always reviews and sends.

See [`AGENTS.md`](AGENTS.md) for tone, structure, and escalation policy.

## Setup

### 1. Cursor Cloud Agents environment (one time, in the dashboard)

Open [Cloud Agents → Environments](https://cursor.com/dashboard/cloud-agents#environments) and create (or pick) a multi-repo environment that clones **both**:

- `BerriAI/litellm`
- `BerriAI/litellm-docs`

The skill and rule live in `BerriAI/litellm` (this repo), so the agent will pick them up automatically when it boots into a checkout of this repo.

### 2. Cursor API key

Get an API key from your Cursor account / team admin and either:

- Set `CURSOR_API_KEY` directly, **or**
- Route through your existing LiteLLM proxy's built-in Cursor pass-through (`/cursor/*`):

  ```bash
  export LITELLM_PROXY_URL=https://your-litellm-proxy.example.com
  export LITELLM_PROXY_API_KEY=sk-...
  ```

  The litellm proxy must have a `cursor` credential configured (UI → LLM Credentials, or `CURSOR_API_KEY` env on the proxy itself).

### 3. Install Python deps

The endpoint uses `fastapi`, `uvicorn`, `httpx`. These are already in litellm's dev environment:

```bash
uv sync --group proxy-dev --extra proxy
```

### 4. Run the service

```bash
uv run python support/customer_support_agent.py
```

Default port `8088` (override with `SUPPORT_AGENT_PORT`).

## API

### `POST /draft-reply`

Request:

```json
{
  "question": "Our proxy is returning 429 from Anthropic even though we set per-key RPM limits. What's going on?",
  "context": "litellm v1.x, Enterprise, proxy logs show retry-after=10s. config.yaml uses router fallbacks.",
  "customer_segment": "paying",
  "tone_override": "customer is frustrated — lead with empathy"
}
```

Fields:

| Field | Required | Notes |
| ----- | -------- | ----- |
| `question` | yes | The customer question or issue |
| `context` | no  | Logs, config snippets, version, provider, etc. (redact secrets first) |
| `customer_segment` | no | `paying` (default) \| `prospect` \| `oss` |
| `tone_override` | no | Reviewer nudge, e.g. "frustrated", "first ticket from this customer" |

Response:

```json
{
  "agent_id": "ag_...",
  "status": "FINISHED",
  "customer_reply": "Thanks for flagging this...",
  "internal_notes": "- Classification: error-triage\n- Sources checked: ...",
  "raw_text": "=== CUSTOMER REPLY ===\n...\n=== INTERNAL NOTES ===\n..."
}
```

If the agent output doesn't match the expected format, `customer_reply` / `internal_notes` are `null` and the raw text is in `raw_text`.

### `GET /healthz`

Liveness check. Returns `{"status": "ok"}`.

## Example

```bash
curl -X POST http://localhost:8088/draft-reply \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I enable SSO with Okta on the LiteLLM proxy admin UI?",
    "customer_segment": "prospect"
  }'
```

## Configuration

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `CURSOR_API_KEY` | — | Cursor Cloud Agents API key (Basic auth). |
| `CURSOR_API_BASE` | `https://api.cursor.com` | Override Cursor API base. |
| `LITELLM_PROXY_URL` | — | If set, calls go to `<proxy>/cursor/...` instead of api.cursor.com. |
| `LITELLM_PROXY_API_KEY` | — | Bearer token for the litellm proxy. |
| `SUPPORT_AGENT_REPO` | `https://github.com/BerriAI/litellm` | Repo the Cursor agent should clone. |
| `SUPPORT_AGENT_REF` | `main` | Git ref / branch to run on. |
| `SUPPORT_AGENT_MODEL` | — | Optional Cursor model id. |
| `SUPPORT_AGENT_POLL_INTERVAL` | `5` | Seconds between status polls. |
| `SUPPORT_AGENT_TIMEOUT` | `600` | Seconds before giving up. |
| `SUPPORT_AGENT_PORT` | `8088` | HTTP port for the service. |

## Limitations (v1)

- **Cursor Cloud Agents API is still maturing.** Some users have reported intermittent 500s on `POST /v0/agents`. The endpoint surfaces those as `502 Bad Gateway` with the raw Cursor error.
- **No streaming.** v1 polls until the agent finishes. Average turnaround is on the order of tens of seconds to a few minutes.
- **No authn on the wrapper itself.** Put this behind your existing auth (proxy, ingress, or `LITELLM_PROXY_*` env vars).
- **No long-term storage of drafts.** Add logging / DB persistence if you want a paper trail. (Cursor stores the agent conversation itself.)
- **Human review required.** This drafts replies; it does not send them.
