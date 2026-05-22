# LiteLLM customer support drafting endpoint (v1)

A single HTTP endpoint that drafts a LiteLLM customer support reply from a pasted question. The drafting itself runs in a **Cursor Cloud Agent** that has both this repo and `litellm-docs` cloned, with the support rule and skill applied. A human always reviews and sends.

See [`AGENTS.md`](AGENTS.md) for tone, structure, and escalation policy.

## Invoke in Cursor (slash command)

In the Cursor chat or Agent panel, type **`/`** and choose:

- `/support` (shortest)
- `/support-draft` (matches Slack)
- `/draft-support-reply`

Then paste the customer question (and optional context). The command loads the support rule and skill automatically.

Sources: [`.cursor/commands/`](../.cursor/commands/) — reload the window after `git pull` if commands do not appear.

## Pasting drafts into Gmail

The customer reply is wrapped in a `text` fenced block on purpose: Cursor's chat panel renders chat as HTML, and a direct copy carries `<ul>`, `<h3>`, `<strong>` tags. Gmail accepts that HTML on paste, but its **outbound HTML normalizer** rewrites repeated `<h3>` + `<ul>` groups as a `<table>` when you hit **Send** — that is the "looks fine in the compose box, breaks after send" failure mode (column layout).

To avoid it:

1. In the Cursor chat panel, click the **Copy** button on the `text` fenced block (top-right of the code block). This copies as plain text.
2. In Gmail, paste with **Cmd+Shift+V** on Mac, **Ctrl+Shift+V** on Windows / Linux. This forces "paste without formatting" even if your clipboard happens to carry rich text.
3. Send normally.

The agent is configured to write the reply with **no `###` headers**, **no nested bullets**, and **no bold for prose** — those are the specific Markdown features that trigger Gmail's table rewrite. If you ever see a draft that includes them, that's a regression in the rule/skill — file an issue.

For Slack, regular paste of the same text works fine; the HTTP endpoint and Slack bot strip the outer fence automatically.

## Shareable bundle

Both the rule and the skill are exported as a single self-contained markdown at [`exports/customer-support-bundle.md`](exports/customer-support-bundle.md). Share that file with colleagues who don't have the repo open — it reads cleanly in Notion, Slack, or any markdown viewer, and includes instructions for applying it as a Cursor rule + skill, or as a system prompt for other LLM tooling.

Regenerate after editing either source file:

```bash
./scripts/export_support_bundle.sh
```

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
uv run python -m support.customer_support_agent
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

## Slack integration

The same service can be called from Slack via a slash command or a "Draft support reply" shortcut. The Slack handler is **only mounted if** `SLACK_BOT_TOKEN` and `SLACK_SIGNING_SECRET` are set and `slack-bolt` is installed.

### 1. Install `slack-bolt`

```bash
pip install "slack-bolt" aiohttp
```

`aiohttp` is required for the async Bolt app used here. Both are intentionally not in `litellm`'s core dependencies.

### 2. Create the Slack app

1. Go to [api.slack.com/apps](https://api.slack.com/apps) -> **Create New App** -> **From scratch**.
2. **OAuth & Permissions -> Bot Token Scopes**, add at minimum:
   - `commands`
   - `chat:write`
   - `chat:write.public`
   - `im:write`
3. **Slash Commands -> Create New Command**:
   - Command: `/support-draft`
   - Request URL: `https://<your-host>/slack/commands`
   - Short description: `Draft a LiteLLM customer support reply`
   - Usage hint: `<paste customer question, or run with no args to open a form>`
4. **Interactivity & Shortcuts** -> **Enable Interactivity**:
   - Request URL: `https://<your-host>/slack/interactions`
   - **Create Shortcut -> Global**:
     - Name: `Draft support reply`
     - Callback ID: `draft_support_reply_global`
   - **Create Shortcut -> On messages**:
     - Name: `Draft support reply`
     - Callback ID: `draft_support_reply_msg`
5. **Event Subscriptions** (optional, leave off for v1 unless you add `@mention` handlers later):
   - Request URL: `https://<your-host>/slack/events`
6. **Install App** to your workspace; copy the **Bot User OAuth Token** (`xoxb-...`) and the **Signing Secret** from **Basic Information**.

### 3. Run

```bash
export SLACK_BOT_TOKEN=xoxb-...
export SLACK_SIGNING_SECRET=...
export CURSOR_API_KEY=...           # or LITELLM_PROXY_URL + LITELLM_PROXY_API_KEY
uv run python -m support.customer_support_agent
```

On startup you should see `Slack handler mounted at /slack/{events,commands,interactions}` in the logs.

### 4. Expose publicly (dev)

Slack must reach your service. For local testing, use a tunnel:

```bash
ngrok http 8088
# or: cloudflared tunnel --url http://localhost:8088
```

Set the public URL in the Slack app config for slash command, interactivity, and (optionally) events. For production, put the service behind your ingress / load balancer with TLS.

### 5. Use it in Slack

- **Slash command (one-shot):**
  `/support-draft Our proxy returns 429 from Anthropic despite per-key RPM limits - what's going on?`
  Posts an ephemeral "Drafting..." right away, then the bot posts the full draft to the same channel when ready.
- **Slash command (form):**
  `/support-draft` (no text) opens a modal with question, context, customer segment, and tone override.
- **Message shortcut:**
  Hover any customer message -> `...` -> **Draft support reply**. The modal opens with the message pre-filled in **Context**.
- **Global shortcut:**
  Top-level Slack search -> `Draft support reply`. Opens the modal anywhere.

The draft is posted as two clearly separated sections (`Customer reply (draft)` and `Internal notes`) with a footer showing the Cursor `agent_id` and a reminder that **human review is required before sending**.

### Access control (who can call the bot)

By default, **any member of your Slack workspace** can invoke the bot once it is installed: the slash command appears for everyone, and both shortcuts are visible workspace-wide. Slack itself does not gate slash commands or shortcuts by user or channel.

Two env vars let you restrict access without touching code. Both are comma-separated lists of Slack IDs and are checked **before** any drafting happens:

| Variable | What it allows | Where to find IDs |
| -------- | -------------- | ----------------- |
| `SUPPORT_AGENT_SLACK_ALLOWED_USERS` | Only these users can invoke any handler | Slack profile -> More -> Copy member ID (starts with `U`) |
| `SUPPORT_AGENT_SLACK_ALLOWED_CHANNELS` | Only these channels can host slash commands and message shortcuts | Channel details -> About -> Channel ID (starts with `C` or `G`) |
| `SUPPORT_AGENT_SLACK_BLOCK_GLOBAL_SHORTCUT` | Set to `1` to also block the global shortcut / DM path when channel allowlist is set | — |

Behavior:

- Both lists empty: **open to the workspace** (current default).
- Only user list set: any channel is fine, but only listed users can invoke.
- Only channel list set: any user in those channels can invoke. The global shortcut still works for anyone (set `SUPPORT_AGENT_SLACK_BLOCK_GLOBAL_SHORTCUT=1` to deny it).
- Both lists set: requester must satisfy both.

Denied callers see an ephemeral `:lock:` message (slash command), a DM (shortcut), or a modal validation error (modal submission), and the denial is logged with `user`, `channel`, and reason.

Example: restrict to the internal `#litellm-support-drafts` channel and the on-call group:

```bash
export SUPPORT_AGENT_SLACK_ALLOWED_USERS=U01AAA,U01BBB,U01CCC
export SUPPORT_AGENT_SLACK_ALLOWED_CHANNELS=C09SUPPORT
export SUPPORT_AGENT_SLACK_BLOCK_GLOBAL_SHORTCUT=1
```

Other layers worth combining with this:

- **Slack admin console** can disable a slash command for specific users or roles at the workspace level.
- **Private channels** require the bot to be invited (`/invite @<bot>`) before it can post — that gives Slack-side control over which private channels the bot can reach.
- The `chat:write.public` scope only lets the bot post in **public** channels without an invite; remove it if you want bot replies to stay in invited channels only.
- For a managed allowlist tied to a Slack user group (e.g. `@litellm-support`), add the `usergroups:read` scope and resolve the group's members on startup (or every N minutes) into `SUPPORT_AGENT_SLACK_ALLOWED_USERS`. Not in v1.1 — open an issue if you want this.

### Slack-specific notes

- **Async pattern.** The agent run takes tens of seconds to minutes. The slash command and modal submission both `ack` within Slack's 3-second window; the actual draft is posted via `chat.postMessage` when ready.
- **Where the reply lands.** Slash command and message shortcut reply in the same channel/thread. Global shortcut DMs the invoking user (because no channel context exists).
- **Truncation.** Long drafts are chunked into multiple section blocks so they don't hit Slack's 3000-char per-block limit.
- **Auth.** Slack signing-secret verification is enforced by Bolt. The `/draft-reply` HTTP route still has no auth - put it behind your ingress or proxy.
- **Privacy.** Anything pasted into Slack goes to Cursor Cloud Agents during drafting. Treat it as Cursor-visible data and redact secrets first.

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
