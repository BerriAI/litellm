---
sidebar_position: 8
---

# Import a marketplace agent

The dashboard's **Agent Marketplace** view (`/agents` tab → Marketplace)
fetches agent definitions from an external gateway and lets admins
import them with one click.

## How it works

1. Dashboard reads `GET /v1/xct-marketplace/config` →
   `{gateway_url: "https://xct-agents-production.up.railway.app"}`
2. Dashboard fetches `${gateway_url}/agents` → returns a JSON array of
   `{slug, name, description, category, emoji, system_prompt?}`
3. Operator clicks **Import** on a row
4. Dashboard fetches the full detail at `${gateway_url}/agents/{slug}/`
5. POSTs to `/v1/agents` with `agent_type="a2a"` + the upstream URL +
   the parsed agent_card_params

## Change the marketplace URL

```http
PUT /v1/xct-marketplace/config
{"gateway_url": "https://my-agent-source.example/api"}
```
Cached for 60s in-memory; multi-pod proxies hot-reload via the next
cache miss.

## Without the dashboard (curl)

```bash
# 1. Fetch agent metadata
curl https://xct-agents-production.up.railway.app/agents/agent-rosalind/ \
  > /tmp/agent.json

# 2. Import into the proxy
curl -X POST https://api.xct.test/v1/agents \
  -H "Authorization: Bearer sk-admin" \
  -H "Content-Type: application/json" \
  -d @/tmp/agent.json
```

After import the agent goes through the same A2A invocation path as
hand-rolled rows; the version is recorded as `version 1` in
`LiteLLM_AgentVersionTable`.
