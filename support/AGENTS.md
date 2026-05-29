# INSTRUCTIONS FOR LITELLM CUSTOMER SUPPORT DRAFTING

This folder is for tooling that drafts **public-facing LiteLLM customer support replies** (paying users + enterprise prospects). It is a human-in-the-loop drafting workflow, not auto-send.

## Default audience and tone

- **Default audience:** paying LiteLLM Enterprise customers and prospective enterprise customers.
- **Default product assumption:** LiteLLM Proxy (LLM Gateway), Enterprise tier, recent stable version.
- **Tone:** professional, warm, confident, concise; no marketing fluff; no roadmap promises; no blame.
- **Only address OSS vs Enterprise** when the customer asks or self-identifies as OSS.

Full tone and structure rules live in [`.cursor/rules/customer-support.mdc`](../.cursor/rules/customer-support.mdc).

## How to draft (manual)

1. Open the multi-root workspace (`litellm-full.local.code-workspace`) so both `litellm` and `litellm-docs` are indexed.
2. In Ask or Agent mode, paste the customer question. Optionally include: customer segment, version, deployment, provider/model, logs, config.
3. Cursor should pick up the `draft-support-reply` skill ([`.cursor/skills/draft-support-reply/SKILL.md`](../.cursor/skills/draft-support-reply/SKILL.md)) and the support rule via the `support/**` glob.
4. Output is two sections: `=== CUSTOMER REPLY ===` and `=== INTERNAL NOTES ===`. Always have a human review before sending.

## How to draft (single endpoint)

For programmatic use, run the service in [`customer_support_agent.py`](customer_support_agent.py). It exposes:

```
POST /draft-reply
{
  "question": "...",
  "context": "...",                // optional: logs, config, traces
  "customer_segment": "paying"     // optional: "paying" | "prospect" | "oss"
}
```

The service launches a Cursor Cloud Agent with this repo + `litellm-docs` as the multi-repo environment, applies the skill, polls until the draft is ready, and returns:

```
{
  "agent_id": "...",
  "status": "FINISHED",
  "customer_reply": "...",
  "internal_notes": "...",
  "raw_text": "..."
}
```

See [`README.md`](README.md) for setup, auth, and example curl.

## Grounding sources, in priority order

1. **`litellm-docs/`** workspace folder (most authoritative for user-facing answers).
2. **https://docs.litellm.ai** (`@Docs`) — for published wording.
3. **`litellm/`** code (especially `litellm/proxy/`, `litellm/llms/<provider>/`, `litellm/router*`, `litellm/integrations/`) — only to confirm behavior; never paste internal code into the reply.
4. Internal runbooks / Slack channels — out of scope for this folder; reference by name in internal notes only.

## Escalation defaults (must verify before going live)

| Situation | Default escalation in internal notes |
| --------- | ----------------------------------- |
| Confirmed bug, paying customer | File GitHub issue, ping CSM, link the issue in the reply |
| Confirmed bug, prospect | File GitHub issue, do not link until triaged |
| Security disclosure | Do not draft a substantive reply; route to security team |
| Billing / contract | Acknowledge only; route to AE/CSM |
| Doc gap | Open a `litellm-docs` issue/PR; mention "we'll update the docs" in reply only if true |
| Outage / cluster state | Acknowledge + route to on-call; do not speculate |

Replace placeholder addresses (`support@litellm.ai`, "CSM", "AE") with the team's actual contacts before this folder is used in production.

## What this folder is **not**

- Not a customer-facing chat product. It drafts replies for humans to review and send.
- Not a ticketing system. Use Zendesk / Linear / Slack as you do today; paste the question in.
- Not a substitute for on-call. For live incidents, page on-call first.

## When you change tone or structure

If the desired tone changes (e.g. quieter for prospects, more empathetic for outages), update [`.cursor/rules/customer-support.mdc`](../.cursor/rules/customer-support.mdc), not the skill. The skill defines **workflow**; the rule defines **voice**. Keep them separate so the workflow stays stable across tone iterations.
