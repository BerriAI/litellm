# Draft support reply (`/support-draft`)

Alias for the LiteLLM customer support drafting workflow (same as Slack `/support-draft`).

Draft a **public-facing LiteLLM customer support reply** for a paying customer or enterprise prospect.

## Apply these project artifacts

- Rule: `.cursor/rules/customer-support.mdc` (tone, structure, Enterprise default)
- Skill: `.cursor/skills/draft-support-reply/SKILL.md` (workflow and output format)
- Scope: `support/AGENTS.md` (grounding and escalation)

Search **both** `litellm` and `litellm-docs` (and https://docs.litellm.ai if @Docs is available) before answering.

## Defaults

- **Segment:** `paying` (LiteLLM Enterprise) unless I specify otherwise below.
- **Deployment:** LiteLLM Proxy (LLM Gateway) unless I say SDK.
- Do **not** compare OSS vs Enterprise unless I ask.

## My input

If I did not paste a customer question in this message, ask me once for:

1. The customer question (required)
2. Optional: version, provider/model, logs or config (redacted), segment (`paying` | `prospect` | `oss`), tone notes

## Output (required format)

Produce exactly these two sections — no PRs, no file edits:

```
=== CUSTOMER REPLY ===
<reply ready to copy-paste>

=== INTERNAL NOTES ===
- Classification: ...
- Sources checked: ...
- Confidence: high | medium | low (reason)
- Open questions for reviewer
- Suggested follow-ups
```

Reminder: this is a **draft** for human review before sending to the customer.
