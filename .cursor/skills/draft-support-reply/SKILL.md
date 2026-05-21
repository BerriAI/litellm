---
name: draft-support-reply
description: Draft a public-facing LiteLLM customer support reply (paying customer or enterprise prospect) from a pasted question and optional context. Defaults to LiteLLM Enterprise. Produces a customer-ready reply plus internal notes.
---

# Draft Support Reply (v1)

Use this skill when a teammate pastes a **customer question** (with optional context like logs, config, version, customer segment) and asks for a draft reply.

Apply the [customer-support](../../rules/customer-support.mdc) rule for tone, structure, and grounding. This skill defines the **workflow** to produce the draft.

## Inputs to look for in the user's message

- **Question / issue** (required) — what the customer asked or what's broken.
- **Customer segment** — one of: `paying`, `prospect`, `oss`. Default `paying` (LiteLLM Enterprise).
- **Version** — LiteLLM version if mentioned.
- **Deployment** — Proxy (LLM Gateway) vs SDK. Default Proxy.
- **Provider/model** — OpenAI, Anthropic, Bedrock, etc., if mentioned.
- **Logs / errors / config snippets** — paste-in context.
- **Tone overrides** — e.g. "the customer is frustrated, lead with empathy".

If the question is unclear or critical fields are missing, **list the missing info in internal notes** but still produce a best-effort draft.

## Workflow

1. **Classify** the question in one short phrase:
   - how-to / config / error-triage / feature-availability / billing-or-licensing / oss-vs-enterprise / unclear
2. **Search the docs first** (`litellm-docs` workspace folder and/or https://docs.litellm.ai via @Docs):
   - Pick 1–3 most relevant doc pages.
3. **Confirm in code if behavior is in question.** Look in `litellm/` (e.g. `litellm/proxy/`, `litellm/llms/<provider>/`, `litellm/router*`). Do not expose code internals in the reply; use them to verify.
4. **Draft the customer reply** following the rule:
   acknowledge → direct answer → steps → why (only if it prevents the next ticket) → docs links → next step.
5. **Write internal notes**: what you checked, confidence (high/medium/low), open questions, suggested follow-ups (CSM ping, bug filing, doc gap).
6. **Output** both sections using this format:

```
=== CUSTOMER REPLY ===
<reply>

=== INTERNAL NOTES ===
- Classification: <one of the categories above>
- Sources checked:
  - litellm-docs/<path>
  - litellm/<path>
  - https://docs.litellm.ai/<path>
- Confidence: high | medium | low (reason)
- Open questions for reviewer: ...
- Suggested follow-ups: ...
```

## Customer-segment defaults

| Segment | Default assumption | Tone shift |
| ------- | ------------------ | ---------- |
| `paying` (default) | LiteLLM Enterprise; has support channel / CSM | confident, direct, link to enterprise docs |
| `prospect` | Evaluating Enterprise; not in production yet | educational; offer technical call; never oversell |
| `oss` | Self-hosted OSS | factual; mention Enterprise alternative only if the user asked or it materially solves the problem |

## Guardrails

- If the customer pasted **secrets** (API keys, tokens, full DB URLs), redact them in the reply and add an internal note recommending rotation.
- Do not commit to **roadmap dates** or **pricing**.
- If you cannot confirm an answer from docs + code, **say so** in the reply ("let me confirm with the team") and mark confidence `low`.
- If the request is **billing, legal, security disclosure, or account access**, draft a short acknowledgment and route to the appropriate team in internal notes — do not answer the substance.

## When to ask back instead of drafting

Ask the customer (in the reply) for more info only when the question is genuinely unanswerable without it. Be specific: ask for **one** of the following at a time:

- Full error and last 20 lines of proxy log (redacted)
- `litellm --version`
- Minimal `config.yaml` (redacted)
- The model name and provider you're hitting

Otherwise: draft a best-effort reply and put clarifying questions in **internal notes**, not in the customer reply.
