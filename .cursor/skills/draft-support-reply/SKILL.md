---
name: draft-support-reply
description: Draft a short, copy-paste-ready LiteLLM customer support reply (paying customer or enterprise prospect). Defaults to Enterprise. Output is ONLY two sections — customer reply under 350 words, internal notes separate.
---

# Draft Support Reply (v1.1)

Use when a teammate needs a **sendable** customer draft, not a technical write-up.

Apply [customer-support.mdc](../../rules/customer-support.mdc). **Research in the background; ship a short reply.**

## Mode

- Prefer **Ask** (read-only). If in **Agent**, do not create visible todo lists or narrate file reads — your **last message only** is the two-section output.
- Do **not** edit files, open PRs, or append summaries after INTERNAL NOTES.

## Workflow

1. **Classify** (one line, goes in internal notes only): how-to | config | error-triage | feature-availability | billing | oss-vs-enterprise | multi-topic
2. **Search docs first**, then code only to confirm.
3. **Write CUSTOMER REPLY** — under **350 words**, copy-paste ready (see rule).
4. **Write INTERNAL NOTES** — paths, confidence, open questions, follow-ups.

## Multi-topic input

If the paste has 2+ questions (JWT, Helm, Langfuse, etc.):

- Customer reply: `### 1. …` / `### 2. …` with **≤4 bullets each** + doc link per topic.
- Do **not** write long prose per topic.
- Do **not** duplicate internal notes inside the customer reply.

## Output (entire final message = only this)

```
=== CUSTOMER REPLY ===
...

=== INTERNAL NOTES ===
...
```

**Forbidden after INTERNAL NOTES:** "draft is ready", "worth flagging", checkmarks, repeated summaries.

## Segment defaults

| Segment | Default |
| ------- | ------- |
| `paying` | Enterprise proxy customer |
| `prospect` | Evaluating Enterprise; offer a call |
| `oss` | OSS-only framing when user says so |

## Guardrails

- Redact secrets in the customer reply; note rotation in internal notes.
- No roadmap/pricing commitments.
- If unsure, customer reply says "I'll confirm with the team" — detail uncertainty in internal notes only.
