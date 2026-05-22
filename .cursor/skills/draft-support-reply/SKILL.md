---
name: draft-support-reply
description: Draft a short LiteLLM customer support reply that pastes cleanly into Gmail and Slack. Defaults to Enterprise. Output is two sections; customer reply is wrapped in a ```text fence so the Cursor "Copy" button gives plain text.
---

# Draft Support Reply (v1.2)

Use when a teammate needs a **sendable** customer draft — not a technical write-up.

Apply [customer-support.mdc](../../rules/customer-support.mdc). The rule defines voice and **email-safe formatting**; this skill defines the workflow.

## Mode

- Prefer **Ask** (read-only). In **Agent** mode, do not surface todo lists or narrate file reads — your **last message only** is the two-section output.
- Do **not** edit files, open PRs, or append summaries after INTERNAL NOTES.

## Workflow

1. **Classify** (one line, internal notes only): how-to | config | error-triage | feature-availability | billing | oss-vs-enterprise | multi-topic
2. **Search docs first**, then code only to confirm.
3. **Write CUSTOMER REPLY** inside a `text` fence — under 350 words, plain text with light formatting only (no `###`, no nested bullets, no bold for prose). See rule for the full banned list.
4. **Write INTERNAL NOTES** — paths, confidence, open questions, follow-ups (no fence).

## Why the ```text fence

Cursor renders chat as HTML. Copying directly from chat carries `<ul>` / `<h3>` / `<strong>` tags that Gmail re-flows into a table layout when you hit Send. Wrapping the customer reply in a `text` fenced block:

- Surfaces Cursor's per-block **Copy** button, which copies the raw text only.
- Lets Slack / HTTP consumers strip the fence automatically (parser handles it).
- Forces you to write plain text — the fence syntax discourages embedded `###` and `**bold**`.

Even with the fence, **tell reviewers in internal notes**: in Gmail, paste with **Cmd+Shift+V** (Ctrl+Shift+V on Windows/Linux) to bypass rich-text paste.

## Multi-topic input

If the paste has 2+ questions (JWT, Helm, Langfuse, etc.):

- Customer reply: numbered sections written as `1. Title` (a plain line, no `###`).
- At most **4 bullets per topic**, single-level, `- ` prefix.
- One doc link per topic, on its own line.
- No nested lists, no per-topic code blocks.

## Output (entire final message = only this)

````
=== CUSTOMER REPLY ===
```text
<short, copy-paste, email-safe reply>
```

=== INTERNAL NOTES ===
- Classification: ...
- Sources: ...
- Confidence: ...
- Open questions: ...
- Follow-ups: ...
- Reviewer tip: paste into Gmail with Cmd+Shift+V to keep plain text
````

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
