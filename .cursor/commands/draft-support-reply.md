# Draft support reply (concise, copy-paste, email-safe)

**Output only** the two sections at the end. The customer reply must paste cleanly into Gmail AND Slack — Gmail re-flows nested bullets and `###` headers into a table on send.

Apply `.cursor/rules/customer-support.mdc` and `.cursor/skills/draft-support-reply/SKILL.md`. Use **Ask mode** if available.

**CUSTOMER REPLY rules (strict):**

- Wrap the entire customer reply inside one ```text fenced block so Cursor surfaces a Copy button that gives plain text.
- Under **350 words**.
- Numbered sections as `1. Title` on their own line — **no `###`, no `##`, no `#`**.
- Single-level bullets only (`- `), one blank line between sections.
- No bold `**...**`, no italic `*...*` for prose. Inline backticks OK for short identifiers.
- **At most one** small code fence (≤15 lines) — and only inside the reply if absolutely needed.
- No repo paths, no Python symbols, no confidence scores in the customer text.
- Doc links as plain URLs on their own lines (https://docs.litellm.ai/...).
- No postamble after the body ("draft is ready", "worth flagging").

**INTERNAL NOTES:** paths, confidence, open questions, follow-ups — only below `=== INTERNAL NOTES ===`. End with a one-line reviewer tip: "paste into Gmail with Cmd+Shift+V to keep plain text".

Default segment: **paying** (Enterprise).

If no customer question is pasted, ask once for the question + optional version/logs (redacted).

Output exactly this shape:

````
=== CUSTOMER REPLY ===
```text
...
```

=== INTERNAL NOTES ===
...
````
