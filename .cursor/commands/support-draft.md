# Draft support reply (`/support-draft`)

Same as `/support` — **short, email-safe customer draft** (matches Slack bot behavior).

**Output only** the two sections at the end. Customer reply must paste cleanly into Gmail (no `###`, no nested bullets — Gmail turns them into a table on send).

Apply `.cursor/rules/customer-support.mdc` and `.cursor/skills/draft-support-reply/SKILL.md`. Prefer **Ask mode**.

**Customer reply (strict):**

- Wrap the reply in one ```text fenced block.
- Under **350 words**.
- Numbered sections as `1. Title` (plain line, no `###`).
- Single-level `- ` bullets only.
- No bold/italic prose. Inline backticks only for short identifiers.
- No repo paths or Python symbols in customer text.
- Doc URLs on their own lines.

Internal notes: paths, confidence, follow-ups — only below `=== INTERNAL NOTES ===`. End with: "paste into Gmail with Cmd+Shift+V".

Default: **paying** / Enterprise proxy.

Paste the customer question in this message (or the next).

````
=== CUSTOMER REPLY ===
```text
...
```

=== INTERNAL NOTES ===
...
````
