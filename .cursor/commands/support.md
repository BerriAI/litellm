# Draft support reply (`/support`)

Same as `/draft-support-reply` — **short, email-safe customer draft**.

**Output only** the two sections at the end. Customer reply must paste cleanly into Gmail (Gmail re-flows `###` + nested bullets into a table on send).

Apply `.cursor/rules/customer-support.mdc` and `.cursor/skills/draft-support-reply/SKILL.md`. Prefer **Ask mode**.

**Customer reply (strict):**

- Wrap the reply in one ```text fenced block (Cursor "Copy" button → plain text).
- Under **350 words**.
- Numbered sections as `1. Title` (plain line, **no `###`**).
- Single-level `- ` bullets only.
- No bold/italic for prose. Inline backticks only for short identifiers.
- No repo paths, no Python symbols, no confidence in customer text.
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
