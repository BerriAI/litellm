# Draft support reply (concise, copy-paste)

**Output only** the two sections at the end — no todos, no research narrative, no summary after INTERNAL NOTES.

Apply `.cursor/rules/customer-support.mdc` and `.cursor/skills/draft-support-reply/SKILL.md`. Use **Ask mode** if available.

**CUSTOMER REPLY rules:** under **350 words**, copy-paste ready for Slack/email. No repo paths, no Python symbols, no confidence scores in the customer text. Multiple topics → `### 1.` / `### 2.` with max **4 bullets** each + doc links. One small config block max.

**INTERNAL NOTES:** paths, confidence, open questions, follow-ups — only below `=== INTERNAL NOTES ===`.

Search `litellm` + `litellm-docs` (+ @Docs). Default segment: **paying** (Enterprise).

If no customer question is pasted, ask once for the question + optional version/logs (redacted).

```
=== CUSTOMER REPLY ===
...

=== INTERNAL NOTES ===
...
```
