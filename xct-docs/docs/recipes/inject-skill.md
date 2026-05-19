---
sidebar_position: 3
---

# Inject a skill into chat

Pin a skill by version → its system prompt + tools are added server-side.

```python
reply = xct.chat.completions.create(
    model="deepseek-v3.2",
    messages=[{"role": "user", "content": "Is this true: ..."}],
    skills=["fact-check@v3"],
    skill_inputs={"strictness": "high"},   # available in Jinja2 template
)
```

Behavior on the proxy:
1. resolve `fact-check@v3` → 404 if missing, 400 on bare name miss
2. render `system_prompt_template` (Jinja2 if installed, else format-string fallback) with `skill_inputs`
3. prepend / merge as `role:"system"` message
4. concat `tool_schema` into request's `tools` array (dedup by `function.name`)
5. strip `skills` + `skill_inputs` before sending to provider

If an existing `role:"system"` message is at index 0, the skill prompt is
**prepended** to its content — never clobbered.

Multiple skills: combined with `\n\n` separator, in order given.

Spend log row gets `entity_type="skill"`, `entity_id=<first skill id>`.
Metrics + webhooks attribute likewise.
