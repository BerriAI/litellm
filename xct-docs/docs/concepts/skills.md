---
sidebar_position: 5
---

# Skills

A **skill** is a server-stored bundle of:

- a `system_prompt_template` (Jinja2 or `str.format`-style)
- an optional `tool_schema` (OpenAI-shape array)
- arbitrary `xct_metadata` (category, tags, ...)

Inject it into a chat completion with one line:

```python
xct.chat.completions.create(
    model="deepseek-v3.2",
    messages=[{"role": "user", "content": "Is the moon a planet?"}],
    skills=["fact-check@v3"],
    skill_inputs={"strictness": "high"},
)
```

The proxy renders the template against `skill_inputs`, prepends the result
as a `role: "system"` message (or merges with an existing one), and adds
`tool_schema` entries to the request's `tools` array. The `skills` /
`skill_inputs` fields are stripped before the request reaches the provider.

## Anatomy of a skill ZIP

```
my-skill.zip
├── manifest.yaml          REQUIRED — display_title + metadata
├── SKILL.md               system_prompt_template (Jinja2 supported)
├── tools.json             OpenAI-shape tool array (optional)
└── README.md              description fallback (optional)
```

`manifest.yaml`:

```yaml
display_title: "Fact Check"
description: "Verify claims with citations."
version: "3"
is_public: false
category: research
tags: [verification, science]
xct:
  difficulty: hard
  model_recommendations: [gpt-4o, claude-3-7-sonnet]
```

Upload via the dashboard or:

```http
POST /v1/xct-skills/upload
Content-Type: multipart/form-data
file: <zip bytes>
```

Validations: 10 MB cap (compressed + uncompressed), no symlinks, no
path traversal, `manifest.yaml` must include `display_title`.

## Publishing

Published skills are **immutable** on `system_prompt_template`,
`tool_schema`, `instructions`, `file_content`. Subsequent PATCH on a
content field returns **409**. `display_title`, `description`,
`is_public` stay mutable.

```http
POST /v1/xct-skills/{skill_id}/publish
```

Bumps `xct_metadata.published = true` and stamps `published_at` /
`published_by`. Idempotent.

## Version pinning

Reference a specific version via `@`:

```python
skills=["fact-check@v3"]      # version v3 — 404 if not found
skills=["fact-check"]          # latest published
```

Once published, content fields are frozen so `@v3` is a stable snapshot.

## Storage

Rows live in `LiteLLM_SkillsTable` distinguished by `source`:

| `source` | Owner | Notes |
|---|---|---|
| `"custom"` | xct-native | All the fields above |
| `"anthropic"` | Anthropic SDK pass-through | Existing surface, unchanged |

The two never mix — `/v1/xct-skills` only reads `source="custom"`, the
Anthropic pass-through at `/v1/skills?beta=true` only reads
`source="anthropic"`.

## See also

- **[Recipe: inject a skill into chat](../recipes/inject-skill.md)**
- **[Recipe: upload a skill ZIP](../recipes/upload-skill-zip.md)**
