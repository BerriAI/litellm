---
sidebar_position: 9
---

# Upload a skill ZIP

Bundle a skill as a ZIP and POST it.

## Archive layout

```
my-skill.zip
├── manifest.yaml     REQUIRED
├── SKILL.md          system_prompt_template (Jinja2 or {format})
├── tools.json        OpenAI-shape tool array (optional)
└── README.md         description fallback (optional)
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
```

## Upload

Dashboard: **Skills** → **Upload ZIP** → drop archive.

Or curl:
```bash
curl -X POST https://api.xct.test/v1/xct-skills/upload \
  -H "Authorization: Bearer sk-..." \
  -F "file=@my-skill.zip" \
  -F "is_public_override=true"   # optional — overrides manifest.is_public
```

## Validations on upload (will fail-fast)

- ZIP ≤ 10 MB (compressed AND uncompressed sums; zip-bomb guard)
- no symlinks (S_IFLNK in external_attr is rejected)
- no path traversal (`..` segments / absolute paths)
- `manifest.yaml` parses + has non-empty `display_title`
- `tools.json` is a JSON array of objects

## After upload

Row created with `source="custom"`. To make it pinnable via
`skills=["fact-check@v3"]`:

```http
POST /v1/xct-skills/{skill_id}/publish
```

Publishing freezes the content fields (`system_prompt_template`,
`tool_schema`, `instructions`, `file_content`). Display title / description /
is_public stay mutable.
