# Tag Regex Routing

Route requests to specific deployments based on regex patterns matched against request headers — without requiring per-user tag configuration.

## Overview

With standard [tag-based routing](tag_routing), each request must carry a matching tag (e.g. `tags: ["vibe-coding"]`). This works well when you control the client, but becomes impractical at scale.

**Tag regex routing** lets you match on headers the client already sends automatically — like `User-Agent` — so requests are routed correctly with zero client-side configuration.

### Use case: route all Claude Code traffic to dedicated AWS accounts

> "We need to route Claude Code traffic to a dedicated set of AWS accounts. We want to roll out the Claude Code → LiteLLM integration to 5,000 employees — it's not practical to ask every developer to configure a tag. Claude Code always sends a `User-Agent` header that starts with `claude-code/`, so we'd like LiteLLM to use that automatically."

This is exactly what `tag_regex` is for.

---

## Quick Start

### 1. Configure `tag_regex` on the target deployment

Add a `tag_regex` list to `litellm_params`. Each entry is a regex pattern matched against `"Header-Name: value"` strings built from the request metadata.

```yaml
model_list:
  # Claude Code traffic → dedicated Bedrock account, matched by User-Agent
  - model_name: claude-sonnet
    litellm_params:
      model: bedrock/converse/anthropic-claude-sonnet-4-6
      aws_region_name: us-east-1
      aws_role_name: arn:aws:iam::111122223333:role/LiteLLMRole
      tag_regex:
        - "^User-Agent: claude-code\\/"   # matches claude-code/1.x, claude-code/2.x, …
    model_info:
      id: claude-code-deployment

  # All other traffic → standard deployment (catch-all default)
  - model_name: claude-sonnet
    litellm_params:
      model: bedrock/converse/anthropic-claude-sonnet-4-6
      aws_region_name: us-east-1
      aws_role_name: arn:aws:iam::444455556666:role/LiteLLMRole
      tags:
        - default
    model_info:
      id: regular-deployment

router_settings:
  enable_tag_filtering: true
  tag_filtering_match_any: true

general_settings:
  master_key: sk-1234
```

### 2. Start the proxy

```shell
litellm --config config.yaml
```

### 3. Send a request from Claude Code

Claude Code automatically sets `User-Agent: claude-code/<version>`. No extra configuration needed on the client side.

```shell
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-1234" \
  -H "User-Agent: claude-code/1.2.3" \
  -d '{
    "model": "claude-sonnet",
    "messages": [{"role": "user", "content": "hello"}]
  }'
```

Check the response header to confirm routing:

```
x-litellm-model-id: claude-code-deployment
```

### 4. Send a request from any other client

No `User-Agent: claude-code/` header → falls through to the `default` deployment.

```shell
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "claude-sonnet",
    "messages": [{"role": "user", "content": "hello"}]
  }'
```

```
x-litellm-model-id: regular-deployment
```

---

## How it works

When `enable_tag_filtering: true` is set and a deployment has `tag_regex` configured, LiteLLM builds a `"Header-Name: value"` string from the request's `User-Agent` and tests each regex pattern against it using `re.search`.

**Matching priority (in order):**

1. **Exact tag match** — if the request includes `tags: ["vibe-coding"]` and a deployment has `tags: ["vibe-coding"]`, that match fires first.
2. **Regex match** — if no exact tag match, `tag_regex` patterns are tested against request headers.
3. **Default fallback** — if nothing matches, deployments tagged `default` are used.
4. **All deployments** — if no `default` tag exists, all healthy deployments are returned (existing behaviour unchanged).

**Backwards compatibility:** deployments that use only plain `tags` (no `tag_regex`) are unaffected, even when requests carry a `User-Agent` header.

---

## Combining `tags` and `tag_regex`

You can mix both on the same deployment — a request matches if either the tag or the regex matches:

```yaml
- model_name: claude-sonnet
  litellm_params:
    model: bedrock/converse/anthropic-claude-sonnet-4-6
    aws_role_name: arn:aws:iam::111122223333:role/LiteLLMRole
    tags:
      - vibe-coding          # explicit tag still works for teams that set it
    tag_regex:
      - "^User-Agent: claude-code\\/"  # automatic match for everyone else
```

---

## Observability: `tag_routing` in SpendLogs

When a regex matches, LiteLLM writes a `tag_routing` block into the request metadata. This flows automatically into SpendLogs so you can see how each request was routed:

```json
{
  "tag_routing": {
    "matched_deployment": "claude-sonnet",
    "matched_via": "tag_regex",
    "matched_value": "^User-Agent: claude-code\\/",
    "user_agent": "claude-code/1.2.3",
    "request_tags": []
  }
}
```

| Field | Description |
|-------|-------------|
| `matched_via` | `"tag_regex"` or `"tags"` |
| `matched_value` | The specific pattern or tag that matched |
| `user_agent` | The `User-Agent` value from the request |
| `request_tags` | Explicit tags on the request (if any) |

---

## Reference

### `tag_regex` field

| | |
|-|-|
| **Location** | `litellm_params` in `config.yaml` |
| **Type** | `list[str]` |
| **Matching** | `re.search(pattern, "User-Agent: <value>")` |
| **Error handling** | Invalid regex patterns are skipped with a warning at startup and at match time |

### Supported header sources

Currently `User-Agent` is the only header source. The pattern format is `"Header-Name: value"`, so for a request with `User-Agent: claude-code/1.2.3` the string tested is `"User-Agent: claude-code/1.2.3"`.

### Pattern tips

| Goal | Pattern |
|------|---------|
| Match any Claude Code version | `^User-Agent: claude-code\/` |
| Match specific major version | `^User-Agent: claude-code\/1\.` |
| Match any semver | `^User-Agent: claude-code\/\d+\.\d+` |

---

## Related

- [Tag Based Routing](tag_routing) — explicit per-request tags
- [Team Based Routing](team_based_routing) — route by team membership
- [Request Tags](request_tags) — how tags flow through requests
