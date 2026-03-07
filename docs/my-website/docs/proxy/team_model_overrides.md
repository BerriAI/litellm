import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Team Model Overrides (Per-Member Model Access)

Grant different team members access to different models within the same team.

By default, all team members share the same `team.models` list. With **team model overrides**, you can:

- Set `default_models` on a team — a base set of models every member gets
- Set per-member `models` overrides — additional models for specific members
- A member's **effective models** = `default_models` + their personal `models` override

:::info

This feature is gated behind a feature flag. Set this in your environment or proxy config:

```
LITELLM_TEAM_MODEL_OVERRIDES=true
```

Teams that don't use `default_models` or member overrides are completely unaffected.

:::

## Quick Start

### 1. Enable the feature flag

<Tabs>
<TabItem value="env" label="Environment Variable">

```bash
export LITELLM_TEAM_MODEL_OVERRIDES=true
```

</TabItem>

<TabItem value="yaml" label="Proxy Config (YAML)">

```yaml
environment_variables:
  LITELLM_TEAM_MODEL_OVERRIDES: "true"
```

</TabItem>
</Tabs>

### 2. Create a team with `default_models`

`default_models` is the base set of models all team members get access to. It must be a subset of `team.models`.

```shell
curl -X POST 'http://0.0.0.0:4000/team/new' \
  -H 'Authorization: Bearer <your-master-key>' \
  -H 'Content-Type: application/json' \
  -d '{
    "team_alias": "ml-team",
    "models": ["gpt-4o", "gpt-4o-mini", "claude-sonnet"],
    "default_models": ["gpt-4o"]
  }'
```

| Field | Description |
| --- | --- |
| `models` | Full list of models the team is allowed to use (same as existing behavior) |
| `default_models` | Subset of `models` that every member gets by default |

:::info

`default_models` must be a subset of `team.models`. Setting `default_models: ["claude-opus"]` when `models: ["gpt-4o"]` will return a 400 error.

:::

### 3. Add a member to the team

```shell
curl -X POST 'http://0.0.0.0:4000/team/member_add' \
  -H 'Authorization: Bearer <your-master-key>' \
  -H 'Content-Type: application/json' \
  -d '{
    "team_id": "<team-id>",
    "member": {"role": "user", "user_id": "user-1"}
  }'
```

At this point, `user-1` can access the `default_models` (`gpt-4o`).

### 4. Set per-member model overrides

Grant `user-1` access to `claude-sonnet` in addition to the team defaults:

```shell
curl -X POST 'http://0.0.0.0:4000/team/member_update' \
  -H 'Authorization: Bearer <your-master-key>' \
  -H 'Content-Type: application/json' \
  -d '{
    "team_id": "<team-id>",
    "user_id": "user-1",
    "models": ["claude-sonnet"]
  }'
```

Now `user-1`'s **effective models** = `default_models` + member `models` = `["gpt-4o", "claude-sonnet"]`.

:::info

Member `models` must be a subset of `team.models`. You cannot grant a member access to models the team itself doesn't have.

:::

### 5. Generate a key and make requests

```shell
curl -X POST 'http://0.0.0.0:4000/key/generate' \
  -H 'Authorization: Bearer <your-master-key>' \
  -H 'Content-Type: application/json' \
  -d '{
    "team_id": "<team-id>",
    "user_id": "user-1"
  }'
```

The key will automatically inherit the user's effective models.

<Tabs>
<TabItem label="Allowed (gpt-4o)" value="allowed-default">

`gpt-4o` is in `default_models` — allowed for all team members.

```shell
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
  -H 'Authorization: Bearer <user-key>' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

</TabItem>

<TabItem label="Allowed (claude-sonnet)" value="allowed-member">

`claude-sonnet` is in this member's `models` override — allowed for this specific user.

```shell
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
  -H 'Authorization: Bearer <user-key>' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "claude-sonnet",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

</TabItem>

<TabItem label="Blocked (gpt-4o-mini)" value="blocked">

`gpt-4o-mini` is in `team.models` but NOT in this member's effective models — blocked.

```shell
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
  -H 'Authorization: Bearer <user-key>' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Expected error:
```json
{
  "error": {
    "message": "model not allowed for this key",
    "type": "key_model_access_denied",
    "code": "401"
  }
}
```

</TabItem>
</Tabs>

## How It Works

```
team.models         = [gpt-4o, gpt-4o-mini, claude-sonnet]   # full team allowlist
team.default_models = [gpt-4o]                                 # base for all members

member.models       = [claude-sonnet]                          # per-user override

effective_models    = default_models ∪ member.models
                    = [gpt-4o, claude-sonnet]                   # what this member can use
```

| Concept | Description |
| --- | --- |
| `team.models` | The full set of models the team can access. Acts as the ceiling — no member can exceed this. |
| `team.default_models` | Base models every member gets. Must be a subset of `team.models`. |
| `member.models` | Additional models for a specific member. Must be a subset of `team.models`. |
| **Effective models** | `default_models ∪ member.models` — the actual models a member can use at runtime. |

## Key Generation Behavior

When generating a key for a team member:

| Scenario | Behavior |
| --- | --- |
| Key with no `models` specified | Key inherits the member's effective models |
| Key with `models` that are a subset of effective models | Key is created with only the requested models |
| Key with `models` outside effective models | Returns 403 — model not available for this user |

```shell
# This works — gpt-4o is in effective models
curl -X POST 'http://0.0.0.0:4000/key/generate' \
  -H 'Authorization: Bearer <your-master-key>' \
  -H 'Content-Type: application/json' \
  -d '{
    "team_id": "<team-id>",
    "user_id": "user-1",
    "models": ["gpt-4o"]
  }'

# This fails 403 — gpt-4o-mini is NOT in effective models
curl -X POST 'http://0.0.0.0:4000/key/generate' \
  -H 'Authorization: Bearer <your-master-key>' \
  -H 'Content-Type: application/json' \
  -d '{
    "team_id": "<team-id>",
    "user_id": "user-1",
    "models": ["gpt-4o-mini"]
  }'
```

## Budget Preservation

Updating a member's `models` does not affect their budget or rate limits. You can safely update models without worrying about losing existing budget configuration:

```shell
# Set budget
curl -X POST 'http://0.0.0.0:4000/team/member_update' \
  -H 'Authorization: Bearer <your-master-key>' \
  -H 'Content-Type: application/json' \
  -d '{
    "team_id": "<team-id>",
    "user_id": "user-1",
    "max_budget_in_team": 100.0,
    "tpm_limit": 5000
  }'

# Update models later — budget is preserved
curl -X POST 'http://0.0.0.0:4000/team/member_update' \
  -H 'Authorization: Bearer <your-master-key>' \
  -H 'Content-Type: application/json' \
  -d '{
    "team_id": "<team-id>",
    "user_id": "user-1",
    "models": ["claude-sonnet", "gpt-4o"]
  }'
```

## Teams Without Overrides

Teams that don't set `default_models` or member `models` are **completely unaffected**. The existing `team.models` behavior continues to work as before — all members share the same model list.

```shell
# Normal team — no overrides, existing behavior preserved
curl -X POST 'http://0.0.0.0:4000/team/new' \
  -H 'Authorization: Bearer <your-master-key>' \
  -H 'Content-Type: application/json' \
  -d '{
    "team_alias": "simple-team",
    "models": ["gpt-4o", "gpt-4o-mini"]
  }'
```

All members of this team can access `gpt-4o` and `gpt-4o-mini` — no change from existing behavior.

## Database Migration

The feature adds two new columns:

| Table | Column | Type | Default |
| --- | --- | --- | --- |
| `LiteLLM_TeamTable` | `default_models` | `String[]` | `[]` |
| `LiteLLM_TeamMembership` | `models` | `String[]` | `[]` |

- **Default deployments**: Columns are auto-created on startup via `prisma db push`.
- **`disable_prisma_schema_update: true` deployments**: Apply the migration manually before enabling the feature flag:

```sql
ALTER TABLE "LiteLLM_TeamTable" ADD COLUMN IF NOT EXISTS "default_models" TEXT[] DEFAULT ARRAY[]::TEXT[];
ALTER TABLE "LiteLLM_TeamMembership" ADD COLUMN IF NOT EXISTS "models" TEXT[] DEFAULT ARRAY[]::TEXT[];
```
