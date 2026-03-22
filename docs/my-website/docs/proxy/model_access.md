import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Restrict Model Access

## **Restrict models by Virtual Key**

Set allowed models for a key using the `models` param


```shell
curl 'http://0.0.0.0:4000/key/generate' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data-raw '{"models": ["gpt-3.5-turbo", "gpt-4"]}'
```

:::info

This key can only make requests to `models` that are `gpt-3.5-turbo` or `gpt-4`

:::

Verify this is set correctly by 

<Tabs>
<TabItem label="Allowed Access" value = "allowed">

```shell
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "user", "content": "Hello"}
    ]
  }'
```

</TabItem>

<TabItem label="Disallowed Access" value = "not-allowed">

:::info

Expect this to fail since gpt-4o is not in the `models` for the key generated

:::

```shell
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "user", "content": "Hello"}
    ]
  }'
```

</TabItem>

</Tabs>


### [API Reference](https://litellm-api.up.railway.app/#/key%20management/generate_key_fn_key_generate_post)

## **Restrict models by `team_id`**
`litellm-dev` can only access `azure-gpt-3.5`

**1. Create a team via `/team/new`**
```shell
curl --location 'http://localhost:4000/team/new' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data-raw '{
  "team_alias": "litellm-dev",
  "models": ["azure-gpt-3.5"]
}' 

# returns {...,"team_id": "my-unique-id"}
```

**2. Create a key for team**
```shell
curl --location 'http://localhost:4000/key/generate' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data-raw '{"team_id": "my-unique-id"}'
```

**3. Test it**
```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
    --header 'Content-Type: application/json' \
    --header 'Authorization: Bearer sk-qo992IjKOC2CHKZGRoJIGA' \
    --data '{
        "model": "BEDROCK_GROUP",
        "messages": [
            {
                "role": "user",
                "content": "hi"
            }
        ]
    }'
```

```shell
{"error":{"message":"Invalid model for team litellm-dev: BEDROCK_GROUP.  Valid models for team are: ['azure-gpt-3.5']\n\n\nTraceback (most recent call last):\n  File \"/Users/ishaanjaffer/Github/litellm/litellm/proxy/proxy_server.py\", line 2298, in chat_completion\n    _is_valid_team_configs(\n  File \"/Users/ishaanjaffer/Github/litellm/litellm/proxy/utils.py\", line 1296, in _is_valid_team_configs\n    raise Exception(\nException: Invalid model for team litellm-dev: BEDROCK_GROUP.  Valid models for team are: ['azure-gpt-3.5']\n\n","type":"None","param":"None","code":500}}%            
```         

### [API Reference](https://litellm-api.up.railway.app/#/team%20management/new_team_team_new_post)

## **Per-Member Model Overrides (Team-Scoped Defaults)**

:::info

Requires `TEAM_MODEL_OVERRIDES=true` environment variable or `litellm.team_model_overrides_enabled = True`.

:::

By default, every team member can access all models in `team.models`. With per-member model overrides, you can:

- Set **`default_models`** on a team — the models every member gets by default
- Set **`models`** on individual team members — additional models only they can access

A member's **effective models** = `default_models` ∪ `member.models`. If neither is set, falls back to `team.models` (full backward compatibility).

### Enable the Feature

Add to your `config.yaml`:

```yaml
environment_variables:
  TEAM_MODEL_OVERRIDES: "true"
```

### 1. Create a Team with Default Models

```shell
curl -L 'http://localhost:4000/team/new' \
  -H 'Authorization: Bearer <your-master-key>' \
  -H 'Content-Type: application/json' \
  -d '{
    "team_alias": "engineering",
    "models": ["gpt-4", "gpt-4o-mini", "gpt-4o"],
    "default_models": ["gpt-4o-mini"]
  }'
```

- `models` — the full pool of models the team is allowed to use
- `default_models` — the subset every member gets by default (must be a subset of `models`)

### 2. Add Members with Per-User Overrides

```shell
# Alice gets the default (gpt-4o-mini only)
curl -L 'http://localhost:4000/team/member_add' \
  -H 'Authorization: Bearer <your-master-key>' \
  -H 'Content-Type: application/json' \
  -d '{
    "team_id": "<team-id>",
    "member": {"role": "user", "user_id": "alice"}
  }'

# Bob gets gpt-4o in addition to the default
curl -L 'http://localhost:4000/team/member_add' \
  -H 'Authorization: Bearer <your-master-key>' \
  -H 'Content-Type: application/json' \
  -d '{
    "team_id": "<team-id>",
    "member": {"role": "user", "user_id": "bob", "models": ["gpt-4o"]}
  }'
```

| Member | Override | Effective Models |
|--------|----------|-----------------|
| Alice | none | `["gpt-4o-mini"]` |
| Bob | `["gpt-4o"]` | `["gpt-4o-mini", "gpt-4o"]` |

### 3. Generate Keys and Test

```shell
# Generate key for Bob
curl -L 'http://localhost:4000/key/generate' \
  -H 'Authorization: Bearer <your-master-key>' \
  -H 'Content-Type: application/json' \
  -d '{"team_id": "<team-id>", "user_id": "bob"}'
```

<Tabs>
<TabItem label="Allowed (Bob → gpt-4o)" value="allowed">

```shell
curl -L 'http://localhost:4000/chat/completions' \
  -H 'Authorization: Bearer <bob-key>' \
  -H 'Content-Type: application/json' \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]}'
```

Returns `200 OK` — `gpt-4o` is in Bob's effective set.

</TabItem>
<TabItem label="Denied (Bob → gpt-4)" value="denied">

```shell
curl -L 'http://localhost:4000/chat/completions' \
  -H 'Authorization: Bearer <bob-key>' \
  -H 'Content-Type: application/json' \
  -d '{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}'
```

Returns `401 Unauthorized` — `gpt-4` is in the team pool but not in Bob's effective set.

</TabItem>
</Tabs>

### 4. Update Member Overrides

```shell
# Add gpt-4 to Bob's overrides
curl -L 'http://localhost:4000/team/member_update' \
  -H 'Authorization: Bearer <your-master-key>' \
  -H 'Content-Type: application/json' \
  -d '{
    "team_id": "<team-id>",
    "user_id": "bob",
    "models": ["gpt-4o", "gpt-4"]
  }'

# Remove all overrides (Bob falls back to default_models only)
curl -L 'http://localhost:4000/team/member_update' \
  -H 'Authorization: Bearer <your-master-key>' \
  -H 'Content-Type: application/json' \
  -d '{
    "team_id": "<team-id>",
    "user_id": "bob",
    "models": []
  }'
```

### Validation Rules

| Rule | Error |
|------|-------|
| `default_models` must be a subset of `team.models` | `400` on `/team/new` and `/team/update` |
| Member `models` must be a subset of `team.models` | `400` on `/team/member_add` and `/team/member_update` |
| Key `models` must be a subset of effective models | `403` on `/key/generate` |
| Narrowing `team.models` auto-prunes stale `default_models` | Automatic on `/team/update` |

### Important Notes

- **Effective models are capped by `team.models`**: If `team.models` is later narrowed, any member overrides outside the new pool are silently excluded at runtime — no access is granted beyond the team's allowed list.
- **Service/bot keys** (keys without a `user_id`) are **not affected** by this feature. They always use the full `team.models` pool, preserving backward compatibility.
- **Access groups** (`access_group_ids`) are a team-level concept and are **not** restricted by per-member overrides. Models granted via access groups remain available to all team members.

### Backward Compatibility

When the feature flag is off **or** when neither `default_models` nor member `models` is configured:

- `get_effective_team_models()` returns `team.models` unchanged
- All existing teams and keys work exactly as before
- Zero extra database queries on the auth hot path


## **View Available Fallback Models**

Use the `/v1/models` endpoint to discover available fallback models for a given model. This helps you understand which backup models are available when your primary model is unavailable or restricted.

:::info Extension Point

The `include_metadata` parameter serves as an extension point for exposing additional model metadata in the future. While currently focused on fallback models, this approach will be expanded to include other model metadata such as pricing information, capabilities, rate limits, and more.

:::

### Basic Usage

Get all available models:

```shell
curl -X GET 'http://localhost:4000/v1/models' \
  -H 'Authorization: Bearer <your-api-key>'
```

### Get Fallback Models with Metadata

Include metadata to see fallback model information:

```shell
curl -X GET 'http://localhost:4000/v1/models?include_metadata=true' \
  -H 'Authorization: Bearer <your-api-key>'
```

### Get Specific Fallback Types

You can specify the type of fallbacks you want to see:

<Tabs>
<TabItem value="general" label="General Fallbacks">

```shell
curl -X GET 'http://localhost:4000/v1/models?include_metadata=true&fallback_type=general' \
  -H 'Authorization: Bearer <your-api-key>'
```

General fallbacks are alternative models that can handle the same types of requests.

</TabItem>

<TabItem value="context_window" label="Context Window Fallbacks">

```shell
curl -X GET 'http://localhost:4000/v1/models?include_metadata=true&fallback_type=context_window' \
  -H 'Authorization: Bearer <your-api-key>'
```

Context window fallbacks are models with larger context windows that can handle requests when the primary model's context limit is exceeded.

</TabItem>

<TabItem value="content_policy" label="Content Policy Fallbacks">

```shell
curl -X GET 'http://localhost:4000/v1/models?include_metadata=true&fallback_type=content_policy' \
  -H 'Authorization: Bearer <your-api-key>'
```

Content policy fallbacks are models that can handle requests when the primary model rejects content due to safety policies.

</TabItem>

</Tabs>

### Example Response

When `include_metadata=true` is specified, the response includes fallback information:

```json
{
  "data": [
    {
      "id": "gpt-4",
      "object": "model",
      "created": 1677610602,
      "owned_by": "openai",
      "fallbacks": {
        "general": ["gpt-3.5-turbo", "claude-3-sonnet"],
        "context_window": ["gpt-4-turbo", "claude-3-opus"],
        "content_policy": ["claude-3-haiku"]
      }
    }
  ]
}
```

### Use Cases

- **High Availability**: Identify backup models to ensure service continuity
- **Cost Optimization**: Find cheaper alternatives when primary models are expensive
- **Content Filtering**: Discover models with different content policies
- **Context Length**: Find models that can handle larger inputs
- **Load Balancing**: Distribute requests across multiple compatible models

### API Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `include_metadata` | boolean | Include additional model metadata including fallbacks |
| `fallback_type` | string | Filter fallbacks by type: `general`, `context_window`, or `content_policy` |

## Advanced: Model Access Groups

For advanced use cases, use [Model Access Groups](./model_access_groups) to dynamically group multiple models and manage access without restarting the proxy.

## [Role Based Access Control (RBAC)](./jwt_auth_arch)