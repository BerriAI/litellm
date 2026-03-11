import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# [Beta] Guardrail Policies

Use policies to group guardrails and control which ones run for specific teams, keys, or models.

## Why use policies?

- Enable/disable specific guardrails for teams, keys, or models
- Group guardrails into a single policy
- Inherit from existing policies and override what you need

## Quick Start

<Tabs>
<TabItem value="config" label="config.yaml">

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4

# 1. Define your guardrails
guardrails:
  - guardrail_name: pii_masking
    litellm_params:
      guardrail: presidio
      mode: pre_call

  - guardrail_name: prompt_injection
    litellm_params:
      guardrail: lakera
      mode: pre_call
      api_key: os.environ/LAKERA_API_KEY

# 2. Create a policy
policies:
  my-policy:
    guardrails:
      add:
        - pii_masking
        - prompt_injection

# 3. Attach the policy
policy_attachments:
  - policy: my-policy
    scope: "*"  # apply to all requests
```

</TabItem>
<TabItem value="ui" label="UI (LiteLLM Dashboard)">

**Step 1: Create a Policy**

Go to **Policies** tab and click **+ Create New Policy**. Fill in the policy name, description, and select guardrails to add.

![Enter policy name](https://colony-recorder.s3.amazonaws.com/files/2026-02-11/4ba62cc8-d2c4-4af1-a526-686295466928/ascreenshot_401eab3e2081466e8f4d4ffa3bf7bff4_text_export.jpeg)

![Add a description for the policy](https://colony-recorder.s3.amazonaws.com/files/2026-02-11/51685e47-1d94-4d9c-acb0-3c88dce9f938/ascreenshot_a5cd40066ff34afbb1e4089a3c93d889_text_export.jpeg)

![Select a parent policy to inherit from](https://colony-recorder.s3.amazonaws.com/files/2026-02-11/1d96c3d3-187a-4f7c-97d2-6ac1f093d51e/ascreenshot_8a3af3b2210547dca3d4709df920d005_text_export.jpeg)

![Select guardrails to add to the policy](https://colony-recorder.s3.amazonaws.com/files/2026-02-11/23781274-e600-4d5f-a8a6-4a2a977a166c/ascreenshot_a2a45d2c5d064c77ab7cb47b569ad9e9_text_export.jpeg)

![Click Create Policy to save](https://colony-recorder.s3.amazonaws.com/files/2026-02-11/1d1ae8a8-daa5-451b-9fa2-c5b607ff6220/ascreenshot_218c2dd259714be4aa3c4e1894c96878_text_export.jpeg)

</TabItem>
</Tabs>

Response headers show what ran:

```
x-litellm-applied-policies: my-policy
x-litellm-applied-guardrails: pii_masking,prompt_injection
```

## Add guardrails for a specific team

:::info
✨ Enterprise only feature for team/key-based policy attachments. [Get a free trial](https://www.litellm.ai/enterprise#trial)
:::

You have a global baseline, but want to add extra guardrails for a specific team.

<Tabs>
<TabItem value="config" label="config.yaml">

```yaml showLineNumbers title="config.yaml"
policies:
  global-baseline:
    guardrails:
      add:
        - pii_masking

  finance-team-policy:
    inherit: global-baseline
    guardrails:
      add:
        - strict_compliance_check
        - audit_logger

policy_attachments:
  - policy: global-baseline
    scope: "*"

  - policy: finance-team-policy
    teams:
      - finance  # team alias from /team/new
```

</TabItem>
<TabItem value="ui" label="UI (LiteLLM Dashboard)">

**Option 1: Create a team-scoped attachment**

Go to **Policies** > **Attachments** tab and click **+ Create New Attachment**. Select the policy and the teams to scope it to.

![Select teams for the attachment](https://colony-recorder.s3.amazonaws.com/files/2026-02-11/50e58f54-3bc3-477e-a106-e58cb65fde7e/ascreenshot_85d2e3d9d8d24842baced92fea170427_text_export.jpeg)

![Select the teams to attach the policy to](https://colony-recorder.s3.amazonaws.com/files/2026-02-11/f24066bb-0a73-49fb-87b6-c65ad3ca5b2f/ascreenshot_242476fbdac447309f65de78b0ed9fdd_text_export.jpeg)

**Option 2: Attach from team settings**

Go to **Teams** > click on a team > **Settings** tab > under **Policies**, select the policies to attach.

![Open team settings and click Edit Settings](https://colony-recorder.s3.amazonaws.com/files/2026-02-11/c31c3735-4f9d-4c6a-896b-186e97296940/ascreenshot_4749bb24ce5942cca462acc958fd3822_text_export.jpeg)

![Select policies to attach to this team](https://colony-recorder.s3.amazonaws.com/files/2026-02-11/da8d5d7a-d975-4bfe-acd2-f41dcea29520/ascreenshot_835a33b6cec545cbb2987f017fbaff90_text_export.jpeg)

<Image img={require('../../../img/policy_team_attach.png')} />

</TabItem>
</Tabs>

Now the `finance` team gets `pii_masking` + `strict_compliance_check` + `audit_logger`, while everyone else just gets `pii_masking`.

## Remove guardrails for a specific team

:::info
✨ Enterprise only feature for team/key-based policy attachments. [Get a free trial](https://www.litellm.ai/enterprise#trial)
:::

You have guardrails running globally, but want to disable some for a specific team (e.g., internal testing).

```yaml showLineNumbers title="config.yaml"
policies:
  global-baseline:
    guardrails:
      add:
        - pii_masking
        - prompt_injection

  internal-team-policy:
    inherit: global-baseline
    guardrails:
      remove:
        - pii_masking  # don't need PII masking for internal testing

policy_attachments:
  - policy: global-baseline
    scope: "*"

  - policy: internal-team-policy
    teams:
      - internal-testing  # team alias from /team/new
```

Now the `internal-testing` team only gets `prompt_injection`, while everyone else gets both guardrails.

## Inheritance

Start with a base policy and build on it:

```yaml showLineNumbers title="config.yaml"
policies:
  base:
    guardrails:
      add:
        - pii_masking
        - toxicity_filter

  strict:
    inherit: base
    guardrails:
      add:
        - prompt_injection

  relaxed:
    inherit: base
    guardrails:
      remove:
        - toxicity_filter
```

What you get:
- `base` → `[pii_masking, toxicity_filter]`
- `strict` → `[pii_masking, toxicity_filter, prompt_injection]`
- `relaxed` → `[pii_masking]`

## Model Conditions

Run guardrails only for specific models:

```yaml showLineNumbers title="config.yaml"
policies:
  gpt4-safety:
    guardrails:
      add:
        - strict_content_filter
    condition:
      model: "gpt-4.*"  # regex - matches gpt-4, gpt-4-turbo, gpt-4o

  bedrock-compliance:
    guardrails:
      add:
        - audit_logger
    condition:
      model:  # exact match list
        - bedrock/claude-3
        - bedrock/claude-2
```

## Attachments

Policies don't do anything until you attach them. Attachments tell LiteLLM *where* to apply each policy.

**Global** - runs on every request:

```yaml showLineNumbers title="config.yaml"
policy_attachments:
  - policy: default
    scope: "*"
```

**Team-specific** (uses team alias from `/team/new`):

```yaml showLineNumbers title="config.yaml"
policy_attachments:
  - policy: hipaa-compliance
    teams:
      - healthcare-team  # team alias
      - medical-research  # team alias
```

**Key-specific** (uses key alias from `/key/generate`, wildcards supported):

```yaml showLineNumbers title="config.yaml"
policy_attachments:
  - policy: internal-testing
    keys:
      - "dev-*"  # key alias pattern
      - "test-*"  # key alias pattern
```

**Tag-based** (matches keys/teams by metadata tags, wildcards supported):

```yaml showLineNumbers title="config.yaml"
policy_attachments:
  - policy: hipaa-compliance
    tags:
      - "healthcare"
      - "health-*"  # wildcard - matches health-team, health-dev, etc.
```

Tags are read from key and team `metadata.tags`. For example, a key created with `metadata: {"tags": ["healthcare"]}` would match the attachment above.

## Test Policy Matching

Debug which policies and guardrails apply for a given context. Use this to verify your policy configuration before deploying.

<Tabs>
<TabItem value="ui" label="UI (LiteLLM Dashboard)">

Go to **Policies** > **Test** tab. Enter a team alias, key alias, model, or tags and click **Test** to see which policies match and what guardrails would be applied.

<Image img={require('../../../img/policy_test_matching.png')} />

</TabItem>
<TabItem value="api" label="API">

```bash
curl -X POST "http://localhost:4000/policies/resolve" \
    -H "Authorization: Bearer <your_api_key>" \
    -H "Content-Type: application/json" \
    -d '{
        "tags": ["healthcare"],
        "model": "gpt-4"
    }'
```

Response:

```json
{
    "effective_guardrails": ["pii_masking"],
    "matched_policies": [
        {
            "policy_name": "hipaa-compliance",
            "matched_via": "tag:healthcare",
            "guardrails_added": ["pii_masking"]
        }
    ]
}
```

</TabItem>
</Tabs>

## Config Reference

### `policies`

```yaml
policies:
  <policy-name>:
    description: ...
    inherit: ...
    guardrails:
      add: [...]
      remove: [...]
    condition:
      model: ...
```

| Field | Type | Description |
|-------|------|-------------|
| `description` | `string` | Optional. What this policy does. |
| `inherit` | `string` | Optional. Parent policy to inherit guardrails from. |
| `guardrails.add` | `list[string]` | Guardrails to enable. |
| `guardrails.remove` | `list[string]` | Guardrails to disable (useful with inheritance). |
| `condition.model` | `string` or `list[string]` | Optional. Only apply when model matches. Supports regex. |

### `policy_attachments`

```yaml
policy_attachments:
  - policy: ...
    scope: ...
    teams: [...]
    keys: [...]
    models: [...]
    tags: [...]
```

| Field | Type | Description |
|-------|------|-------------|
| `policy` | `string` | **Required.** Name of the policy to attach. |
| `scope` | `string` | Use `"*"` to apply globally. |
| `teams` | `list[string]` | Team aliases (from `/team/new`). Supports `*` wildcard. |
| `keys` | `list[string]` | Key aliases (from `/key/generate`). Supports `*` wildcard. |
| `models` | `list[string]` | Model names. Supports `*` wildcard. |
| `tags` | `list[string]` | Tag patterns (from key/team `metadata.tags`). Supports `*` wildcard. |

### Response Headers

| Header | Description |
|--------|-------------|
| `x-litellm-applied-policies` | Policies that matched this request |
| `x-litellm-applied-guardrails` | Guardrails that actually ran |
| `x-litellm-policy-sources` | Why each policy matched (e.g., `hipaa=tag:healthcare; baseline=scope:*`) |

## How it works

Example config:

```yaml showLineNumbers title="config.yaml"
policies:
  base:
    guardrails:
      add: [pii_masking]

  finance-policy:
    inherit: base
    guardrails:
      add: [audit_logger]

policy_attachments:
  - policy: base
    scope: "*"
  - policy: finance-policy
    teams: [finance]
```

```mermaid
flowchart TD
    A["Request with team_alias='finance'"] --> B["Matches policies: base, finance-policy"]
    B --> C["Resolves guardrails: pii_masking, audit_logger"]
```

1. Request comes in with `team_alias='finance'`
2. Matches `base` (via `scope: "*"`) and `finance-policy` (via `teams: [finance]`)
3. Resolves guardrails: `base` adds `pii_masking`, `finance-policy` inherits and adds `audit_logger`
4. Final guardrails: `pii_masking`, `audit_logger`
