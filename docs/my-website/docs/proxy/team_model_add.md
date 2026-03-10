# ✨ Allow Teams to Add Models

:::info

This is an Enterprise feature.
[Enterprise Pricing](https://www.litellm.ai/#pricing)

[Contact us here to get a free trial](https://calendly.com/d/cx9p-5yf-2nm/litellm-introductions)

:::

Allow team to add a their own models/key for that project - so any OpenAI call they make uses their OpenAI key.

Useful for teams that want to call their own finetuned models.

## Specify Team ID in `/model/add` endpoint


```bash
curl -L -X POST 'http://0.0.0.0:4000/model/new' \
-H 'Authorization: Bearer sk-******2ql3-sm28WU0tTAmA' \ # 👈 Team API Key (has same 'team_id' as below)
-H 'Content-Type: application/json' \
-d '{
  "model_name": "my-team-model", # 👈 Call LiteLLM with this model name
  "litellm_params": {
    "model": "openai/gpt-4o",
    "custom_llm_provider": "openai",
    "api_key": "******ccb07",
    "api_base": "https://my-endpoint-sweden-berri992.openai.azure.com",
    "api_version": "2023-12-01-preview"
  },
  "model_info": {
    "team_id": "e59e2671-a064-436a-a0fa-16ae96e5a0a1" # 👈 Specify the team ID it belongs to
  }
}'

```

## Set Team Model Rate Limits on Registration

You can set RPM/TPM for a specific team model when creating it via `/model/new`.

Use `model_info.team_model_rpm_limit` and `model_info.team_model_tpm_limit`:

```bash
curl -L -X POST 'http://0.0.0.0:4000/model/new' \
-H 'Authorization: Bearer sk-******2ql3-sm28WU0tTAmA' \
-H 'Content-Type: application/json' \
-d '{
  "model_name": "my-team-model",
  "litellm_params": {
    "model": "openai/gpt-4o",
    "custom_llm_provider": "openai",
    "api_key": "******ccb07"
  },
  "model_info": {
    "team_id": "e59e2671-a064-436a-a0fa-16ae96e5a0a1",
    "team_model_rpm_limit": 60,
    "team_model_tpm_limit": 120000
  }
}'
```

### How team model rate limits are enforced

- Limits are applied per `{team_id}:{model}` bucket.
- Usage from **all keys/team members** for that team model is aggregated into the same bucket.
- If a team has multiple models with different limits, each model gets its own independent team-level bucket.


## Patch Existing Team Model Rate Limits

You can update team model rate limits on existing models via:

- `PATCH /model/{model_id}/update` (recommended)
- `POST /model/update` (legacy)

:::note Removal semantics

- `team_model_rpm_limit` / `team_model_tpm_limit` are **upsert-style** on update.
- Passing `null` (or omitting a field) does **not** remove an existing per-team-model limit.
- Limits are removed automatically when the team model is deleted.

If you need to remove a specific existing limit without deleting the model, update the team's
`metadata.model_rpm_limit` / `metadata.model_tpm_limit` map via team management endpoints.

:::

### PATCH example

```bash
curl -L -X PATCH 'http://0.0.0.0:4000/model/<model_id>/update' \
-H 'Authorization: Bearer sk-******2ql3-sm28WU0tTAmA' \
-H 'Content-Type: application/json' \
-d '{
  "model_info": {
    "team_id": "e59e2671-a064-436a-a0fa-16ae96e5a0a1",
    "team_model_rpm_limit": 80,
    "team_model_tpm_limit": 180000
  }
}'
```

### Legacy `/model/update` example

```bash
curl -L -X POST 'http://0.0.0.0:4000/model/update' \
-H 'Authorization: Bearer sk-******2ql3-sm28WU0tTAmA' \
-H 'Content-Type: application/json' \
-d '{
  "model_info": {
    "id": "<model_id>",
    "team_id": "e59e2671-a064-436a-a0fa-16ae96e5a0a1",
    "team_model_rpm_limit": 80,
    "team_model_tpm_limit": 180000
  },
  "litellm_params": {
    "model": "openai/gpt-4o"
  }
}'
```

## Test it! 

```bash
curl -L -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-******2ql3-sm28WU0tTAmA' \ # 👈 Team API Key
-d '{
  "model": "my-team-model", # 👈 team model name
  "messages": [
    {
      "role": "user",
      "content": "What's the weather like in Boston today?"
    }
  ]
}'

```

## Debugging

### 'model_name' not found 

Check if model alias exists in team table. 

```bash
curl -L -X GET 'http://localhost:4000/team/info?team_id=e59e2671-a064-436a-a0fa-16ae96e5a0a1' \
-H 'Authorization: Bearer sk-******2ql3-sm28WU0tTAmA' \
```

**Expected Response:**

```json
{
    {
    "team_id": "e59e2671-a064-436a-a0fa-16ae96e5a0a1",
    "team_info": {
        ...,
        "litellm_model_table": {
            "model_aliases": {
                "my-team-model": # 👈 public model name "model_name_e59e2671-a064-436a-a0fa-16ae96e5a0a1_e81c9286-2195-4bd9-81e1-cf393788a1a0" 👈 internally generated model name (used to ensure uniqueness)
            },
            "created_by": "default_user_id",
            "updated_by": "default_user_id"
        }
    },
}
```

