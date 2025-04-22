# ✨ Budget / Rate Limit Tiers

Define tiers with rate limits. Assign them to keys. 

Use this to control access and budgets across a lot of keys.

:::info 

This is a LiteLLM Enterprise feature.

Get a 7 day free trial + get in touch [here](https://litellm.ai/#trial).

See pricing [here](https://litellm.ai/#pricing).

:::


## 1. Create a budget 

```bash
curl -L -X POST 'http://0.0.0.0:4000/budget/new' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{
    "budget_id": "my-test-tier",
    "rpm_limit": 0
}'
```

## 2. Assign budget to a key 

```bash
curl -L -X POST 'http://0.0.0.0:4000/key/generate' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{
    "budget_id": "my-test-tier"
}'
```

Expected Response:

```json
{
    "key": "sk-...",
    "budget_id": "my-test-tier",
    "litellm_budget_table": {
        "budget_id": "my-test-tier",
        "rpm_limit": 0
    }
}
```

## 3. Check if budget is enforced on key 

```bash
curl -L -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-...' \ # 👈 KEY from step 2.
-d '{
    "model": "<REPLACE_WITH_MODEL_NAME_FROM_CONFIG.YAML>",
    "messages": [
      {"role": "user", "content": "hi my email is ishaan"}
    ]
}'
```


## [API Reference](https://litellm-api.up.railway.app/#/budget%20management)

