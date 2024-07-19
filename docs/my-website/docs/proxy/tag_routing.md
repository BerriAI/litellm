# 💸 Tag Based Routing

Route requests based on tags

### 1. Define free, paid tier models on config.yaml 

```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/fake
      api_key: fake-key
      api_base: https://exampleopenaiendpoint-production.up.railway.app/
      tags: ["free"]
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
      tags: ["paid"]

general_settings: 
  master_key: sk-1234 
```

### Make Request with Key on `Free Tier`

```shell
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-4",
    "metadata": {"tags": ["paid"]},
    "messages": [
      {"role": "user", "content": "Hello, Claude gm!"}
    ]
  }'
```
