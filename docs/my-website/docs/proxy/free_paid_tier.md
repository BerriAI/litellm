# 💸 Free, Paid Tier Routing

Route Virtual Keys on `free tier` to cheaper models

### 1. Define free, paid tier models on config.yaml 

:::info
Requests with `model=gpt-4` will be routed to either `openai/fake` or `openai/gpt-4o` depending on which tier the virtual key is on
:::

```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/fake
      api_key: fake-key
      api_base: https://exampleopenaiendpoint-production.up.railway.app/
    model_info:
      tier: free # 👈 Key Change - set `tier to paid or free`
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
    model_info:
      tier: paid # 👈 Key Change - set `tier to paid or free`

general_settings: 
  master_key: sk-1234 
```

### 2. Create Virtual Keys with pricing `tier=free`

```shell
curl --location 'http://0.0.0.0:4000/key/generate' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
        "metadata": {"tier": "free"}
}'
```

### 3. Make Request with Key on `Free Tier`

```shell
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-inxzoSurQsjog9gPrVOCcA" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "user", "content": "Hello, Claude gm!"}
    ]
  }'
```

**Expected Response**

If this worked as expected then `x-litellm-model-api-base` should be `https://exampleopenaiendpoint-production.up.railway.app/` in the response headers

```shell
x-litellm-model-api-base: https://exampleopenaiendpoint-production.up.railway.app/

{"id":"chatcmpl-657b750f581240c1908679ed94b31bfe","choices":[{"finish_reason":"stop","index":0,"message":{"content":"\n\nHello there, how may I assist you today?","role":"assistant","tool_calls":null,"function_call":null}}],"created":1677652288,"model":"gpt-3.5-turbo-0125","object":"chat.completion","system_fingerprint":"fp_44709d6fcb","usage":{"completion_tokens":12,"prompt_tokens":9,"total_tokens":21}}%
```


### 4. Create Virtual Keys with pricing `tier=paid`

```shell
curl --location 'http://0.0.0.0:4000/key/generate' \
        --header 'Authorization: Bearer sk-1234' \
        --header 'Content-Type: application/json' \
        --data '{
            "metadata": {"tier": "paid"}
    }'
```

### 5. Make Request with Key on `Paid Tier`

```shell
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-mnJoeSc6jFjzZr256q-iqA" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "user", "content": "Hello, Claude gm!"}
    ]
  }'
```

**Expected Response**

If this worked as expected then `x-litellm-model-api-base` should be `https://api.openai.com` in the response headers

```shell
x-litellm-model-api-base: https://api.openai.com

{"id":"chatcmpl-9mW75EbJCgwmLcO0M5DmwxpiBgWdc","choices":[{"finish_reason":"stop","index":0,"message":{"content":"Good morning! How can I assist you today?","role":"assistant","tool_calls":null,"function_call":null}}],"created":1721350215,"model":"gpt-4o-2024-05-13","object":"chat.completion","system_fingerprint":"fp_c4e5b6fa31","usage":{"completion_tokens":10,"prompt_tokens":12,"total_tokens":22}}
```
