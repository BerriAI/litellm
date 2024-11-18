# Getting Started - E2E Tutorial

End-to-End tutorial for LiteLLM Proxy to:
- Add an Azure OpenAI model 
- Make a successful /chat/completion call 
- Generate a virtual key 
- Set RPM limit on virtual key 


## Pre-Requisites 

- Install LiteLLM Docker Image 

```
docker pull ghcr.io/berriai/litellm:main-latest
```

[**See all docker images**](https://github.com/orgs/BerriAI/packages)

## 1. Add a model 

Control LiteLLM Proxy with a config.yaml file.

Setup your config.yaml with your azure model.

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/my_azure_deployment
      api_base: os.environ/AZURE_API_BASE
      api_key: "os.environ/AZURE_API_KEY"
      api_version: "2024-07-01-preview" # [OPTIONAL] litellm uses the latest azure api_version by default
```
---

### Model List Specification

- **`model_name`** (`str`) - This field should contain the name of the model as received.
- **`litellm_params`** (`dict`) [See All LiteLLM Params](https://github.com/BerriAI/litellm/blob/559a6ad826b5daef41565f54f06c739c8c068b28/litellm/types/router.py#L222)
    - **`model`** (`str`) - Specifies the model name to be sent to `litellm.acompletion` / `litellm.aembedding`, etc. This is the identifier used by LiteLLM to route to the correct model + provider logic on the backend. 
    - **`api_key`** (`str`) - The API key required for authentication. It can be retrieved from an environment variable using `os.environ/`.
    - **`api_base`** (`str`) - The API base for your azure deployment.
    - **`api_version`** (`str`) - The API Version to use when calling Azure's OpenAI API. Get the latest Inference API version [here](https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation?source=recommendations#latest-preview-api-releases).


### Useful Links
- [**All Supported LLM API Providers (OpenAI/Bedrock/Vertex/etc.)**](../providers/)
- [**Full Config.Yaml Spec**](./configs.md)
- [**Pass provider-specific params**](../completion/provider_specific_params.md#proxy-usage)


## 2. Make a successful /chat/completion call 

LiteLLM Proxy is 100% OpenAI-compatible. Test your azure model via the `/chat/completions` route.

### 2.1 Start Proxy 

Save your config.yaml from step 1. as `litellm_config.yaml`.

```bash
docker run \
    -v $(pwd)/litellm_config.yaml:/app/config.yaml \
    -e AZURE_API_KEY=d6*********** \
    -e AZURE_API_BASE=https://openai-***********/ \
    -p 4000:4000 \
    ghcr.io/berriai/litellm:main-latest \
    --config /app/config.yaml --detailed_debug

# RUNNING on http://0.0.0.0:4000
```

Confirm your config.yaml got mounted correctly

```bash
Loaded config YAML (api_key and environment_variables are not shown):
{
"model_list": [
{
"model_name ...
```

### 2.2 Make Call 


```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful math tutor. Guide the user through the solution step by step."
      },
      {
        "role": "user",
        "content": "how can I solve 8x + 7 = -23"
      }
    ]
}'
```

**Expected Response**

```bash
{
    "id": "chatcmpl-2076f062-3095-4052-a520-7c321c115c68",
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "I am gpt-3.5-turbo",
                "role": "assistant",
                "tool_calls": null,
                "function_call": null
            }
        }
    ],
    "created": 1724962831,
    "model": "gpt-3.5-turbo",
    "object": "chat.completion",
    "system_fingerprint": null,
    "usage": {
        "completion_tokens": 20,
        "prompt_tokens": 10,
        "total_tokens": 30
    }
}
```



### Useful Links
- [All Supported LLM API Providers (OpenAI/Bedrock/Vertex/etc.)](../providers/)
- [Call LiteLLM Proxy via OpenAI SDK, Langchain, etc.](./user_keys.md#request-format)
- [All API Endpoints Swagger](https://litellm-api.up.railway.app/#/chat%2Fcompletions)
- [Other/Non-Chat Completion Endpoints](../embedding/supported_embedding.md)
- [Pass-through for VertexAI, Bedrock, etc.](../pass_through/vertex_ai.md)

## 3. Generate a virtual key

Track Spend, and control model access via virtual keys for the proxy

### 3.1 Set up a Database 

**Requirements**
- Need a postgres database (e.g. [Supabase](https://supabase.com/), [Neon](https://neon.tech/), etc)


```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/my_azure_deployment
      api_base: os.environ/AZURE_API_BASE
      api_key: "os.environ/AZURE_API_KEY"
      api_version: "2024-07-01-preview" # [OPTIONAL] litellm uses the latest azure api_version by default

general_settings: 
  master_key: sk-1234 
  database_url: "postgresql://<user>:<password>@<host>:<port>/<dbname>" # 👈 KEY CHANGE
```

Save config.yaml as `litellm_config.yaml` (used in 3.2).

---

**What is `general_settings`?**

These are settings for the LiteLLM Proxy Server. 

See All General Settings [here](http://localhost:3000/docs/proxy/configs#all-settings).

1. **`master_key`** (`str`)
   - **Description**: 
     - Set a `master key`, this is your Proxy Admin key - you can use this to create other keys (🚨 must start with `sk-`).
   - **Usage**: 
     - ** Set on config.yaml** set your master key under `general_settings:master_key`, example - 
        `master_key: sk-1234`
     - ** Set env variable** set `LITELLM_MASTER_KEY`

2. **`database_url`** (str)
   - **Description**: 
     - Set a `database_url`, this is the connection to your Postgres DB, which is used by litellm for generating keys, users, teams.
   - **Usage**: 
     - ** Set on config.yaml** set your master key under `general_settings:database_url`, example - 
        `database_url: "postgresql://..."`
     - Set `DATABASE_URL=postgresql://<user>:<password>@<host>:<port>/<dbname>` in your env 

### 3.2 Start Proxy 

```bash
docker run \
    -v $(pwd)/litellm_config.yaml:/app/config.yaml \
    -e AZURE_API_KEY=d6*********** \
    -e AZURE_API_BASE=https://openai-***********/ \
    -p 4000:4000 \
    ghcr.io/berriai/litellm:main-latest \
    --config /app/config.yaml --detailed_debug
```


### 3.3 Create Key w/ RPM Limit

Create a key with `rpm_limit: 1`. This will only allow 1 request per minute for calls to proxy with this key.

```bash 
curl -L -X POST 'http://0.0.0.0:4000/key/generate' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{
    "rpm_limit": 1
}
```

[**See full API Spec**](https://litellm-api.up.railway.app/#/key%20management/generate_key_fn_key_generate_post)

**Expected Response**

```bash
{
    "key": "sk-12..."
}
```

### 3.4 Test it! 

**Use your virtual key from step 3.3**

1st call - Expect to work! 

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-12...' \
-d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful math tutor. Guide the user through the solution step by step."
      },
      {
        "role": "user",
        "content": "how can I solve 8x + 7 = -23"
      }
    ]
}'
```

**Expected Response**

```bash
{
    "id": "chatcmpl-2076f062-3095-4052-a520-7c321c115c68",
    "choices": [
        ...
}
```

2nd call - Expect to fail! 

**Why did this call fail?**

We set the virtual key's requests per minute (RPM) limit to 1. This has now been crossed.


```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-12...' \
-d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful math tutor. Guide the user through the solution step by step."
      },
      {
        "role": "user",
        "content": "how can I solve 8x + 7 = -23"
      }
    ]
}'
```

**Expected Response**

```bash
{
  "error": {
    "message": "Max parallel request limit reached. Hit limit for api_key: daa1b272072a4c6841470a488c5dad0f298ff506e1cc935f4a181eed90c182ad. tpm_limit: 100, current_tpm: 29, rpm_limit: 1, current_rpm: 2.",
    "type": "None",
    "param": "None",
    "code": "429"
  }
}
```

### Useful Links 

- [Creating Virtual Keys](./virtual_keys.md)
- [Key Management API Endpoints Swagger](https://litellm-api.up.railway.app/#/key%20management)
- [Set Budgets / Rate Limits per key/user/teams](./users.md)
- [Dynamic TPM/RPM Limits for keys](./team_budgets.md#dynamic-tpmrpm-allocation)


## Troubleshooting 

### Non-root docker image?

If you need to run the docker image as a non-root user, use [this](https://github.com/BerriAI/litellm/pkgs/container/litellm-non_root).

### SSL Verification Issue / Connection Error.

If you see 

```bash
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate in certificate chain (_ssl.c:1006)
```

OR

```bash
Connection Error.
```

You can disable ssl verification with: 

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/my_azure_deployment
      api_base: os.environ/AZURE_API_BASE
      api_key: "os.environ/AZURE_API_KEY"
      api_version: "2024-07-01-preview"

litellm_settings:
    ssl_verify: false # 👈 KEY CHANGE
```

**What is `litellm_settings`?**

LiteLLM Proxy uses the [LiteLLM Python SDK](https://docs.litellm.ai/docs/routing) for handling LLM API calls. 

`litellm_settings` are module-level params for the LiteLLM Python SDK (equivalent to doing `litellm.<some_param>` on the SDK). You can see all params [here](https://github.com/BerriAI/litellm/blob/208fe6cb90937f73e0def5c97ccb2359bf8a467b/litellm/__init__.py#L114)

## Support & Talk with founders

- [Schedule Demo 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)

- [Community Discord 💭](https://discord.gg/wuPM9dRgDw)

- Our emails ✉️ ishaan@berri.ai / krrish@berri.ai

[![Chat on WhatsApp](https://img.shields.io/static/v1?label=Chat%20on&message=WhatsApp&color=success&logo=WhatsApp&style=flat-square)](https://wa.link/huol9n) [![Chat on Discord](https://img.shields.io/static/v1?label=Chat%20on&message=Discord&color=blue&logo=Discord&style=flat-square)](https://discord.gg/wuPM9dRgDw) 

