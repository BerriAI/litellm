
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import Image from '@theme/IdealImage';

# Getting Started Tutorial

End-to-End tutorial for LiteLLM Proxy to:
- Add an Azure OpenAI model 
- Make a successful /chat/completion call 
- Generate a virtual key 
- Set RPM limit on virtual key 


## Pre-Requisites 

- Install LiteLLM Docker Image **OR** LiteLLM CLI (pip package)

<Tabs>

<TabItem value="docker" label="Docker">

```
docker pull docker.litellm.ai/berriai/litellm:main-latest
```

[**See all docker images**](https://github.com/orgs/BerriAI/packages)

</TabItem>

<TabItem value="pip" label="LiteLLM CLI (pip package)">

```shell
$ pip install 'litellm[proxy]'
```

</TabItem>

<TabItem value="docker-compose" label="Docker Compose (Proxy + DB)">

### Step 1 — Pull the LiteLLM database image

LiteLLM provides a dedicated `litellm-database` image for proxy deployments that connect to Postgres. Pull it before starting Docker Compose.

```bash
docker pull ghcr.io/berriai/litellm-database:main-latest
```

See all available tags on the [GitHub Container Registry](https://github.com/BerriAI/litellm/pkgs/container/litellm-database).

---

### Step 2 — Set up a database

Get the docker compose file and create your `.env` first:

```bash
# Get the docker compose file
curl -O https://raw.githubusercontent.com/BerriAI/litellm/main/docker-compose.yml

# Add the master key - you can change this after setup
echo 'LITELLM_MASTER_KEY="sk-1234"' > .env

# Add the litellm salt key — cannot be changed after adding a model
# Used to encrypt/decrypt your LLM API key credentials
# Generate a strong random value: https://1password.com/password-generator/
echo 'LITELLM_SALT_KEY="sk-1234"' >> .env

# Add your model credentials
echo 'AZURE_API_BASE="https://openai-***********/"' >> .env
echo 'AZURE_API_KEY="your-azure-api-key"' >> .env
```

Then finish your `config.yaml` before starting the proxy server. If you use the default `docker-compose.yml`, the Postgres container is available at `db:5432`.

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: azure/my_azure_deployment
      api_base: os.environ/AZURE_API_BASE
      api_key: os.environ/AZURE_API_KEY
      api_version: "2025-01-01-preview"

general_settings:
  master_key: sk-1234   # 🔑 your proxy admin key (must start with sk-)
  database_url: "postgresql://llmproxy:dbpassword9090@db:5432/litellm"
```

:::tip
`database_url` is required for virtual keys, spend tracking, and the UI. If you want a managed database instead, replace it with your [Supabase](https://supabase.com/) or [Neon](https://neon.tech/) connection string.
:::

Save this file as `config.yaml`.

---

### Step 3 — Start the proxy server and test it

After `config.yaml` and `.env` are complete, start the proxy:

```bash
docker compose up
```

Once running, test it with a curl request:

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-1234' \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**Expected response:**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "choices": [{"message": {"role": "assistant", "content": "Hello! How can I help you?"}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 10, "completion_tokens": 9, "total_tokens": 19}
}
```

---

### Optional — Navigate to the LiteLLM UI and generate a virtual key

Open [http://localhost:4000/ui](http://localhost:4000/ui) in your browser and log in with your master key (`sk-1234`).

Navigate to **Virtual Keys** and click **+ Create New Key**:

<Image img={require('../../img/litellm_ui_create_key.png')} alt="LiteLLM UI — Create Virtual Key" />

Virtual keys let you track spend, set rate limits, and control model access per user or team.

</TabItem>
</Tabs>

## 1. Add a model 

Control LiteLLM Proxy with a config.yaml file.

Setup your config.yaml with your azure model.

Note: When using the proxy with a database, you can also **just add models via UI** (UI is available on `/ui` route).

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: azure/my_azure_deployment
      api_base: os.environ/AZURE_API_BASE
      api_key: "os.environ/AZURE_API_KEY"
      api_version: "2025-01-01-preview" # [OPTIONAL] litellm uses the latest azure api_version by default
```
---

### Model List Specification

You can read more about how model resolution works in the [Model Configuration](#understanding-model-configuration) section.

- **`model_name`** (`str`) - This field should contain the name of the model as received.
- **`litellm_params`** (`dict`) [See All LiteLLM Params](https://github.com/BerriAI/litellm/blob/559a6ad826b5daef41565f54f06c739c8c068b28/litellm/types/router.py#L222)
    - **`model`** (`str`) - Specifies the model name to be sent to `litellm.acompletion` / `litellm.aembedding`, etc. This is the identifier used by LiteLLM to route to the correct model + provider logic on the backend. 
    - **`api_key`** (`str`) - The API key required for authentication. It can be retrieved from an environment variable using `os.environ/`.
    - **`api_base`** (`str`) - The API base for your azure deployment.
    - **`api_version`** (`str`) - The API Version to use when calling Azure's OpenAI API. Get the latest Inference API version [here](https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation?source=recommendations#latest-preview-api-releases).

---


---

### Useful Links
- [**All Supported LLM API Providers (OpenAI/Bedrock/Vertex/etc.)**](../providers/)
- [**Full Config.Yaml Spec**](./configs.md)
- [**Pass provider-specific params**](../completion/provider_specific_params.md#proxy-usage)


## 2. Make a successful /chat/completion call 

LiteLLM Proxy is 100% OpenAI-compatible. Test your azure model via the `/chat/completions` route.

### 2.1 Start Proxy 

Save your config.yaml from step 1. as `litellm_config.yaml`.

<Tabs>


<TabItem value="docker" label="Docker">

```bash
docker run \
    -v $(pwd)/litellm_config.yaml:/app/config.yaml \
    -e AZURE_API_KEY=d6*********** \
    -e AZURE_API_BASE=https://openai-***********/ \
    -p 4000:4000 \
    docker.litellm.ai/berriai/litellm:main-latest \
    --config /app/config.yaml --detailed_debug

# RUNNING on http://0.0.0.0:4000
```

</TabItem>

<TabItem value="pip" label="LiteLLM CLI (pip package)">

```shell
$ litellm --config /app/config.yaml --detailed_debug
```

</TabItem>


</Tabs>


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
    "model": "gpt-4o",
    "messages": [
      {
        "role": "system",
        "content": "You are an LLM named gpt-4o"
      },
      {
        "role": "user",
        "content": "what is your name?"
      }
    ]
}'
```

**Expected Response**

```bash
{
  "id": "chatcmpl-BcO8tRQmQV6Dfw6onqMufxPkLLkA8",
  "created": 1748488967,
  "model": "gpt-4o-2024-11-20",
  "object": "chat.completion",
  "system_fingerprint": "fp_ee1d74bde0",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "My name is **gpt-4o**! How can I assist you today?",
        "role": "assistant",
        "tool_calls": null,
        "function_call": null,
        "annotations": []
      }
    }
  ],
  "usage": {
    "completion_tokens": 19,
    "prompt_tokens": 28,
    "total_tokens": 47,
    "completion_tokens_details": {
      "accepted_prediction_tokens": 0,
      "audio_tokens": 0,
      "reasoning_tokens": 0,
      "rejected_prediction_tokens": 0
    },
    "prompt_tokens_details": {
      "audio_tokens": 0,
      "cached_tokens": 0
    }
  },
  "service_tier": null,
  "prompt_filter_results": [
    {
      "prompt_index": 0,
      "content_filter_results": {
        "hate": {
          "filtered": false,
          "severity": "safe"
        },
        "self_harm": {
          "filtered": false,
          "severity": "safe"
        },
        "sexual": {
          "filtered": false,
          "severity": "safe"
        },
        "violence": {
          "filtered": false,
          "severity": "safe"
        }
      }
    }
  ]
}
```



### Useful Links
- [All Supported LLM API Providers (OpenAI/Bedrock/Vertex/etc.)](../providers/)
- [Call LiteLLM Proxy via OpenAI SDK, Langchain, etc.](./user_keys.md#request-format)
- [All API Endpoints Swagger](https://litellm-api.up.railway.app/#/chat%2Fcompletions)
- [Other/Non-Chat Completion Endpoints](../embedding/supported_embedding.md)
- [Pass-through for VertexAI, Bedrock, etc.](../pass_through/vertex_ai.md)

## Optional: Generate a virtual key

Track spend and control model access via virtual keys for the proxy.

### Prerequisite — Set up a database

**Requirements**
- Need a postgres database (e.g. [Supabase](https://supabase.com/), [Neon](https://neon.tech/), etc)


```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: azure/my_azure_deployment
      api_base: os.environ/AZURE_API_BASE
      api_key: "os.environ/AZURE_API_KEY"
      api_version: "2025-01-01-preview" # [OPTIONAL] litellm uses the latest azure api_version by default

general_settings: 
  master_key: sk-1234 
  database_url: "postgresql://<user>:<password>@<host>:<port>/<dbname>" # 👈 KEY CHANGE
```

Save config.yaml as `litellm_config.yaml` before continuing.

You must finish this setup before starting the proxy server.

---

**What is `general_settings`?**

These are settings for the LiteLLM Proxy Server. 

See All General Settings [here](http://localhost:3000/docs/proxy/configs#all-settings).

1. **`master_key`** (`str`)
   - **Description**: 
     - Set a `master key`, this is your Proxy Admin key - you can use this to create other keys (🚨 must start with `sk-`).
   - **Usage**: 
     - **Set on config.yaml** set your master key under `general_settings:master_key`, example - 
        `master_key: sk-1234`
     - **Set env variable** set `LITELLM_MASTER_KEY`

2. **`database_url`** (str)
   - **Description**: 
     - Set a `database_url`, this is the connection to your Postgres DB, which is used by litellm for generating keys, users, teams.
   - **Usage**: 
     - **Set on config.yaml** set your `database_url` under `general_settings:database_url`, example - 
        `database_url: "postgresql://..."`
     - Set `DATABASE_URL=postgresql://<user>:<password>@<host>:<port>/<dbname>` in your env 

### Start Proxy

```bash
docker run \
    -v $(pwd)/litellm_config.yaml:/app/config.yaml \
    -e AZURE_API_KEY=d6*********** \
    -e AZURE_API_BASE=https://openai-***********/ \
    -p 4000:4000 \
    ghcr.io/berriai/litellm-database:main-latest \
    --config /app/config.yaml --detailed_debug
```


### Create Key w/ RPM Limit

Create a key with `rpm_limit: 1`. This will only allow 1 request per minute for calls to proxy with this key.

```bash 
curl -L -X POST 'http://0.0.0.0:4000/key/generate' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{
    "rpm_limit": 1
}'
```

[**See full API Spec**](https://litellm-api.up.railway.app/#/key%20management/generate_key_fn_key_generate_post)

**Expected Response**

```bash
{
    "key": "sk-12..."
}
```

### Test it!

**Use the virtual key you just created.**

1st call - Expect to work! 

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-12...' \
-d '{
    "model": "gpt-4o",
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
    "model": "gpt-4o",
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
    "message": "LiteLLM Rate Limit Handler for rate limit type = key. Crossed TPM / RPM / Max Parallel Request Limit. current rpm: 1, rpm limit: 1, current tpm: 348, tpm limit: 9223372036854775807, current max_parallel_requests: 0, max_parallel_requests: 9223372036854775807",
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

## Key Concepts

This section explains key concepts on LiteLLM AI Gateway.

### Understanding Model Configuration

For this config.yaml example:

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: azure/my_azure_deployment
      api_base: os.environ/AZURE_API_BASE
      api_key: "os.environ/AZURE_API_KEY"
      api_version: "2025-01-01-preview" # [OPTIONAL] litellm uses the latest azure api_version by default
```

**How Model Resolution Works:**

```
Client Request                LiteLLM Proxy                 Provider API
──────────────              ────────────────              ─────────────
    
POST /chat/completions      
{                           1. Looks up model_name
  "model": "gpt-4o" ──────────▶ in config.yaml
  ...                          
}                           2. Finds matching entry:
                               model_name: gpt-4o
                               
                            3. Extracts litellm_params:
                               model: azure/my_azure_deployment
                               api_base: https://...
                               api_key: sk-...
                               
                            4. Routes to provider ──▶ Azure OpenAI API
                                                      POST /deployments/my_azure_deployment/...
```

**Breaking Down the `model` Parameter under `litellm_params`:**

```yaml
model_list:
  - model_name: gpt-4o                       # What the client calls
    litellm_params:
      model: azure/my_azure_deployment       # <provider>/<model-name>
             ─────  ───────────────────
               │           │
               │           └─────▶ Model name sent to the provider API
               │
               └─────────────────▶ Provider that LiteLLM routes to
```

**Visual Breakdown:**

```
model: azure/my_azure_deployment
       └─┬─┘ └─────────┬─────────┘
         │             │
         │             └────▶ The actual model identifier that gets sent to Azure
         │                   (e.g., your deployment name, or the model name)
         │
         └──────────────────▶ Tells LiteLLM which provider to use
                             (azure, openai, anthropic, bedrock, etc.)
```

**Key Concepts:**

- **`model_name`**: The alias your client uses to call the model. This is what you send in your API requests (e.g., `gpt-4o`).

- **`model` (in litellm_params)**: Format is `<provider>/<model-identifier>`
  - **Provider** (before `/`): Routes to the correct LLM provider (e.g., `azure`, `openai`, `anthropic`, `bedrock`)
  - **Model identifier** (after `/`): The actual model/deployment name sent to that provider's API

**Advanced Configuration Examples:**

For custom OpenAI-compatible endpoints (e.g., vLLM, Ollama, custom deployments):

```yaml
model_list:
  - model_name: my-custom-model
    litellm_params:
      model: openai/nvidia/llama-3.2-nv-embedqa-1b-v2
      api_base: http://my-service.svc.cluster.local:8000/v1
      api_key: "sk-1234"
```

**Breaking down complex model paths:**

```
model: openai/nvidia/llama-3.2-nv-embedqa-1b-v2
       └─┬──┘ └────────────┬────────────────┘
         │                 │
         │                 └────▶ Full model string sent to the provider API
         │                       (in this case: "nvidia/llama-3.2-nv-embedqa-1b-v2")
         │
         └──────────────────────▶ Provider (openai = OpenAI-compatible API)
```

The key point: Everything after the first `/` is passed as-is to the provider's API.

**Common Patterns:**

```yaml
model_list:
  # Azure deployment
  - model_name: gpt-4
    litellm_params:
      model: azure/gpt-4-deployment
      api_base: https://my-azure.openai.azure.com
      
  # OpenAI
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY
      
  # Custom OpenAI-compatible endpoint
  - model_name: my-llama-model
    litellm_params:
      model: openai/meta/llama-3-8b
      api_base: http://my-vllm-server:8000/v1
      api_key: "optional-key"
      
  # Bedrock
  - model_name: claude-3
    litellm_params:
      model: bedrock/anthropic.claude-3-sonnet-20240229-v1:0
      aws_region_name: us-east-1
```


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
  - model_name: gpt-4o
    litellm_params:
      model: azure/my_azure_deployment
      api_base: os.environ/AZURE_API_BASE
      api_key: "os.environ/AZURE_API_KEY"
      api_version: "2025-01-01-preview"

litellm_settings:
    ssl_verify: false # 👈 KEY CHANGE
```


### (DB) All connection attempts failed 


If you see:

```
httpx.ConnectError: All connection attempts failed                                                                        
                                                                                                                         
ERROR:    Application startup failed. Exiting.                                                                            
3:21:43 - LiteLLM Proxy:ERROR: utils.py:2207 - Error getting LiteLLM_SpendLogs row count: All connection attempts failed 
```

This might be a DB permission issue. 

1. Validate db user permission issue 

Try creating a new database. 

```bash
STATEMENT: CREATE DATABASE "litellm"
```

If you get:

```
ERROR: permission denied to create 
```

This indicates you have a permission issue. 

2. Grant permissions to your DB user

It should look something like this:

```
psql -U postgres
```

```
CREATE DATABASE litellm;
```

On CloudSQL, this is:

```
GRANT ALL PRIVILEGES ON DATABASE litellm TO your_username;
```


**What is `litellm_settings`?**

LiteLLM Proxy uses the [LiteLLM Python SDK](https://docs.litellm.ai/docs/routing) for handling LLM API calls. 

`litellm_settings` are module-level params for the LiteLLM Python SDK (equivalent to doing `litellm.<some_param>` on the SDK). You can see all params [here](https://github.com/BerriAI/litellm/blob/208fe6cb90937f73e0def5c97ccb2359bf8a467b/litellm/__init__.py#L114)

## Support & Talk with founders

- [Schedule Demo 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)

- [Community Discord 💭](https://discord.gg/wuPM9dRgDw)
- [Community Slack 💭](https://www.litellm.ai/support)

- Our emails ✉️ ishaan@berri.ai / krrish@berri.ai

[![Chat on WhatsApp](https://img.shields.io/static/v1?label=Chat%20on&message=WhatsApp&color=success&logo=WhatsApp&style=flat-square)](https://wa.link/huol9n) [![Chat on Discord](https://img.shields.io/static/v1?label=Chat%20on&message=Discord&color=blue&logo=Discord&style=flat-square)](https://discord.gg/wuPM9dRgDw) 
