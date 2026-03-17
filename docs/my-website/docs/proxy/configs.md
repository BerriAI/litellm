import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Overview
Set model list, `api_base`, `api_key`, `temperature` & proxy server settings (`master-key`) on the config.yaml. 

| Param Name           | Description                                                   |
|----------------------|---------------------------------------------------------------|
| `model_list`         | List of supported models on the server, with model-specific configs |
| `router_settings`   | litellm Router settings, example `routing_strategy="least-busy"` [**see all**](#router-settings)|
| `litellm_settings`   | litellm Module settings, example `litellm.drop_params=True`, `litellm.set_verbose=True`, `litellm.api_base`, `litellm.cache` [**see all**](#all-settings)|
| `general_settings`   | Server settings, example setting `master_key: sk-my_special_key` |
| `environment_variables`   | Environment Variables example, `REDIS_HOST`, `REDIS_PORT` |

**Complete List:** Check the Swagger UI docs on `<your-proxy-url>/#/config.yaml` (e.g. http://0.0.0.0:4000/#/config.yaml), for everything you can pass in the config.yaml.


## Quick Start 

Set a model alias for your deployments. 

In the `config.yaml` the model_name parameter is the user-facing name to use for your deployment. 

In the config below:
- `model_name`: the name to pass TO litellm from the external client  
- `litellm_params.model`: the model string passed to the litellm.completion() function

E.g.: 
- `model=vllm-models` will route to `openai/facebook/opt-125m`. 
- `model=gpt-4o` will load balance between `azure/gpt-4o-eu` and `azure/gpt-4o-ca`

```yaml
model_list:
  - model_name: gpt-4o ### RECEIVED MODEL NAME ###
    litellm_params: # all params accepted by litellm.completion() - https://docs.litellm.ai/docs/completion/input
      model: azure/gpt-4o-eu ### MODEL NAME sent to `litellm.completion()` ###
      api_base: https://my-endpoint-europe-berri-992.openai.azure.com/
      api_key: "os.environ/AZURE_API_KEY_EU" # does os.getenv("AZURE_API_KEY_EU")
      rpm: 6      # [OPTIONAL] Rate limit for this deployment: in requests per minute (rpm)
  - model_name: bedrock-claude-v1 
    litellm_params:
      model: bedrock/anthropic.claude-instant-v1
  - model_name: gpt-4o
    litellm_params:
      model: azure/gpt-4o-ca
      api_base: https://my-endpoint-canada-berri992.openai.azure.com/
      api_key: "os.environ/AZURE_API_KEY_CA"
      rpm: 6
  - model_name: anthropic-claude
    litellm_params: 
      model: bedrock/anthropic.claude-instant-v1
      ### [OPTIONAL] SET AWS REGION ###
      aws_region_name: us-east-1
  - model_name: vllm-models
    litellm_params:
      model: openai/facebook/opt-125m # the `openai/` prefix tells litellm it's openai compatible
      api_base: http://0.0.0.0:4000/v1
      api_key: none
      rpm: 1440
    model_info: 
      version: 2
  
  # Use this if you want to make requests to `claude-3-haiku-20240307`,`claude-3-opus-20240229`,`claude-2.1` without defining them on the config.yaml
  # Default models
  # Works for ALL Providers and needs the default provider credentials in .env
  - model_name: "*" 
    litellm_params:
      model: "*"

litellm_settings: # module level litellm settings - https://github.com/BerriAI/litellm/blob/main/litellm/__init__.py
  drop_params: True
  success_callback: ["langfuse"] # OPTIONAL - if you want to start sending LLM Logs to Langfuse. Make sure to set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in your env

general_settings: 
  master_key: sk-1234 # [OPTIONAL] Only use this if you to require all calls to contain this key (Authorization: Bearer sk-1234)
  alerting: ["slack"] # [OPTIONAL] If you want Slack Alerts for Hanging LLM requests, Slow llm responses, Budget Alerts. Make sure to set `SLACK_WEBHOOK_URL` in your env
```
:::info

For more provider-specific info, [go here](../providers/)

:::

#### Step 2: Start Proxy with config

```shell
$ litellm --config /path/to/config.yaml
```

:::tip

Run with `--detailed_debug` if you need detailed debug logs 

```shell
$ litellm --config /path/to/config.yaml --detailed_debug
```

:::

#### Step 3: Test it

Sends request to model where `model_name=gpt-4o` on config.yaml. 

If multiple with `model_name=gpt-4o` does [Load Balancing](https://docs.litellm.ai/docs/proxy/load_balancing)

**[Langchain, OpenAI SDK Usage Examples](../proxy/user_keys#request-format)**

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "gpt-4o",
      "messages": [
        {
          "role": "user",
          "content": "what llm are you"
        }
      ]
    }
'
```

## LLM configs `model_list`

### Model-specific params (API Base, Keys, Temperature, Max Tokens, Organization, Headers etc.)
You can use the config to save model-specific information like api_base, api_key, temperature, max_tokens, etc. 

[**All input params**](https://docs.litellm.ai/docs/completion/input#input-params-1)

**Step 1**: Create a `config.yaml` file
```yaml
model_list:
  - model_name: gpt-4-team1
    litellm_params: # params for litellm.completion() - https://docs.litellm.ai/docs/completion/input#input---request-body
      model: azure/chatgpt-v-2
      api_base: https://openai-gpt-4-test-v-1.openai.azure.com/
      api_version: "2023-05-15"
      azure_ad_token: eyJ0eXAiOiJ
      seed: 12
      max_tokens: 20
  - model_name: gpt-4-team2
    litellm_params:
      model: azure/gpt-4
      api_key: sk-123
      api_base: https://openai-gpt-4-test-v-2.openai.azure.com/
      temperature: 0.2
  - model_name: openai-gpt-4o
    litellm_params:
      model: openai/gpt-4o
      extra_headers: {"AI-Resource Group": "ishaan-resource"}
      api_key: sk-123
      organization: org-ikDc4ex8NB
      temperature: 0.2
  - model_name: mistral-7b
    litellm_params:
      model: ollama/mistral
      api_base: your_ollama_api_base
```

**Step 2**: Start server with config

```shell
$ litellm --config /path/to/config.yaml
```

**Expected Logs:**

Look for this line in your console logs to confirm the config.yaml was loaded in correctly.
```
LiteLLM: Proxy initialized with Config, Set models:
```

### Embedding Models - Use Sagemaker, Bedrock, Azure, OpenAI, XInference

See supported Embedding Providers & Models [here](https://docs.litellm.ai/docs/embedding/supported_embedding)


<Tabs>
<TabItem value="bedrock" label="Bedrock Completion/Chat">

```yaml
model_list:
  - model_name: bedrock-cohere
    litellm_params:
      model: "bedrock/cohere.command-text-v14"
      aws_region_name: "us-west-2"
  - model_name: bedrock-cohere
    litellm_params:
      model: "bedrock/cohere.command-text-v14"
      aws_region_name: "us-east-2"
  - model_name: bedrock-cohere
    litellm_params:
      model: "bedrock/cohere.command-text-v14"
      aws_region_name: "us-east-1"

```

</TabItem>

<TabItem value="sagemaker" label="Sagemaker, Bedrock Embeddings">

Here's how to route between GPT-J embedding (sagemaker endpoint), Amazon Titan embedding (Bedrock) and Azure OpenAI embedding on the proxy server: 

```yaml
model_list:
  - model_name: sagemaker-embeddings
    litellm_params: 
      model: "sagemaker/berri-benchmarking-gpt-j-6b-fp16"
  - model_name: amazon-embeddings
    litellm_params:
      model: "bedrock/amazon.titan-embed-text-v1"
  - model_name: azure-embeddings
    litellm_params: 
      model: "azure/azure-embedding-model"
      api_base: "os.environ/AZURE_API_BASE" # os.getenv("AZURE_API_BASE")
      api_key: "os.environ/AZURE_API_KEY" # os.getenv("AZURE_API_KEY")
      api_version: "2023-07-01-preview"

general_settings:
  master_key: sk-1234 # [OPTIONAL] if set all calls to proxy will require either this key or a valid generated token
```

</TabItem>

<TabItem value="Hugging Face emb" label="Hugging Face Embeddings">
LiteLLM Proxy supports all <a href="https://huggingface.co/models?pipeline_tag=feature-extraction">Feature-Extraction Embedding models</a>.

```yaml
model_list:
  - model_name: deployed-codebert-base
    litellm_params: 
      # send request to deployed hugging face inference endpoint
      model: huggingface/microsoft/codebert-base # add huggingface prefix so it routes to hugging face
      api_key: hf_LdS                            # api key for hugging face inference endpoint
      api_base: https://uysneno1wv2wd4lw.us-east-1.aws.endpoints.huggingface.cloud # your hf inference endpoint 
  - model_name: codebert-base
    litellm_params: 
      # no api_base set, sends request to hugging face free inference api https://api-inference.huggingface.co/models/
      model: huggingface/microsoft/codebert-base # add huggingface prefix so it routes to hugging face
      api_key: hf_LdS                            # api key for hugging face                     

```

</TabItem>

<TabItem value="azure" label="Azure OpenAI Embeddings">

```yaml
model_list:
  - model_name: azure-embedding-model # model group
    litellm_params:
      model: azure/azure-embedding-model # model name for litellm.embedding(model=azure/azure-embedding-model) call
      api_base: your-azure-api-base
      api_key: your-api-key
      api_version: 2023-07-01-preview
```

</TabItem>

<TabItem value="openai" label="OpenAI Embeddings">

```yaml
model_list:
- model_name: text-embedding-ada-002 # model group
  litellm_params:
    model: text-embedding-ada-002 # model name for litellm.embedding(model=text-embedding-ada-002) 
    api_key: your-api-key-1
- model_name: text-embedding-ada-002 
  litellm_params:
    model: text-embedding-ada-002
    api_key: your-api-key-2
```

</TabItem>


<TabItem value="xinf" label="XInference">

https://docs.litellm.ai/docs/providers/xinference

**Note add `xinference/` prefix to `litellm_params`: `model` so litellm knows to route to OpenAI**

```yaml
model_list:
- model_name: embedding-model  # model group
  litellm_params:
    model: xinference/bge-base-en   # model name for litellm.embedding(model=xinference/bge-base-en) 
    api_base: http://0.0.0.0:9997/v1
```

</TabItem>

<TabItem value="openai emb" label="OpenAI Compatible Embeddings">

<p>Use this for calling <a href="https://github.com/xorbitsai/inference">/embedding endpoints on OpenAI Compatible Servers</a>.</p>

**Note add `openai/` prefix to `litellm_params`: `model` so litellm knows to route to OpenAI**

```yaml
model_list:
- model_name: text-embedding-ada-002  # model group
  litellm_params:
    model: openai/<your-model-name>   # model name for litellm.embedding(model=text-embedding-ada-002) 
    api_base: <model-api-base>
```

</TabItem>
</Tabs>

#### Start Proxy

```shell
litellm --config config.yaml
```

#### Make Request
Sends Request to `bedrock-cohere`

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
  --header 'Content-Type: application/json' \
  --data ' {
  "model": "bedrock-cohere",
  "messages": [
      {
      "role": "user",
      "content": "gm"
      }
  ]
}'
```


### Multiple OpenAI Organizations 

Add all openai models across all OpenAI organizations with just 1 model definition 

```yaml
  - model_name: *
    litellm_params:
      model: openai/*
      api_key: os.environ/OPENAI_API_KEY
      organization:
       - org-1 
       - org-2 
       - org-3
```

LiteLLM will automatically create separate deployments for each org.

Confirm this via 

```bash
curl --location 'http://0.0.0.0:4000/v1/model/info' \
--header 'Authorization: Bearer ${LITELLM_KEY}' \
--data ''
```

### Load Balancing 

:::info
For more on this, go to [this page](https://docs.litellm.ai/docs/proxy/load_balancing)
:::

Use this to call multiple instances of the same model and configure things like [routing strategy](https://docs.litellm.ai/docs/routing#advanced).

For optimal performance:
- Set `tpm/rpm` per model deployment. Weighted picks are then based on the established tpm/rpm.
- Select your optimal routing strategy in `router_settings:routing_strategy`.

LiteLLM supports
```python
["simple-shuffle", "least-busy", "usage-based-routing","latency-based-routing"], default="simple-shuffle"`
```

When `tpm/rpm` is set + `routing_strategy==simple-shuffle` litellm will use a weighted pick based on set tpm/rpm. **In our load tests setting tpm/rpm for all deployments + `routing_strategy==simple-shuffle` maximized throughput**
- When using multiple LiteLLM Servers / Kubernetes set redis settings `router_settings:redis_host` etc

```yaml
model_list:
  - model_name: zephyr-beta
    litellm_params:
        model: huggingface/HuggingFaceH4/zephyr-7b-beta
        api_base: http://0.0.0.0:8001
        rpm: 60      # Optional[int]: When rpm/tpm set - litellm uses weighted pick for load balancing. rpm = Rate limit for this deployment: in requests per minute (rpm).
        tpm: 1000   # Optional[int]: tpm = Tokens Per Minute 
  - model_name: zephyr-beta
    litellm_params:
        model: huggingface/HuggingFaceH4/zephyr-7b-beta
        api_base: http://0.0.0.0:8002
        rpm: 600      
  - model_name: zephyr-beta
    litellm_params:
        model: huggingface/HuggingFaceH4/zephyr-7b-beta
        api_base: http://0.0.0.0:8003
        rpm: 60000      
  - model_name: gpt-4o
    litellm_params:
        model: gpt-4o
        api_key: <my-openai-key>
        rpm: 200      
  - model_name: gpt-3.5-turbo-16k
    litellm_params:
        model: gpt-3.5-turbo-16k
        api_key: <my-openai-key>
        rpm: 100      

litellm_settings:
  num_retries: 3 # retry call 3 times on each model_name (e.g. zephyr-beta)
  request_timeout: 10 # raise Timeout error if call takes longer than 10s. Sets litellm.request_timeout 
  fallbacks: [{"zephyr-beta": ["gpt-4o"]}] # fallback to gpt-4o if call fails num_retries 
  context_window_fallbacks: [{"zephyr-beta": ["gpt-3.5-turbo-16k"]}, {"gpt-4o": ["gpt-3.5-turbo-16k"]}] # fallback to gpt-3.5-turbo-16k if context window error
  allowed_fails: 3 # cooldown model if it fails > 1 call in a minute. 

router_settings: # router_settings are optional
  routing_strategy: simple-shuffle # Literal["simple-shuffle", "least-busy", "usage-based-routing","latency-based-routing"], default="simple-shuffle"
  model_group_alias: {"gpt-4": "gpt-4o"} # all requests with `gpt-4` will be routed to models with `gpt-4o`
  num_retries: 2
  timeout: 30                                  # 30 seconds
  redis_host: <your redis host>                # set this when using multiple litellm proxy deployments, load balancing state stored in redis
  redis_password: <your redis password>
  redis_port: 1992
```

You can view your cost once you set up [Virtual keys](https://docs.litellm.ai/docs/proxy/virtual_keys) or [custom_callbacks](https://docs.litellm.ai/docs/proxy/logging)


### Load API Keys / config values from Environment 

If you have secrets saved in your environment, and don't want to expose them in the config.yaml, here's how to load model-specific keys from the environment. **This works for ANY value on the config.yaml**

```yaml
os.environ/<YOUR-ENV-VAR> # runs os.getenv("YOUR-ENV-VAR")
```

```yaml 
model_list:
  - model_name: gpt-4-team1
    litellm_params: # params for litellm.completion() - https://docs.litellm.ai/docs/completion/input#input---request-body
      model: azure/chatgpt-v-2
      api_base: https://openai-gpt-4-test-v-1.openai.azure.com/
      api_version: "2023-05-15"
      api_key: os.environ/AZURE_NORTH_AMERICA_API_KEY # 👈 KEY CHANGE
```

[**See Code**](https://github.com/BerriAI/litellm/blob/c12d6c3fe80e1b5e704d9846b246c059defadce7/litellm/utils.py#L2366)

s/o to [@David Manouchehri](https://www.linkedin.com/in/davidmanouchehri/) for helping with this. 

### Centralized Credential Management

Define credentials once and reuse them across multiple models. This helps with:
- Secret rotation
- Reducing config duplication

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: azure/gpt-4o
      litellm_credential_name: default_azure_credential  # Reference credential below

credential_list:
  - credential_name: default_azure_credential
    credential_values:
      api_key: os.environ/AZURE_API_KEY  # Load from environment
      api_base: os.environ/AZURE_API_BASE
      api_version: "2023-05-15"
    credential_info:
      description: "Production credentials for EU region"
      custom_llm_provider: "azure"
```

#### Key Parameters
- `credential_name`: Unique identifier for the credential set
- `credential_values`: Key-value pairs of credentials/secrets (supports `os.environ/` syntax)
- `credential_info`: Key-value pairs of user provided credentials information.  No key-value pairs are required, but the dictionary must exist.

### Load API Keys from Secret Managers (Azure Vault, etc)

[**Using Secret Managers with LiteLLM Proxy**](../secret)


### Set Supported Environments for a model - `production`, `staging`, `development`

Use this if you want to control which model is exposed on a specific litellm environment

Supported Environments:
- `production`
- `staging`
- `development`

1. Set `LITELLM_ENVIRONMENT="<environment>"` in your environment. Can be one of `production`, `staging` or `development`


2. For each model set the list of supported environments in `model_info.supported_environments`
```yaml
model_list:
 - model_name: gpt-3.5-turbo-16k
   litellm_params:
     model: openai/gpt-3.5-turbo-16k
     api_key: os.environ/OPENAI_API_KEY
   model_info:
     supported_environments: ["development", "production", "staging"]
 - model_name: gpt-4
   litellm_params:
     model: openai/gpt-4
     api_key: os.environ/OPENAI_API_KEY
   model_info:
     supported_environments: ["production", "staging"]
 - model_name: gpt-4o
   litellm_params:
     model: openai/gpt-4o
     api_key: os.environ/OPENAI_API_KEY
   model_info:
     supported_environments: ["production"]
```


### Set Custom Prompt Templates

LiteLLM by default checks if a model has a [prompt template and applies it](../completion/prompt_formatting.md) (e.g. if a huggingface model has a saved chat template in it's tokenizer_config.json). However, you can also set a custom prompt template on your proxy in the `config.yaml`: 

**Step 1**: Save your prompt template in a `config.yaml`
```yaml
# Model-specific parameters
model_list:
  - model_name: mistral-7b # model alias
    litellm_params: # actual params for litellm.completion()
      model: "huggingface/mistralai/Mistral-7B-Instruct-v0.1" 
      api_base: "<your-api-base>"
      api_key: "<your-api-key>" # [OPTIONAL] for hf inference endpoints
      initial_prompt_value: "\n"
      roles: {"system":{"pre_message":"<|im_start|>system\n", "post_message":"<|im_end|>"}, "assistant":{"pre_message":"<|im_start|>assistant\n","post_message":"<|im_end|>"}, "user":{"pre_message":"<|im_start|>user\n","post_message":"<|im_end|>"}}
      final_prompt_value: "\n"
      bos_token: " "
      eos_token: " "
      max_tokens: 4096
```

**Step 2**: Start server with config

```shell
$ litellm --config /path/to/config.yaml
``` 

### Set custom tokenizer 

If you're using the [`/utils/token_counter` endpoint](https://litellm-api.up.railway.app/#/llm%20utils/token_counter_utils_token_counter_post), and want to set a custom huggingface tokenizer for a model, you can do so in the `config.yaml`

```yaml
model_list:
  - model_name: openai-deepseek
    litellm_params:
      model: deepseek/deepseek-chat
      api_key: os.environ/OPENAI_API_KEY
    model_info:
      access_groups: ["restricted-models"]
      custom_tokenizer: 
        identifier: deepseek-ai/DeepSeek-V3-Base
        revision: main
        auth_token: os.environ/HUGGINGFACE_API_KEY
```

**Spec**
```
custom_tokenizer: 
  identifier: str # huggingface model identifier
  revision: str # huggingface model revision (usually 'main')
  auth_token: Optional[str] # huggingface auth token 
```

## General Settings `general_settings` (DB Connection, etc)

### Configure DB Pool Limits + Connection Timeouts 

```yaml
general_settings: 
  database_connection_pool_limit: 10 # sets connection pool per worker for prisma client to postgres db (default: 10, recommended: 10-20)
  database_connection_timeout: 60 # sets a 60s timeout for any connection call to the db 
```

**How to calculate the right value:**

The connection limit is applied **per worker process**, not per instance. This means if you have multiple workers, each worker will create its own connection pool.

**Formula:**
```
database_connection_pool_limit = MAX_DB_CONNECTIONS ÷ (number_of_instances × number_of_workers_per_instance)
```

**Example:**
- Your database allows a maximum of **100 connections**
- You're running **1 instance** of LiteLLM
- Each instance has **8 workers** (set via `--num_workers 8`)

Calculation: `100 ÷ (1 × 8) = 12.5`

Since you shouldn't use 12.5, round down to **10** to leave a safety buffer. This means:
- Each of the 8 workers will have a connection pool limit of 10
- Total maximum connections: 8 workers × 10 connections = 80 connections
- This stays safely under your database's 100 connection limit

## Extras


### Disable Swagger UI 

To disable the Swagger docs from the base url, set 

```env
NO_DOCS="True"
```

in your environment, and restart the proxy. 

### Disable Redoc

To disable the Redoc docs (defaults to `<your-proxy-url>/redoc`), set 

```env
NO_REDOC="True"
```

in your environment, and restart the proxy. 

### Use CONFIG_FILE_PATH for proxy (Easier Azure container deployment)

1. Setup config.yaml

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: gpt-4o
      api_key: os.environ/OPENAI_API_KEY
```

2. Store filepath as env var 

```bash
CONFIG_FILE_PATH="/path/to/config.yaml"
```

3. Start Proxy

```bash
$ litellm 

# RUNNING on http://0.0.0.0:4000
```


### Providing LiteLLM config.yaml file as a s3, GCS Bucket Object/url

Use this if you cannot mount a config file on your deployment service (example - AWS Fargate, Railway etc)

LiteLLM Proxy will read your config.yaml from an s3 Bucket or GCS Bucket 

<Tabs>
<TabItem value="gcs" label="GCS Bucket">

Set the following .env vars 
```shell
LITELLM_CONFIG_BUCKET_TYPE = "gcs"                              # set this to "gcs"         
LITELLM_CONFIG_BUCKET_NAME = "litellm-proxy"                    # your bucket name on GCS
LITELLM_CONFIG_BUCKET_OBJECT_KEY = "proxy_config.yaml"         # object key on GCS
```

Start litellm proxy with these env vars - litellm will read your config from GCS 

```shell
docker run --name litellm-proxy \
   -e DATABASE_URL=<database_url> \
   -e LITELLM_CONFIG_BUCKET_NAME=<bucket_name> \
   -e LITELLM_CONFIG_BUCKET_OBJECT_KEY="<object_key>> \
   -e LITELLM_CONFIG_BUCKET_TYPE="gcs" \
   -p 4000:4000 \
   docker.litellm.ai/berriai/litellm-database:main-latest --detailed_debug
```

</TabItem>

<TabItem value="s3" label="s3">

Set the following .env vars 
```shell
LITELLM_CONFIG_BUCKET_NAME = "litellm-proxy"                    # your bucket name on s3 
LITELLM_CONFIG_BUCKET_OBJECT_KEY = "litellm_proxy_config.yaml"  # object key on s3
```

Start litellm proxy with these env vars - litellm will read your config from s3 

```shell
docker run --name litellm-proxy \
   -e DATABASE_URL=<database_url> \
   -e LITELLM_CONFIG_BUCKET_NAME=<bucket_name> \
   -e LITELLM_CONFIG_BUCKET_OBJECT_KEY="<object_key>> \
   -p 4000:4000 \
   docker.litellm.ai/berriai/litellm-database:main-latest
```
</TabItem>
</Tabs>
