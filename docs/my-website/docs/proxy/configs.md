import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Proxy Config.yaml
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
- `model=gpt-3.5-turbo` will load balance between `azure/gpt-turbo-small-eu` and `azure/gpt-turbo-small-ca`

```yaml
model_list:
  - model_name: gpt-3.5-turbo ### RECEIVED MODEL NAME ###
    litellm_params: # all params accepted by litellm.completion() - https://docs.litellm.ai/docs/completion/input
      model: azure/gpt-turbo-small-eu ### MODEL NAME sent to `litellm.completion()` ###
      api_base: https://my-endpoint-europe-berri-992.openai.azure.com/
      api_key: "os.environ/AZURE_API_KEY_EU" # does os.getenv("AZURE_API_KEY_EU")
      rpm: 6      # [OPTIONAL] Rate limit for this deployment: in requests per minute (rpm)
  - model_name: bedrock-claude-v1 
    litellm_params:
      model: bedrock/anthropic.claude-instant-v1
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/gpt-turbo-small-ca
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

Sends request to model where `model_name=gpt-3.5-turbo` on config.yaml. 

If multiple with `model_name=gpt-3.5-turbo` does [Load Balancing](https://docs.litellm.ai/docs/proxy/load_balancing)

**[Langchain, OpenAI SDK Usage Examples](../proxy/user_keys#request-format)**

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "gpt-3.5-turbo",
      "messages": [
        {
          "role": "user",
          "content": "what llm are you"
        }
      ],
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
  - model_name: openai-gpt-3.5
    litellm_params:
      model: openai/gpt-3.5-turbo
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

 
### Provider specific wildcard routing 
**Proxy all models from a provider**

Use this if you want to **proxy all models from a specific provider without defining them on the config.yaml**

**Step 1** - define provider specific routing on config.yaml
```yaml
model_list:
  # provider specific wildcard routing
  - model_name: "anthropic/*"
    litellm_params:
      model: "anthropic/*"
      api_key: os.environ/ANTHROPIC_API_KEY
  - model_name: "groq/*"
    litellm_params:
      model: "groq/*"
      api_key: os.environ/GROQ_API_KEY
  - model_name: "fo::*:static::*" # all requests matching this pattern will be routed to this deployment, example: model="fo::hi::static::hi" will be routed to deployment: "openai/fo::*:static::*"
    litellm_params:
      model: "openai/fo::*:static::*"
      api_key: os.environ/OPENAI_API_KEY
```

Step 2 - Run litellm proxy 

```shell
$ litellm --config /path/to/config.yaml
```

Step 3 Test it 

Test with `anthropic/` - all models with `anthropic/` prefix will get routed to `anthropic/*`
```shell
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "anthropic/claude-3-sonnet-20240229",
    "messages": [
      {"role": "user", "content": "Hello, Claude!"}
    ]
  }'
```

Test with `groq/` - all models with `groq/` prefix will get routed to `groq/*`
```shell
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "groq/llama3-8b-8192",
    "messages": [
      {"role": "user", "content": "Hello, Claude!"}
    ]
  }'
```

Test with `fo::*::static::*` - all requests matching this pattern will be routed to `openai/fo::*:static::*`
```shell
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "fo::hi::static::hi",
    "messages": [
      {"role": "user", "content": "Hello, Claude!"}
    ]
  }'
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
  - model_name: gpt-3.5-turbo
    litellm_params:
        model: gpt-3.5-turbo
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
  fallbacks: [{"zephyr-beta": ["gpt-3.5-turbo"]}] # fallback to gpt-3.5-turbo if call fails num_retries 
  context_window_fallbacks: [{"zephyr-beta": ["gpt-3.5-turbo-16k"]}, {"gpt-3.5-turbo": ["gpt-3.5-turbo-16k"]}] # fallback to gpt-3.5-turbo-16k if context window error
  allowed_fails: 3 # cooldown model if it fails > 1 call in a minute. 

router_settings: # router_settings are optional
  routing_strategy: simple-shuffle # Literal["simple-shuffle", "least-busy", "usage-based-routing","latency-based-routing"], default="simple-shuffle"
  model_group_alias: {"gpt-4": "gpt-3.5-turbo"} # all requests with `gpt-4` will be routed to models with `gpt-3.5-turbo`
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
 - model_name: gpt-3.5-turbo
   litellm_params:
     model: openai/gpt-3.5-turbo
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

## General Settings `general_settings` (DB Connection, etc)

### Configure DB Pool Limits + Connection Timeouts 

```yaml
general_settings: 
  database_connection_pool_limit: 100 # sets connection pool for prisma client to postgres db at 100
  database_connection_timeout: 60 # sets a 60s timeout for any connection call to the db 
```

## **All settings**


```yaml
environment_variables: {}

model_list:
  - model_name: string
    litellm_params: {}
    model_info:
      id: string
      mode: embedding
      input_cost_per_token: 0
      output_cost_per_token: 0
      max_tokens: 2048
      base_model: gpt-4-1106-preview
      additionalProp1: {}

litellm_settings:
  # Logging/Callback settings
  success_callback: ["langfuse"]  # list of success callbacks
  failure_callback: ["sentry"]  # list of failure callbacks
  callbacks: ["otel"]  # list of callbacks - runs on success and failure
  service_callbacks: ["datadog", "prometheus"]  # logs redis, postgres failures on datadog, prometheus
  turn_off_message_logging: boolean  # prevent the messages and responses from being logged to on your callbacks, but request metadata will still be logged.
  redact_user_api_key_info: boolean  # Redact information about the user api key (hashed token, user_id, team id, etc.), from logs. Currently supported for Langfuse, OpenTelemetry, Logfire, ArizeAI logging.
  langfuse_default_tags: ["cache_hit", "cache_key", "proxy_base_url", "user_api_key_alias", "user_api_key_user_id", "user_api_key_user_email", "user_api_key_team_alias", "semantic-similarity", "proxy_base_url"] # default tags for Langfuse Logging
  
  request_timeout: 10 # (int) llm requesttimeout in seconds. Raise Timeout error if call takes longer than 10s. Sets litellm.request_timeout 
  
  set_verbose: boolean # sets litellm.set_verbose=True to view verbose debug logs. DO NOT LEAVE THIS ON IN PRODUCTION
  json_logs: boolean # if true, logs will be in json format

  # Fallbacks, reliability
  default_fallbacks: ["claude-opus"] # set default_fallbacks, in case a specific model group is misconfigured / bad.
  content_policy_fallbacks: [{"gpt-3.5-turbo-small": ["claude-opus"]}] # fallbacks for ContentPolicyErrors
  context_window_fallbacks: [{"gpt-3.5-turbo-small": ["gpt-3.5-turbo-large", "claude-opus"]}] # fallbacks for ContextWindowExceededErrors



  # Caching settings
  cache: true 
  cache_params:        # set cache params for redis
    type: redis        # type of cache to initialize

    # Optional - Redis Settings
    host: "localhost"  # The host address for the Redis cache. Required if type is "redis".
    port: 6379  # The port number for the Redis cache. Required if type is "redis".
    password: "your_password"  # The password for the Redis cache. Required if type is "redis".
    namespace: "litellm.caching.caching" # namespace for redis cache
  
    # Optional - Redis Cluster Settings
    redis_startup_nodes: [{"host": "127.0.0.1", "port": "7001"}] 

    # Optional - Redis Sentinel Settings
    service_name: "mymaster"
    sentinel_nodes: [["localhost", 26379]]

    # Optional - Qdrant Semantic Cache Settings
    qdrant_semantic_cache_embedding_model: openai-embedding # the model should be defined on the model_list
    qdrant_collection_name: test_collection
    qdrant_quantization_config: binary
    similarity_threshold: 0.8   # similarity threshold for semantic cache

    # Optional - S3 Cache Settings
    s3_bucket_name: cache-bucket-litellm   # AWS Bucket Name for S3
    s3_region_name: us-west-2              # AWS Region Name for S3
    s3_aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID  # us os.environ/<variable name> to pass environment variables. This is AWS Access Key ID for S3
    s3_aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY  # AWS Secret Access Key for S3
    s3_endpoint_url: https://s3.amazonaws.com  # [OPTIONAL] S3 endpoint URL, if you want to use Backblaze/cloudflare s3 bucket

    # Common Cache settings
    # Optional - Supported call types for caching
    supported_call_types: ["acompletion", "atext_completion", "aembedding", "atranscription"]
                          # /chat/completions, /completions, /embeddings, /audio/transcriptions
    mode: default_off # if default_off, you need to opt in to caching on a per call basis
    ttl: 600 # ttl for caching


callback_settings:
  otel:
    message_logging: boolean  # OTEL logging callback specific settings

general_settings:
  completion_model: string
  disable_spend_logs: boolean  # turn off writing each transaction to the db
  disable_master_key_return: boolean  # turn off returning master key on UI (checked on '/user/info' endpoint)
  disable_retry_on_max_parallel_request_limit_error: boolean  # turn off retries when max parallel request limit is reached
  disable_reset_budget: boolean  # turn off reset budget scheduled task
  disable_adding_master_key_hash_to_db: boolean  # turn off storing master key hash in db, for spend tracking
  enable_jwt_auth: boolean  # allow proxy admin to auth in via jwt tokens with 'litellm_proxy_admin' in claims
  enforce_user_param: boolean  # requires all openai endpoint requests to have a 'user' param
  allowed_routes: ["route1", "route2"]  # list of allowed proxy API routes - a user can access. (currently JWT-Auth only)
  key_management_system: google_kms  # either google_kms or azure_kms
  master_key: string

  # Database Settings
  database_url: string
  database_connection_pool_limit: 0  # default 100
  database_connection_timeout: 0  # default 60s
  allow_requests_on_db_unavailable: boolean  # if true, will allow requests that can not connect to the DB to verify Virtual Key to still work 

  custom_auth: string
  max_parallel_requests: 0  # the max parallel requests allowed per deployment 
  global_max_parallel_requests: 0  # the max parallel requests allowed on the proxy all up 
  infer_model_from_keys: true
  background_health_checks: true
  health_check_interval: 300
  alerting: ["slack", "email"]
  alerting_threshold: 0
  use_client_credentials_pass_through_routes: boolean  # use client credentials for all pass through routes like "/vertex-ai", /bedrock/. When this is True Virtual Key auth will not be applied on these endpoints
```

### litellm_settings - Reference

| Name | Type | Description |
|------|------|-------------|
| success_callback | array of strings | List of success callbacks. [Doc Proxy logging callbacks](logging), [Doc Metrics](prometheus) |
| failure_callback | array of strings | List of failure callbacks [Doc Proxy logging callbacks](logging), [Doc Metrics](prometheus) |
| callbacks | array of strings | List of callbacks - runs on success and failure [Doc Proxy logging callbacks](logging), [Doc Metrics](prometheus) |
| service_callbacks | array of strings | System health monitoring - Logs redis, postgres failures on specified services (e.g. datadog, prometheus) [Doc Metrics](prometheus) |
| turn_off_message_logging | boolean | If true, prevents messages and responses from being logged to callbacks, but request metadata will still be logged [Proxy Logging](logging) |
| modify_params | boolean | If true, allows modifying the parameters of the request before it is sent to the LLM provider |
| enable_preview_features | boolean | If true, enables preview features - e.g. Azure O1 Models with streaming support.|
| redact_user_api_key_info | boolean | If true, redacts information about the user api key from logs [Proxy Logging](logging#redacting-userapikeyinfo) |
| langfuse_default_tags | array of strings | Default tags for Langfuse Logging. Use this if you want to control which LiteLLM-specific fields are logged as tags by the LiteLLM proxy. By default LiteLLM Proxy logs no LiteLLM-specific fields as tags. [Further docs](./logging#litellm-specific-tags-on-langfuse---cache_hit-cache_key) |
| set_verbose | boolean | If true, sets litellm.set_verbose=True to view verbose debug logs. DO NOT LEAVE THIS ON IN PRODUCTION |
| json_logs | boolean | If true, logs will be in json format. If you need to store the logs as JSON, just set the `litellm.json_logs = True`. We currently just log the raw POST request from litellm as a JSON [Further docs](./debugging) |
| default_fallbacks | array of strings | List of fallback models to use if a specific model group is misconfigured / bad. [Further docs](./reliability#default-fallbacks) |
| request_timeout | integer | The timeout for requests in seconds. If not set, the default value is `6000 seconds`. [For reference OpenAI Python SDK defaults to `600 seconds`.](https://github.com/openai/openai-python/blob/main/src/openai/_constants.py) |
| content_policy_fallbacks | array of objects | Fallbacks to use when a ContentPolicyViolationError is encountered. [Further docs](./reliability#content-policy-fallbacks) |
| context_window_fallbacks | array of objects | Fallbacks to use when a ContextWindowExceededError is encountered. [Further docs](./reliability#context-window-fallbacks) |
| cache | boolean | If true, enables caching. [Further docs](./caching) |
| cache_params | object | Parameters for the cache. [Further docs](./caching) |
| cache_params.type | string | The type of cache to initialize. Can be one of ["local", "redis", "redis-semantic", "s3", "disk", "qdrant-semantic"]. Defaults to "redis". [Furher docs](./caching) |
| cache_params.host | string | The host address for the Redis cache. Required if type is "redis". |
| cache_params.port | integer | The port number for the Redis cache. Required if type is "redis". |
| cache_params.password | string | The password for the Redis cache. Required if type is "redis". |
| cache_params.namespace | string | The namespace for the Redis cache. |
| cache_params.redis_startup_nodes | array of objects | Redis Cluster Settings. [Further docs](./caching) |
| cache_params.service_name | string | Redis Sentinel Settings. [Further docs](./caching) |
| cache_params.sentinel_nodes | array of arrays | Redis Sentinel Settings. [Further docs](./caching) |
| cache_params.ttl | integer | The time (in seconds) to store entries in cache. |
| cache_params.qdrant_semantic_cache_embedding_model | string | The embedding model to use for qdrant semantic cache. |
| cache_params.qdrant_collection_name | string | The name of the collection to use for qdrant semantic cache. |
| cache_params.qdrant_quantization_config | string | The quantization configuration for the qdrant semantic cache. |
| cache_params.similarity_threshold | float | The similarity threshold for the semantic cache. |
| cache_params.s3_bucket_name | string | The name of the S3 bucket to use for the semantic cache. |
| cache_params.s3_region_name | string | The region name for the S3 bucket. |
| cache_params.s3_aws_access_key_id | string | The AWS access key ID for the S3 bucket. |
| cache_params.s3_aws_secret_access_key | string | The AWS secret access key for the S3 bucket. |
| cache_params.s3_endpoint_url | string | Optional - The endpoint URL for the S3 bucket. |
| cache_params.supported_call_types | array of strings | The types of calls to cache. [Further docs](./caching) |
| cache_params.mode | string | The mode of the cache. [Further docs](./caching) |

### general_settings - Reference

| Name | Type | Description |
|------|------|-------------|
| completion_model | string | The default model to use for completions when `model` is not specified in the request |
| disable_spend_logs | boolean | If true, turns off writing each transaction to the database |
| disable_master_key_return | boolean | If true, turns off returning master key on UI. (checked on '/user/info' endpoint) |
| disable_retry_on_max_parallel_request_limit_error | boolean | If true, turns off retries when max parallel request limit is reached |
| disable_reset_budget | boolean | If true, turns off reset budget scheduled task |
| disable_adding_master_key_hash_to_db | boolean | If true, turns off storing master key hash in db |
| enable_jwt_auth | boolean | allow proxy admin to auth in via jwt tokens with 'litellm_proxy_admin' in claims. [Doc on JWT Tokens](token_auth) |
| enforce_user_param | boolean | If true, requires all OpenAI endpoint requests to have a 'user' param. [Doc on call hooks](call_hooks)|
| allowed_routes | array of strings | List of allowed proxy API routes a user can access [Doc on controlling allowed routes](enterprise#control-available-public-private-routes)|
| key_management_system | string | Specifies the key management system. [Doc Secret Managers](../secret) |
| master_key | string | The master key for the proxy [Set up Virtual Keys](virtual_keys) |
| database_url | string | The URL for the database connection [Set up Virtual Keys](virtual_keys) |
| database_connection_pool_limit | integer | The limit for database connection pool [Setting DB Connection Pool limit](#configure-db-pool-limits--connection-timeouts) |
| database_connection_timeout | integer | The timeout for database connections in seconds [Setting DB Connection Pool limit, timeout](#configure-db-pool-limits--connection-timeouts) |
| allow_requests_on_db_unavailable | boolean | If true, allows requests to succeed even if DB is unreachable. **Only use this if running LiteLLM in your VPC** This will allow requests to work even when LiteLLM cannot connect to the DB to verify a Virtual Key |
| custom_auth | string | Write your own custom authentication logic [Doc Custom Auth](virtual_keys#custom-auth) |
| max_parallel_requests | integer | The max parallel requests allowed per deployment |
| global_max_parallel_requests | integer | The max parallel requests allowed on the proxy overall |
| infer_model_from_keys | boolean | If true, infers the model from the provided keys |
| background_health_checks | boolean | If true, enables background health checks. [Doc on health checks](health) |
| health_check_interval | integer | The interval for health checks in seconds [Doc on health checks](health) |
| alerting | array of strings | List of alerting methods [Doc on Slack Alerting](alerting) |
| alerting_threshold | integer | The threshold for triggering alerts [Doc on Slack Alerting](alerting) |
| use_client_credentials_pass_through_routes | boolean | If true, uses client credentials for all pass-through routes. [Doc on pass through routes](pass_through) |
| health_check_details | boolean | If false, hides health check details (e.g. remaining rate limit). [Doc on health checks](health) |
| public_routes | List[str] | (Enterprise Feature) Control list of public routes |
| alert_types | List[str] | Control list of alert types to send to slack (Doc on alert types)[./alerting.md] |
| enforced_params | List[str] | (Enterprise Feature) List of params that must be included in all requests to the proxy |
| enable_oauth2_auth | boolean | (Enterprise Feature) If true, enables oauth2.0 authentication |
| use_x_forwarded_for | str | If true, uses the X-Forwarded-For header to get the client IP address |
| service_account_settings | List[Dict[str, Any]] | Set `service_account_settings` if you want to create settings that only apply to service account keys (Doc on service accounts)[./service_accounts.md] | 
| image_generation_model | str | The default model to use for image generation - ignores model set in request |
| store_model_in_db | boolean | If true, allows `/model/new` endpoint to store model information in db. Endpoint disabled by default. [Doc on `/model/new` endpoint](./model_management.md#create-a-new-model) |
| max_request_size_mb | int | The maximum size for requests in MB. Requests above this size will be rejected. |
| max_response_size_mb | int | The maximum size for responses in MB. LLM Responses above this size will not be sent. |
| proxy_budget_rescheduler_min_time | int | The minimum time (in seconds) to wait before checking db for budget resets. **Default is 597 seconds** |
| proxy_budget_rescheduler_max_time | int | The maximum time (in seconds) to wait before checking db for budget resets. **Default is 605 seconds** |
| proxy_batch_write_at | int | Time (in seconds) to wait before batch writing spend logs to the db. **Default is 10 seconds** |
| alerting_args | dict | Args for Slack Alerting [Doc on Slack Alerting](./alerting.md) |
| custom_key_generate | str | Custom function for key generation [Doc on custom key generation](./virtual_keys.md#custom--key-generate) |
| allowed_ips | List[str] | List of IPs allowed to access the proxy. If not set, all IPs are allowed. |
| embedding_model | str | The default model to use for embeddings - ignores model set in request |
| default_team_disabled | boolean | If true, users cannot create 'personal' keys (keys with no team_id). |
| alert_to_webhook_url | Dict[str] | [Specify a webhook url for each alert type.](./alerting.md#set-specific-slack-channels-per-alert-type) |
| key_management_settings | List[Dict[str, Any]] | Settings for key management system (e.g. AWS KMS, Azure Key Vault) [Doc on key management](../secret.md) |
| allow_user_auth | boolean | (Deprecated) old approach for user authentication. |
| user_api_key_cache_ttl | int | The time (in seconds) to cache user api keys in memory. |
| disable_prisma_schema_update | boolean | If true, turns off automatic schema updates to DB |
| litellm_key_header_name | str | If set, allows passing LiteLLM keys as a custom header. [Doc on custom headers](./virtual_keys.md#custom-headers) |
| moderation_model | str | The default model to use for moderation. |
| custom_sso | str | Path to a python file that implements custom SSO logic. [Doc on custom SSO](./custom_sso.md) |
| allow_client_side_credentials | boolean | If true, allows passing client side credentials to the proxy. (Useful when testing finetuning models) [Doc on client side credentials](./virtual_keys.md#client-side-credentials) |
| admin_only_routes | List[str] | (Enterprise Feature) List of routes that are only accessible to admin users. [Doc on admin only routes](./enterprise#control-available-public-private-routes) |
| use_azure_key_vault | boolean | If true, load keys from azure key vault | 
| use_google_kms | boolean | If true, load keys from google kms |
| spend_report_frequency | str | Specify how often you want a Spend Report to be sent (e.g. "1d", "2d", "30d") [More on this](./alerting.md#spend-report-frequency) |
| ui_access_mode | Literal["admin_only"] | If set, restricts access to the UI to admin users only. [Docs](./ui.md#restrict-ui-access) |
| litellm_jwtauth | Dict[str, Any] | Settings for JWT authentication. [Docs](./token_auth.md) |
| litellm_license | str | The license key for the proxy. [Docs](../enterprise.md#how-does-deployment-with-enterprise-license-work) |
| oauth2_config_mappings | Dict[str, str] | Define the OAuth2 config mappings | 
| pass_through_endpoints | List[Dict[str, Any]] | Define the pass through endpoints. [Docs](./pass_through) |
| enable_oauth2_proxy_auth | boolean | (Enterprise Feature) If true, enables oauth2.0 authentication |
| forward_openai_org_id | boolean | If true, forwards the OpenAI Organization ID to the backend LLM call (if it's OpenAI). |
| forward_client_headers_to_llm_api | boolean | If true, forwards the client headers (any `x-` headers) to the backend LLM call |

### router_settings - Reference

```yaml
router_settings:
  routing_strategy: usage-based-routing-v2 # Literal["simple-shuffle", "least-busy", "usage-based-routing","latency-based-routing"], default="simple-shuffle"
  redis_host: <your-redis-host>           # string
  redis_password: <your-redis-password>   # string
  redis_port: <your-redis-port>           # string
  enable_pre_call_check: true             # bool - Before call is made check if a call is within model context window 
  allowed_fails: 3 # cooldown model if it fails > 1 call in a minute. 
  cooldown_time: 30 # (in seconds) how long to cooldown model if fails/min > allowed_fails
  disable_cooldowns: True                  # bool - Disable cooldowns for all models 
  enable_tag_filtering: True                # bool - Use tag based routing for requests
  retry_policy: {                          # Dict[str, int]: retry policy for different types of exceptions
    "AuthenticationErrorRetries": 3,
    "TimeoutErrorRetries": 3,
    "RateLimitErrorRetries": 3,
    "ContentPolicyViolationErrorRetries": 4,
    "InternalServerErrorRetries": 4
  }
  allowed_fails_policy: {
    "BadRequestErrorAllowedFails": 1000, # Allow 1000 BadRequestErrors before cooling down a deployment
    "AuthenticationErrorAllowedFails": 10, # int 
    "TimeoutErrorAllowedFails": 12, # int 
    "RateLimitErrorAllowedFails": 10000, # int 
    "ContentPolicyViolationErrorAllowedFails": 15, # int 
    "InternalServerErrorAllowedFails": 20, # int 
  }
  content_policy_fallbacks=[{"claude-2": ["my-fallback-model"]}] # List[Dict[str, List[str]]]: Fallback model for content policy violations
  fallbacks=[{"claude-2": ["my-fallback-model"]}] # List[Dict[str, List[str]]]: Fallback model for all errors
```

| Name | Type | Description |
|------|------|-------------|
| routing_strategy | string | The strategy used for routing requests. Options: "simple-shuffle", "least-busy", "usage-based-routing", "latency-based-routing". Default is "simple-shuffle". [More information here](../routing) |
| redis_host | string | The host address for the Redis server. **Only set this if you have multiple instances of LiteLLM Proxy and want current tpm/rpm tracking to be shared across them** |
| redis_password | string | The password for the Redis server. **Only set this if you have multiple instances of LiteLLM Proxy and want current tpm/rpm tracking to be shared across them** |
| redis_port | string | The port number for the Redis server. **Only set this if you have multiple instances of LiteLLM Proxy and want current tpm/rpm tracking to be shared across them**|
| enable_pre_call_check | boolean | If true, checks if a call is within the model's context window before making the call. [More information here](reliability) |
| content_policy_fallbacks | array of objects | Specifies fallback models for content policy violations. [More information here](reliability) |
| fallbacks | array of objects | Specifies fallback models for all types of errors. [More information here](reliability) |
| enable_tag_filtering | boolean | If true, uses tag based routing for requests [Tag Based Routing](tag_routing) |
| cooldown_time | integer | The duration (in seconds) to cooldown a model if it exceeds the allowed failures. |
| disable_cooldowns | boolean | If true, disables cooldowns for all models. [More information here](reliability) |
| retry_policy | object | Specifies the number of retries for different types of exceptions. [More information here](reliability) |
| allowed_fails | integer | The number of failures allowed before cooling down a model. [More information here](reliability) |
| allowed_fails_policy | object | Specifies the number of allowed failures for different error types before cooling down a deployment. [More information here](reliability) |


### environment variables - Reference

| Name | Description |
|------|-------------|
| ACTIONS_ID_TOKEN_REQUEST_TOKEN | Token for requesting ID in GitHub Actions
| ACTIONS_ID_TOKEN_REQUEST_URL | URL for requesting ID token in GitHub Actions
| AISPEND_ACCOUNT_ID | Account ID for AI Spend
| AISPEND_API_KEY | API Key for AI Spend
| ALLOWED_EMAIL_DOMAINS | List of email domains allowed for access
| ARIZE_API_KEY | API key for Arize platform integration
| ARIZE_SPACE_KEY | Space key for Arize platform
| ARGILLA_BATCH_SIZE | Batch size for Argilla logging
| ARGILLA_API_KEY | API key for Argilla platform
| ARGILLA_SAMPLING_RATE | Sampling rate for Argilla logging
| ARGILLA_DATASET_NAME | Dataset name for Argilla logging
| ARGILLA_BASE_URL | Base URL for Argilla service
| ATHINA_API_KEY | API key for Athina service
| AUTH_STRATEGY | Strategy used for authentication (e.g., OAuth, API key)
| AWS_ACCESS_KEY_ID | Access Key ID for AWS services
| AWS_PROFILE_NAME | AWS CLI profile name to be used
| AWS_REGION_NAME | Default AWS region for service interactions
| AWS_ROLE_NAME | Role name for AWS IAM usage
| AWS_SECRET_ACCESS_KEY | Secret Access Key for AWS services
| AWS_SESSION_NAME | Name for AWS session
| AWS_WEB_IDENTITY_TOKEN | Web identity token for AWS
| AZURE_API_VERSION | Version of the Azure API being used
| AZURE_AUTHORITY_HOST | Azure authority host URL
| AZURE_CLIENT_ID | Client ID for Azure services
| AZURE_CLIENT_SECRET | Client secret for Azure services
| AZURE_FEDERATED_TOKEN_FILE | File path to Azure federated token
| AZURE_KEY_VAULT_URI | URI for Azure Key Vault
| AZURE_TENANT_ID | Tenant ID for Azure Active Directory
| BERRISPEND_ACCOUNT_ID | Account ID for BerriSpend service
| BRAINTRUST_API_KEY | API key for Braintrust integration
| CIRCLE_OIDC_TOKEN | OpenID Connect token for CircleCI
| CIRCLE_OIDC_TOKEN_V2 | Version 2 of the OpenID Connect token for CircleCI
| CONFIG_FILE_PATH | File path for configuration file
| CUSTOM_TIKTOKEN_CACHE_DIR | Custom directory for Tiktoken cache
| DATABASE_HOST | Hostname for the database server
| DATABASE_NAME | Name of the database
| DATABASE_PASSWORD | Password for the database user
| DATABASE_PORT | Port number for database connection
| DATABASE_SCHEMA | Schema name used in the database
| DATABASE_URL | Connection URL for the database
| DATABASE_USER | Username for database connection
| DATABASE_USERNAME | Alias for database user
| DATABRICKS_API_BASE | Base URL for Databricks API
| DD_BASE_URL | Base URL for Datadog integration
| DATADOG_BASE_URL | (Alternative to DD_BASE_URL) Base URL for Datadog integration
| _DATADOG_BASE_URL | (Alternative to DD_BASE_URL) Base URL for Datadog integration
| DD_API_KEY | API key for Datadog integration
| DD_SITE | Site URL for Datadog (e.g., datadoghq.com)
| DD_SOURCE | Source identifier for Datadog logs
| DD_ENV | Environment identifier for Datadog logs. Only supported for `datadog_llm_observability` callback
| DEBUG_OTEL | Enable debug mode for OpenTelemetry
| DIRECT_URL | Direct URL for service endpoint
| DISABLE_ADMIN_UI | Toggle to disable the admin UI
| DISABLE_SCHEMA_UPDATE | Toggle to disable schema updates
| DOCS_DESCRIPTION | Description text for documentation pages
| DOCS_FILTERED | Flag indicating filtered documentation
| DOCS_TITLE | Title of the documentation pages
| EMAIL_SUPPORT_CONTACT | Support contact email address
| GCS_BUCKET_NAME | Name of the Google Cloud Storage bucket
| GCS_PATH_SERVICE_ACCOUNT | Path to the Google Cloud service account JSON file
| GCS_FLUSH_INTERVAL | Flush interval for GCS logging (in seconds). Specify how often you want a log to be sent to GCS. **Default is 20 seconds**
| GCS_BATCH_SIZE | Batch size for GCS logging. Specify after how many logs you want to flush to GCS. If `BATCH_SIZE` is set to 10, logs are flushed every 10 logs. **Default is 2048**
| GENERIC_AUTHORIZATION_ENDPOINT | Authorization endpoint for generic OAuth providers
| GENERIC_CLIENT_ID | Client ID for generic OAuth providers
| GENERIC_CLIENT_SECRET | Client secret for generic OAuth providers
| GENERIC_CLIENT_STATE | State parameter for generic client authentication
| GENERIC_INCLUDE_CLIENT_ID | Include client ID in requests for OAuth
| GENERIC_SCOPE | Scope settings for generic OAuth providers
| GENERIC_TOKEN_ENDPOINT | Token endpoint for generic OAuth providers
| GENERIC_USER_DISPLAY_NAME_ATTRIBUTE | Attribute for user's display name in generic auth
| GENERIC_USER_EMAIL_ATTRIBUTE | Attribute for user's email in generic auth
| GENERIC_USER_FIRST_NAME_ATTRIBUTE | Attribute for user's first name in generic auth
| GENERIC_USER_ID_ATTRIBUTE | Attribute for user ID in generic auth
| GENERIC_USER_LAST_NAME_ATTRIBUTE | Attribute for user's last name in generic auth
| GENERIC_USER_PROVIDER_ATTRIBUTE | Attribute specifying the user's provider
| GENERIC_USER_ROLE_ATTRIBUTE | Attribute specifying the user's role
| GENERIC_USERINFO_ENDPOINT | Endpoint to fetch user information in generic OAuth
| GALILEO_BASE_URL | Base URL for Galileo platform
| GALILEO_PASSWORD | Password for Galileo authentication
| GALILEO_PROJECT_ID | Project ID for Galileo usage
| GALILEO_USERNAME | Username for Galileo authentication
| GREENSCALE_API_KEY | API key for Greenscale service
| GREENSCALE_ENDPOINT | Endpoint URL for Greenscale service
| GOOGLE_APPLICATION_CREDENTIALS | Path to Google Cloud credentials JSON file
| GOOGLE_CLIENT_ID | Client ID for Google OAuth
| GOOGLE_CLIENT_SECRET | Client secret for Google OAuth
| GOOGLE_KMS_RESOURCE_NAME | Name of the resource in Google KMS
| HF_API_BASE | Base URL for Hugging Face API
| HELICONE_API_KEY | API key for Helicone service
| HUGGINGFACE_API_BASE | Base URL for Hugging Face API
| IAM_TOKEN_DB_AUTH | IAM token for database authentication
| JSON_LOGS | Enable JSON formatted logging
| JWT_AUDIENCE | Expected audience for JWT tokens
| JWT_PUBLIC_KEY_URL | URL to fetch public key for JWT verification
| LAGO_API_BASE | Base URL for Lago API
| LAGO_API_CHARGE_BY | Parameter to determine charge basis in Lago
| LAGO_API_EVENT_CODE | Event code for Lago API events
| LAGO_API_KEY | API key for accessing Lago services
| LANGFUSE_DEBUG | Toggle debug mode for Langfuse
| LANGFUSE_FLUSH_INTERVAL | Interval for flushing Langfuse logs
| LANGFUSE_HOST | Host URL for Langfuse service
| LANGFUSE_PUBLIC_KEY | Public key for Langfuse authentication
| LANGFUSE_RELEASE | Release version of Langfuse integration
| LANGFUSE_SECRET_KEY | Secret key for Langfuse authentication
| LANGSMITH_API_KEY | API key for Langsmith platform
| LANGSMITH_BASE_URL | Base URL for Langsmith service
| LANGSMITH_BATCH_SIZE | Batch size for operations in Langsmith
| LANGSMITH_DEFAULT_RUN_NAME | Default name for Langsmith run
| LANGSMITH_PROJECT | Project name for Langsmith integration
| LANGSMITH_SAMPLING_RATE | Sampling rate for Langsmith logging
| LANGTRACE_API_KEY | API key for Langtrace service
| LITERAL_API_KEY | API key for Literal integration
| LITERAL_API_URL | API URL for Literal service
| LITERAL_BATCH_SIZE | Batch size for Literal operations
| LITELLM_DONT_SHOW_FEEDBACK_BOX | Flag to hide feedback box in LiteLLM UI
| LITELLM_DROP_PARAMS | Parameters to drop in LiteLLM requests
| LITELLM_EMAIL | Email associated with LiteLLM account
| LITELLM_GLOBAL_MAX_PARALLEL_REQUEST_RETRIES | Maximum retries for parallel requests in LiteLLM
| LITELLM_GLOBAL_MAX_PARALLEL_REQUEST_RETRY_TIMEOUT | Timeout for retries of parallel requests in LiteLLM
| LITELLM_HOSTED_UI | URL of the hosted UI for LiteLLM
| LITELLM_LICENSE | License key for LiteLLM usage
| LITELLM_LOCAL_MODEL_COST_MAP | Local configuration for model cost mapping in LiteLLM
| LITELLM_LOG | Enable detailed logging for LiteLLM
| LITELLM_MODE | Operating mode for LiteLLM (e.g., production, development)
| LITELLM_SALT_KEY | Salt key for encryption in LiteLLM
| LITELLM_SECRET_AWS_KMS_LITELLM_LICENSE | AWS KMS encrypted license for LiteLLM
| LITELLM_TOKEN | Access token for LiteLLM integration
| LOGFIRE_TOKEN | Token for Logfire logging service
| MICROSOFT_CLIENT_ID | Client ID for Microsoft services
| MICROSOFT_CLIENT_SECRET | Client secret for Microsoft services
| MICROSOFT_TENANT | Tenant ID for Microsoft Azure
| NO_DOCS | Flag to disable documentation generation
| NO_PROXY | List of addresses to bypass proxy
| OAUTH_TOKEN_INFO_ENDPOINT | Endpoint for OAuth token info retrieval
| OPENAI_API_BASE | Base URL for OpenAI API
| OPENAI_API_KEY | API key for OpenAI services
| OPENAI_ORGANIZATION | Organization identifier for OpenAI
| OPENID_BASE_URL | Base URL for OpenID Connect services
| OPENID_CLIENT_ID | Client ID for OpenID Connect authentication
| OPENID_CLIENT_SECRET | Client secret for OpenID Connect authentication
| OPENMETER_API_ENDPOINT | API endpoint for OpenMeter integration
| OPENMETER_API_KEY | API key for OpenMeter services
| OPENMETER_EVENT_TYPE | Type of events sent to OpenMeter
| OTEL_ENDPOINT | OpenTelemetry endpoint for traces
| OTEL_ENVIRONMENT_NAME | Environment name for OpenTelemetry
| OTEL_EXPORTER | Exporter type for OpenTelemetry
| OTEL_HEADERS | Headers for OpenTelemetry requests
| OTEL_SERVICE_NAME | Service name identifier for OpenTelemetry
| OTEL_TRACER_NAME | Tracer name for OpenTelemetry tracing
| PREDIBASE_API_BASE | Base URL for Predibase API
| PRESIDIO_ANALYZER_API_BASE | Base URL for Presidio Analyzer service
| PRESIDIO_ANONYMIZER_API_BASE | Base URL for Presidio Anonymizer service
| PROMETHEUS_URL | URL for Prometheus service
| PROMPTLAYER_API_KEY | API key for PromptLayer integration
| PROXY_ADMIN_ID | Admin identifier for proxy server
| PROXY_BASE_URL | Base URL for proxy service
| PROXY_LOGOUT_URL | URL for logging out of the proxy service
| PROXY_MASTER_KEY | Master key for proxy authentication
| QDRANT_API_BASE | Base URL for Qdrant API
| QDRANT_API_KEY | API key for Qdrant service
| QDRANT_URL | Connection URL for Qdrant database
| REDIS_HOST | Hostname for Redis server
| REDIS_PASSWORD | Password for Redis service
| REDIS_PORT | Port number for Redis server
| SERVER_ROOT_PATH | Root path for the server application
| SET_VERBOSE | Flag to enable verbose logging
| SLACK_DAILY_REPORT_FREQUENCY | Frequency of daily Slack reports (e.g., daily, weekly)
| SLACK_WEBHOOK_URL | Webhook URL for Slack integration
| SMTP_HOST | Hostname for the SMTP server
| SMTP_PASSWORD | Password for SMTP authentication
| SMTP_PORT | Port number for SMTP server
| SMTP_SENDER_EMAIL | Email address used as the sender in SMTP transactions
| SMTP_SENDER_LOGO | Logo used in emails sent via SMTP
| SMTP_TLS | Flag to enable or disable TLS for SMTP connections
| SMTP_USERNAME | Username for SMTP authentication
| SPEND_LOGS_URL | URL for retrieving spend logs
| SSL_CERTIFICATE | Path to the SSL certificate file
| SSL_VERIFY | Flag to enable or disable SSL certificate verification
| SUPABASE_KEY | API key for Supabase service
| SUPABASE_URL | Base URL for Supabase instance
| TEST_EMAIL_ADDRESS | Email address used for testing purposes
| UI_LOGO_PATH | Path to the logo image used in the UI
| UI_PASSWORD | Password for accessing the UI
| UI_USERNAME | Username for accessing the UI
| UPSTREAM_LANGFUSE_DEBUG | Flag to enable debugging for upstream Langfuse
| UPSTREAM_LANGFUSE_HOST | Host URL for upstream Langfuse service
| UPSTREAM_LANGFUSE_PUBLIC_KEY | Public key for upstream Langfuse authentication
| UPSTREAM_LANGFUSE_RELEASE | Release version identifier for upstream Langfuse
| UPSTREAM_LANGFUSE_SECRET_KEY | Secret key for upstream Langfuse authentication
| USE_AWS_KMS | Flag to enable AWS Key Management Service for encryption
| WEBHOOK_URL | URL for receiving webhooks from external services
## Extras


### Disable Swagger UI 

To disable the Swagger docs from the base url, set 

```env
NO_DOCS="True"
```

in your environment, and restart the proxy. 

### Use CONFIG_FILE_PATH for proxy (Easier Azure container deployment)

1. Setup config.yaml

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: gpt-3.5-turbo
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
   ghcr.io/berriai/litellm-database:main-latest --detailed_debug
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
   ghcr.io/berriai/litellm-database:main-latest
```
</TabItem>
</Tabs>