import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Guardrails AI

Use Guardrails AI ([guardrailsai.com](https://www.guardrailsai.com/)) to add checks to LLM output.

## Pre-requisites

- Setup Guardrails AI Server. [quick start](https://www.guardrailsai.com/docs/getting_started/guardrails_server)

## Usage

1. Setup config.yaml 

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: "guardrails_ai-guard"
    litellm_params:
      guardrail: guardrails_ai
      guard_name: "detect-secrets-guard"            # 👈 Guardrail AI guard name
      mode: "pre_call"
      guardrails_ai_api_input_format: "llmOutput"   # 👈 This is the only option that currently works (and it is a default), use it for both pre_call and post_call hooks
      api_base: os.environ/GUARDRAILS_AI_API_BASE   # 👈 Guardrails AI API Base. Defaults to "http://0.0.0.0:8000"
```

2. Start LiteLLM Gateway 

```shell
litellm --config config.yaml --detailed_debug
```

3. Test request 

**[Langchain, OpenAI SDK Usage Examples](../proxy/user_keys#request-format)**

```shell
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-npnwjPQciVRok5yNZgKmFQ" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "hi my email is ishaan@berri.ai"}
    ],
    "guardrails": ["guardrails_ai-guard"]
  }'
```


## ✨ Control Guardrails per Project (API Key)

:::info

✨ This is an Enterprise only feature [Contact us to get a free trial](https://calendly.com/d/cx9p-5yf-2nm/litellm-introductions)

:::

Use this to control what guardrails run per project. In this tutorial we only want the following guardrails to run for 1 project (API Key)
- `guardrails`: ["aporia-pre-guard", "aporia-post-guard"]

**Step 1** Create Key with guardrail settings

<Tabs>
<TabItem value="/key/generate" label="/key/generate">

```shell
curl -X POST 'http://0.0.0.0:4000/key/generate' \
    -H 'Authorization: Bearer sk-1234' \
    -H 'Content-Type: application/json' \
    -d '{
            "guardrails": ["guardrails_ai-guard"]
        }
    }'
```

</TabItem>
<TabItem value="/key/update" label="/key/update">

```shell
curl --location 'http://0.0.0.0:4000/key/update' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
        "key": "sk-jNm1Zar7XfNdZXp49Z1kSQ",
        "guardrails": ["guardrails_ai-guard"]
        }
}'
```

</TabItem>
</Tabs>

**Step 2** Test it with new key

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
    --header 'Authorization: Bearer sk-jNm1Zar7XfNdZXp49Z1kSQ' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "gpt-3.5-turbo",
    "messages": [
        {
        "role": "user",
        "content": "my email is ishaan@berri.ai"
        }
    ]
}'
```



