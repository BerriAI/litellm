import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Use LiteLLM AI Gateway with Aporia Guardrails

In this tutorial we will use LiteLLM Proxy with Aporia to detect PII in requests and profanity in responses

## 1. Setup guardrails on Aporia

### Create Aporia Projects

Create two projects on [Aporia](https://guardrails.aporia.com/)

1. Pre LLM API Call - Set all the policies you want to run on pre LLM API call 
2. Post LLM API Call - Set all the policies you want to run post LLM API call


<Image img={require('../../img/aporia_projs.png')} />


### Pre-Call: Detect PII

Add the `PII - Prompt` to your Pre LLM API Call project

<Image img={require('../../img/aporia_pre.png')} />

### Post-Call: Detect Profanity in Responses

Add the `Toxicity - Response` to your Post LLM API Call project

<Image img={require('../../img/aporia_post.png')} />


## 2. Define Guardrails on your LiteLLM config.yaml 

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  pre_call:  # guardrail only runs on input before LLM API call
      guardrail: "aporia"                   # supported values ["aporia", "bedrock", "lakera"]
      api_key: os.environ/APORIA_API_KEY_1
      api_base: os.environ/APORIA_API_BASE_1
  post_call: # guardrail only runs on output after LLM API call
      guardrail: "aporia"                  # supported values ["aporia", "bedrock", "lakera"]
      api_key: os.environ/APORIA_API_KEY_2
      api_base: os.environ/APORIA_API_BASE_2
```

## 3. Start LiteLLM Gateway 


```shell
litellm --config config.yaml --detailed_debug
```

## 4. Test request 

<Tabs>
<TabItem label="Fails Guardrail" value = "not-allowed">

Expect this to fail since since `ishaan@berri.ai` in the request is PII

```shell
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "hi my email is ishaan@berri.ai"}
    ]
  }'
```

</TabItem>

<TabItem label="Success" value = "allowed">

```shell
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "hi what is the weather?"}
    ]
  }'
```

</TabItem>


</Tabs>

## Advanced
### Control Guardrails per Project (API Key)

Use this to control what guardrail/s run per project. In this tutorial we only want the following guardrails to run for 1 project
- pre_call: aporia
- post_call: aporia

**Step 1** Create Key with guardrail settings

<Tabs>
<TabItem value="/key/generate" label="/key/generate">

```shell
curl -X POST 'http://0.0.0.0:4000/key/generate' \
    -H 'Authorization: Bearer sk-1234' \
    -H 'Content-Type: application/json' \
    -D '{
        "guardrails": {
            "pre_call": ["aporia"],
            "post_call": ["aporia"]
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
        "guardrails": {
            "pre_call": ["aporia"],
            "post_call": ["aporia"]
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



