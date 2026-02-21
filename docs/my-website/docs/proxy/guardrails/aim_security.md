import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Aim Security

## Quick Start
### 1. Create a new Aim Guard

Go to [Aim Application](https://app.aim.security/inventory/custom-ai-apps) and create a new guard.

When prompted, select API option, and name your guard.


:::note 
In case you want to host your guard on-premise, you can enable this option
by [installing Aim Outpost](https://app.aim.security/settings/on-prem-deployment) prior to creating the guard.
:::

### 2. Configure your Aim Guard policies

In the newly created guard's page, you can find a reference to the prompt policy center of this guard.

You can decide which detections will be enabled, and set the threshold for each detection.

:::info 
When using LiteLLM with virtual keys, key-specific policies can be set directly in Aim's guards page by specifying the virtual key alias when creating the guard.

Only the aliases of your virtual keys (and not the actual key secrets) will be sent to Aim.
:::

### 3. Add Aim Guardrail on your LiteLLM config.yaml 

Define your guardrails under the `guardrails` section
```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: aim-protected-app
    litellm_params:
      guardrail: aim
      mode: [pre_call, post_call] # "During_call" is also available
      api_key: os.environ/AIM_API_KEY
      api_base: os.environ/AIM_API_BASE # Optional, use only when using a self-hosted Aim Outpost
      ssl_verify: False # Optional, set to False to disable SSL verification or a string path to a custom CA bundle
```

Under the `api_key`, insert the API key you were issued. The key can be found in the guard's page.
You can also set `AIM_API_KEY` as an environment variable.

By default, the `api_base` is set to `https://api.aim.security`. If you are using a self-hosted Aim Outpost, you can set the `api_base` to your Outpost's URL.

### 4. Start LiteLLM Gateway
```shell
litellm --config config.yaml
```

### 5. Make your first request

:::note
The following example depends on enabling *PII* detection in your guard.
You can adjust the request content to match different guard's policies.
:::

<Tabs>
<TabItem label="Successfully blocked request" value = "blocked">

:::note
When using LiteLLM with virtual keys, an `Authorization` header with the virtual key is required.
:::

```shell
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "hi my email is ishaan@berri.ai"}
    ],
    "guardrails": ["aim-protected-app"]
  }'
```

If configured correctly, since `ishaan@berri.ai` would be detected by the Aim Guard as PII, you'll receive a response similar to the following with a `400 Bad Request` status code:

```json
{
  "error": {
    "message": "\"ishaan@berri.ai\" detected as email",
    "type": "None",
    "param": "None",
    "code": "400"
  }
}
```

</TabItem>

<TabItem label="Successfully permitted request" value = "allowed">

:::note
When using LiteLLM with virtual keys, an `Authorization` header with the virtual key is required.
:::

```shell
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "hi what is the weather"}
    ],
    "guardrails": ["aim-protected-app"]
  }'
```

The above request should not be blocked, and you should receive a regular LLM response (simplified for brevity):

```json
{
  "model": "gpt-3.5-turbo-0125",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "I can’t provide live weather updates without the internet. Let me know if you’d like general weather trends for a location and season instead!",
        "role": "assistant"
      }
    }
  ]
}
```

</TabItem>


</Tabs>

## Advanced

Aim Guard provides user-specific Guardrail policies, enabling you to apply tailored policies to individual users.
To utilize this feature, include the end-user's email in the request payload by setting the `x-aim-user-email` header of your request.

```shell
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-aim-user-email: ishaan@berri.ai" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "hi what is the weather"}
    ],
    "guardrails": ["aim-protected-app"]
  }'
```
