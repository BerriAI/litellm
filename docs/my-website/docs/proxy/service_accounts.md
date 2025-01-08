import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import Image from '@theme/IdealImage';

# [Beta] Service Accounts

Use this if you want to create Virtual Keys that are not owned by a specific user but instead created for production projects

## Usage

### 1. Set settings for Service Accounts

Set `service_account_settings` if you want to create settings that only apply to service account keys

```yaml
general_settings:
    service_account_settings: 
        enforced_params: ["user"] # this means the "user" param is enforced for all requests made through any service account keys
```

### 2. Create Service Account Key on LiteLLM Proxy Admin UI

<Image img={require('../../img/create_service_account.png')} />

### 3. Test Service Account Key 

<Tabs>

<TabItem value="Unsuccessful call" label="Unsuccessful call">


```shell
curl --location 'http://localhost:4000/chat/completions' \
    --header 'Authorization: Bearer <sk-your-service-account>' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "gpt-3.5-turbo",
    "messages": [
        {
        "role": "user",
        "content": "hello"
        }
    ]
}'
```

Expected Response

```json
{
  "error": {
    "message": "BadRequest please pass param=user in request body. This is a required param for service account",
    "type": "bad_request_error",
    "param": "user",
    "code": "400"
  }
}
```

</TabItem>

<TabItem value="Successful call" label="Successful call">


```shell
curl --location 'http://localhost:4000/chat/completions' \
    --header 'Authorization: Bearer <sk-your-service-account>' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "gpt-3.5-turbo",
    "messages": [
        {
        "role": "user",
        "content": "hello"
        }
    ],
    "user": "test-user"
}'
```

Expected Response

```json
{
  "id": "chatcmpl-ad9595c7e3784a6783b469218d92d95c",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "\n\nHello there, how may I assist you today?",
        "role": "assistant",
        "tool_calls": null,
        "function_call": null
      }
    }
  ],
  "created": 1677652288,
  "model": "gpt-3.5-turbo-0125",
  "object": "chat.completion",
  "system_fingerprint": "fp_44709d6fcb",
  "usage": {
    "completion_tokens": 12,
    "prompt_tokens": 9,
    "total_tokens": 21,
    "completion_tokens_details": null
  },
  "service_tier": null
}
```

</TabItem>

</Tabs>

