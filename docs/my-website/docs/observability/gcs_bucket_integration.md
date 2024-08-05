import Image from '@theme/IdealImage';

# 🪣 Google Cloud Storage Buckets - Logging LLM Input/Output

Log LLM Logs to [Google Cloud Storage Buckets](https://cloud.google.com/storage?hl=en)

:::info

✨ This is an Enterprise only feature [Get Started with Enterprise here](https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat)

:::


### Usage

1. Add `gcs_bucket` to LiteLLM Config.yaml
```yaml
model_list:
- litellm_params:
    api_base: https://openai-function-calling-workers.tasslexyz.workers.dev/
    api_key: my-fake-key
    model: openai/my-fake-model
  model_name: fake-openai-endpoint

litellm_settings:
  callbacks: ["gcs_bucket"] # 👈 KEY CHANGE # 👈 KEY CHANGE
```

2. Set required env variables

```shell
GCS_BUCKET_NAME="<your-gcs-bucket-name>"
GCS_PATH_SERVICE_ACCOUNT="/Users/ishaanjaffer/Downloads/adroit-crow-413218-a956eef1a2a8.json" # Add path to service account.json
```

3. Start Proxy

```
litellm --config /path/to/config.yaml
```

4. Test it! 

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "fake-openai-endpoint",
      "messages": [
        {
          "role": "user",
          "content": "what llm are you"
        }
      ],
    }
'
```


## Expected Logs on GCS Buckets

<Image img={require('../../img/gcs_bucket.png')} />

### Fields Logged on GCS Buckets

Example payload of a `/chat/completion` request logged on GCS
```json
{
  "request_kwargs": {
    "model": "gpt-3.5-turbo",
    "messages": [
      {
        "role": "user",
        "content": "This is a test"
      }
    ],
    "optional_params": {
      "temperature": 0.7,
      "max_tokens": 10,
      "user": "ishaan-2",
      "extra_body": {}
    }
  },
  "response_obj": {
    "id": "chatcmpl-bd836a8c-89bc-4abd-bee5-e3f1ebfdb541",
    "choices": [
      {
        "finish_reason": "stop",
        "index": 0,
        "message": {
          "content": "Hi!",
          "role": "assistant",
          "tool_calls": null,
          "function_call": null
        }
      }
    ],
    "created": 1722868456,
    "model": "gpt-3.5-turbo",
    "object": "chat.completion",
    "system_fingerprint": null,
    "usage": {
      "prompt_tokens": 10,
      "completion_tokens": 20,
      "total_tokens": 30
    }
  },
  "start_time": "2024-08-05 07:34:16",
  "end_time": "2024-08-05 07:34:16"
}
```

## Getting `service_account.json` from Google Cloud Console

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Search for IAM & Admin
3. Click on Service Accounts
4. Select a Service Account
5. Click on 'Keys' -> Add Key -> Create New Key -> JSON
6. Save the JSON file and add the path to `GCS_PATH_SERVICE_ACCOUNT`

## Support & Talk to Founders

- [Schedule Demo 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
- [Community Discord 💭](https://discord.gg/wuPM9dRgDw)
- Our numbers 📞 +1 (770) 8783-106 / ‭+1 (412) 618-6238‬
- Our emails ✉️ ishaan@berri.ai / krrish@berri.ai
