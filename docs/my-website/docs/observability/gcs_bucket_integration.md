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
    "request_id": "chatcmpl-3946ddc2-bcfe-43f6-9b8e-2427951de85c",
    "call_type": "acompletion",
    "api_key": "",
    "cache_hit": "None",
    "startTime": "2024-08-01T14:27:12.563246",
    "endTime": "2024-08-01T14:27:12.572709",
    "completionStartTime": "2024-08-01T14:27:12.572709",
    "model": "gpt-3.5-turbo",
    "user": "",
    "team_id": "",
    "metadata": "{}",
    "cache_key": "Cache OFF",
    "spend": 0.000054999999999999995,
    "total_tokens": 30,
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "request_tags": "[]",
    "end_user": "ishaan-2",
    "api_base": "",
    "model_group": "",
    "model_id": "",
    "requester_ip_address": null,
    "output": [
        "{\"finish_reason\":\"stop\",\"index\":0,\"message\":{\"content\":\"Hi!\",\"role\":\"assistant\",\"tool_calls\":null,\"function_call\":null}}"
    ]
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
