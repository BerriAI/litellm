
import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Logging GCS, s3 Buckets

LiteLLM Supports Logging to the following Cloud Buckets
- (Enterprise) ✨ [Google Cloud Storage Buckets](#logging-proxy-inputoutput-to-google-cloud-storage-buckets)
- (Free OSS) [Amazon s3 Buckets](#logging-proxy-inputoutput---s3-buckets) 

## Google Cloud Storage Buckets

Log LLM Logs to [Google Cloud Storage Buckets](https://cloud.google.com/storage?hl=en)

:::info

✨ This is an Enterprise only feature [Get Started with Enterprise here](https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat)

:::


| Property | Details |
|----------|---------|
| Description | Log LLM Input/Output to cloud storage buckets |
| Load Test Benchmarks | [Benchmarks](https://docs.litellm.ai/docs/benchmarks) |
| Google Docs on Cloud Storage | [Google Cloud Storage](https://cloud.google.com/storage?hl=en) |



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


### Expected Logs on GCS Buckets

<Image img={require('../../img/gcs_bucket.png')} />

### Fields Logged on GCS Buckets

[**The standard logging object is logged on GCS Bucket**](../proxy/logging)


### Getting `service_account.json` from Google Cloud Console

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Search for IAM & Admin
3. Click on Service Accounts
4. Select a Service Account
5. Click on 'Keys' -> Add Key -> Create New Key -> JSON
6. Save the JSON file and add the path to `GCS_PATH_SERVICE_ACCOUNT`


## s3 Buckets

We will use the `--config` to set 

- `litellm.success_callback = ["s3"]` 

This will log all successfull LLM calls to s3 Bucket

**Step 1** Set AWS Credentials in .env

```shell
AWS_ACCESS_KEY_ID = ""
AWS_SECRET_ACCESS_KEY = ""
AWS_REGION_NAME = ""
```

**Step 2**: Create a `config.yaml` file and set `litellm_settings`: `success_callback`

```yaml
model_list:
 - model_name: gpt-3.5-turbo
    litellm_params:
      model: gpt-3.5-turbo
litellm_settings:
  success_callback: ["s3"]
  s3_callback_params:
    s3_bucket_name: logs-bucket-litellm   # AWS Bucket Name for S3
    s3_region_name: us-west-2              # AWS Region Name for S3
    s3_aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID  # us os.environ/<variable name> to pass environment variables. This is AWS Access Key ID for S3
    s3_aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY  # AWS Secret Access Key for S3
    s3_path: my-test-path # [OPTIONAL] set path in bucket you want to write logs to
    s3_endpoint_url: https://s3.amazonaws.com  # [OPTIONAL] S3 endpoint URL, if you want to use Backblaze/cloudflare s3 buckets
```

**Step 3**: Start the proxy, make a test request

Start proxy

```shell
litellm --config config.yaml --debug
```

Test Request

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
    --header 'Content-Type: application/json' \
    --data ' {
    "model": "Azure OpenAI GPT-4 East",
    "messages": [
        {
        "role": "user",
        "content": "what llm are you"
        }
    ]
    }'
```

Your logs should be available on the specified s3 Bucket
