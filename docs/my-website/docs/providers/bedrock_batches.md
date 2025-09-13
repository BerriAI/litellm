import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Bedrock Batches

Use Amazon Bedrock Batch Inference API through LiteLLM.

| Property | Details |
|----------|---------|
| Description | Amazon Bedrock Batch Inference allows you to run inference on large datasets asynchronously |
| Provider Doc | [AWS Bedrock Batch Inference ↗](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference.html) |

## Quick Start

#### 1. Configure your model in config.yaml

<Tabs>
<TabItem value="config-yaml" label="config.yaml">

```yaml showLineNumbers title="LiteLLM Proxy Configuration"
model_list:
  - model_name: bedrock/batch-anthropic.claude-3-5-sonnet-20240620-v1:0
    litellm_params:
      model: bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0
      #########################################################
      ########## batch specific params ########################
      s3_bucket_name: litellm-proxy
      s3_region_name: us-west-2
      s3_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      s3_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_batch_role_arn: arn:aws:iam::888602223428:role/service-role/AmazonBedrockExecutionRoleForAgents_BB9HNW6V4CV
    model_info: 
      mode: batch
```

</TabItem>
</Tabs>

**Required Parameters:**
- `s3_bucket_name`: S3 bucket for input/output files
- `s3_region_name`: AWS region for S3 bucket
- `s3_access_key_id`: AWS access key
- `s3_secret_access_key`: AWS secret key
- `aws_batch_role_arn`: IAM role ARN for Bedrock batch operations
- `mode: batch`: Indicates this is a batch model

#### 2. Start the LiteLLM Proxy

```bash showLineNumbers title="Start LiteLLM Proxy"
litellm --config config.yaml
```

#### 3. Create and manage batch requests

<Tabs>
<TabItem value="python" label="Python">

```python showLineNumbers title="Complete Bedrock Batch Example"
from openai import OpenAI

client = OpenAI(
    base_url="http://0.0.0.0:4000",
    api_key="sk-1234",
)

BEDROCK_BATCH_MODEL = "bedrock/batch-anthropic.claude-3-5-sonnet-20240620-v1:0"

# Upload file
batch_input_file = client.files.create(
    file=open("./bedrock_batch_completions.jsonl", "rb"),
    purpose="batch",
    extra_body={"target_model_names": BEDROCK_BATCH_MODEL}
)
print(batch_input_file)

# Create batch
batch = client.batches.create( 
    input_file_id=batch_input_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={"description": "Test batch job"},
)
print(batch)
```

</TabItem>

<TabItem value="curl" label="Curl">

```bash showLineNumbers title="Upload File"
curl http://localhost:4000/v1/files \
    -H "Authorization: Bearer sk-1234" \
    -F purpose="batch" \
    -F file="@bedrock_batch_completions.jsonl" \
    -F extra_body='{"target_model_names": "bedrock/batch-anthropic.claude-3-5-sonnet-20240620-v1:0"}'
```

```bash showLineNumbers title="Create Batch Request"
curl http://localhost:4000/v1/batches \
    -H "Authorization: Bearer sk-1234" \
    -H "Content-Type: application/json" \
    -d '{
        "input_file_id": "file-abc123",
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "metadata": {"description": "Test batch job"}
    }'
```

```bash showLineNumbers title="Retrieve Batch Status"
curl http://localhost:4000/v1/batches/batch_abc123 \
    -H "Authorization: Bearer sk-1234" \
    -H "Content-Type: application/json"
```

```bash showLineNumbers title="List Batches"
curl http://localhost:4000/v1/batches \
    -H "Authorization: Bearer sk-1234" \
    -H "Content-Type: application/json"
```

</TabItem>
</Tabs>

## Input File Format

Create a JSONL file with your batch requests:

```json showLineNumbers title="bedrock_batch_completions.jsonl"
{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "bedrock/batch-anthropic.claude-3-5-sonnet-20240620-v1:0", "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello world!"}], "max_tokens": 1000}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "bedrock/batch-anthropic.claude-3-5-sonnet-20240620-v1:0", "messages": [{"role": "system", "content": "You are an unhelpful assistant."}, {"role": "user", "content": "Hello world!"}], "max_tokens": 1000}}
```

## Batch Workflow

1. **Upload Input File**: Upload your JSONL file containing batch requests
2. **Create Batch Job**: Submit the batch job with the input file ID
3. **Monitor Status**: Poll the batch status until completion
4. **Retrieve Results**: Download the output file containing responses

### Batch Status Values

- `validating`: Input file is being validated
- `in_progress`: Batch is being processed
- `finalizing`: Batch processing is completing
- `completed`: Batch completed successfully
- `failed`: Batch failed
- `expired`: Batch expired
- `cancelled`: Batch was cancelled

## LiteLLM Managed Files

When using `target_model_names` in the file upload, LiteLLM provides additional features:

- **Load Balancing**: Automatically distributes requests across multiple Bedrock deployments
- **Request Validation**: Validates batch requests before processing
- **Model Translation**: Translates virtual model names to actual deployment names

See [LiteLLM Managed Batches](../proxy/managed_batches) for more details.

## Authentication

Bedrock batches require AWS credentials with appropriate permissions:

```yaml showLineNumbers title="Required AWS Permissions"
# IAM Policy for Bedrock Batch Operations
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:CreateModelInvocationJob",
                "bedrock:GetModelInvocationJob",
                "bedrock:ListModelInvocationJobs",
                "bedrock:StopModelInvocationJob"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject"
            ],
            "Resource": "arn:aws:s3:::your-bucket-name/*"
        }
    ]
}
```

## Supported Models

Bedrock batch inference supports various models:

- **Anthropic Claude**: `anthropic.claude-3-5-sonnet-20240620-v1:0`, `anthropic.claude-3-haiku-20240307-v1:0`
- **Amazon Titan**: `amazon.titan-text-express-v1`, `amazon.titan-text-lite-v1`
- **Meta Llama**: `meta.llama3-8b-instruct-v1:0`, `meta.llama3-70b-instruct-v1:0`

## Cost Tracking

LiteLLM automatically tracks costs for Bedrock batch operations:

- Initial batch creation is logged as `acreate_batch`
- Final costs are calculated and logged as `batch_success` upon completion
- Costs include both input and output token usage across all batch responses

## Further Reading

- [AWS Bedrock Batch Inference Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference.html)
- [LiteLLM Managed Batches](../proxy/managed_batches)
- [LiteLLM Authentication to Bedrock](https://docs.litellm.ai/docs/providers/bedrock#boto3---authentication)
