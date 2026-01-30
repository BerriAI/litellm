import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Bedrock Batches

Use Amazon Bedrock Batch Inference API through LiteLLM.

| Property | Details |
|----------|---------|
| Description | Amazon Bedrock Batch Inference allows you to run inference on large datasets asynchronously |
| Provider Doc | [AWS Bedrock Batch Inference ↗](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference.html) |
| Cost Tracking | ✅ Supported |

## Overview

Use this to:

- Run batch inference on large datasets with Bedrock models
- Control batch model access by key/user/team (same as chat completion models)
- Manage S3 storage for batch input/output files

## (Proxy Admin) Usage

Here's how to give developers access to your Bedrock Batch models.

### 1. Setup config.yaml

- Specify `mode: batch` for each model: Allows developers to know this is a batch model
- Configure S3 bucket and AWS credentials for batch operations

```yaml showLineNumbers title="litellm_config.yaml"
model_list:
  - model_name: "bedrock-batch-claude"
    litellm_params:
      model: bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0
      #########################################################
      ########## batch specific params ########################
      s3_bucket_name: litellm-proxy
      s3_region_name: us-west-2
      s3_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      s3_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_batch_role_arn: arn:aws:iam::888602223428:role/service-role/AmazonBedrockExecutionRoleForAgents_BB9HNW6V4CV
      # Optional: Custom KMS encryption key for S3 output
      # s3_encryption_key_id: arn:aws:kms:us-west-2:123456789012:key/12345678-1234-1234-1234-123456789012
    model_info: 
      mode: batch # 👈 SPECIFY MODE AS BATCH, to tell user this is a batch model
```

**Required Parameters:**

| Parameter | Description |
|-----------|-------------|
| `s3_bucket_name` | S3 bucket for batch input/output files |
| `s3_region_name` | AWS region for S3 bucket |
| `s3_access_key_id` | AWS access key for S3 bucket |
| `s3_secret_access_key` | AWS secret key for S3 bucket |
| `aws_batch_role_arn` | IAM role ARN for Bedrock batch operations. Bedrock Batch APIs require an IAM role ARN to be set. |
| `mode: batch` | Indicates to LiteLLM this is a batch model |

**Optional Parameters:**

| Parameter | Description |
|-----------|-------------|
| `s3_encryption_key_id` | Custom KMS encryption key ID for S3 output data. If not specified, Bedrock uses AWS managed encryption keys. |

### 2. Create Virtual Key

```bash showLineNumbers title="create_virtual_key.sh"
curl -L -X POST 'https://{PROXY_BASE_URL}/key/generate' \
-H 'Authorization: Bearer ${PROXY_API_KEY}' \
-H 'Content-Type: application/json' \
-d '{"models": ["bedrock-batch-claude"]}'
```

You can now use the virtual key to access the batch models (See Developer flow).

## (Developer) Usage

Here's how to create a LiteLLM managed file and execute Bedrock Batch CRUD operations with the file.

### 1. Create request.jsonl

- Check models available via `/model_group/info`
- See all models with `mode: batch`
- Set `model` in .jsonl to the model from `/model_group/info`

```json showLineNumbers title="bedrock_batch_completions.jsonl"
{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "bedrock-batch-claude", "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello world!"}], "max_tokens": 1000}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "bedrock-batch-claude", "messages": [{"role": "system", "content": "You are an unhelpful assistant."}, {"role": "user", "content": "Hello world!"}], "max_tokens": 1000}}
```

Expectation:

- LiteLLM translates this to the bedrock deployment specific value (e.g. `bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0`)

### 2. Upload File

Specify `target_model_names: "<model-name>"` to enable LiteLLM managed files and request validation.

model-name should be the same as the model-name in the request.jsonl

<Tabs>
<TabItem value="python" label="Python">

```python showLineNumbers title="bedrock_batch.py"
from openai import OpenAI

client = OpenAI(
    base_url="http://0.0.0.0:4000",
    api_key="sk-1234",
)

# Upload file
batch_input_file = client.files.create(
    file=open("./bedrock_batch_completions.jsonl", "rb"), # {"model": "bedrock-batch-claude"} <-> {"model": "bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0"}
    purpose="batch",
    extra_body={"target_model_names": "bedrock-batch-claude"}
)
print(batch_input_file)
```

</TabItem>
<TabItem value="curl" label="Curl">

```bash showLineNumbers title="Upload File"
curl http://localhost:4000/v1/files \
    -H "Authorization: Bearer sk-1234" \
    -F purpose="batch" \
    -F file="@bedrock_batch_completions.jsonl" \
    -F extra_body='{"target_model_names": "bedrock-batch-claude"}'
```

</TabItem>
</Tabs>

**Where is the file written?**:

The file is written to S3 bucket specified in your config and prepared for Bedrock batch inference.

### 3. Create the batch

<Tabs>
<TabItem value="python" label="Python">

```python showLineNumbers title="bedrock_batch.py"
...
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

</TabItem>
</Tabs>

### 4. Retrieve batch results

Once the batch job is completed, download the results from S3:

<Tabs>
<TabItem value="python" label="Python">

```python showLineNumbers title="bedrock_batch.py"
...
# Wait for batch completion (check status periodically)
batch_status = client.batches.retrieve(batch_id=batch.id)

if batch_status.status == "completed":
    # Download the output file
    result = client.files.content(
        file_id=batch_status.output_file_id,
        extra_headers={"custom-llm-provider": "bedrock"}
    )
    
    # Save or process the results
    with open("batch_output.jsonl", "wb") as f:
        f.write(result.content)
    
    # Parse JSONL results
    for line in result.text.strip().split('\n'):
        record = json.loads(line)
        print(f"Record ID: {record['recordId']}")
        print(f"Output: {record.get('modelOutput', {})}")
```

</TabItem>
<TabItem value="curl" label="Curl">

```bash showLineNumbers title="Download Batch Results"
# First retrieve batch to get output_file_id
curl http://localhost:4000/v1/batches/batch_abc123 \
    -H "Authorization: Bearer sk-1234"

# Then download the output file
curl http://localhost:4000/v1/files/{output_file_id}/content \
    -H "Authorization: Bearer sk-1234" \
    -H "custom-llm-provider: bedrock" \
    -o batch_output.jsonl
```

</TabItem>
<TabItem value="litellm-direct" label="LiteLLM Direct">

```python showLineNumbers title="bedrock_batch.py"
import litellm
from litellm import file_content

# Download using litellm directly (bypasses proxy managed files)
result = file_content(
    file_id=batch_status.output_file_id,  # Can be S3 URI or unified file ID
    custom_llm_provider="bedrock",
    aws_region_name="us-west-2",
)

# Process results
print(result.text)
```

</TabItem>
</Tabs>

**Output Format:**

The batch output file is in JSONL format with each line containing:

```json
{
  "recordId": "request-1",
  "modelInput": {
    "messages": [...],
    "max_tokens": 1000
  },
  "modelOutput": {
    "content": [...],
    "id": "msg_abc123",
    "model": "claude-3-5-sonnet-20240620-v1:0",
    "role": "assistant",
    "stop_reason": "end_turn",
    "usage": {
      "input_tokens": 15,
      "output_tokens": 10
    }
  }
}
```

## FAQ

### Where are my files written?

When a `target_model_names` is specified, the file is written to the S3 bucket configured in your Bedrock batch model configuration.

### What models are supported?

LiteLLM only supports Bedrock Anthropic Models for Batch API. If you want other bedrock models file an issue [here](https://github.com/BerriAI/litellm/issues/new/choose).

### How do I use a custom KMS encryption key?

If your S3 bucket requires a custom KMS encryption key, you can specify it in your configuration using `s3_encryption_key_id`. This is useful for enterprise customers with specific encryption requirements.

You can set the encryption key in 2 ways:

1. **In config.yaml** (recommended):
```yaml
model_list:
  - model_name: "bedrock-batch-claude"
    litellm_params:
      model: bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0
      s3_encryption_key_id: arn:aws:kms:us-west-2:123456789012:key/12345678-1234-1234-1234-123456789012
      # ... other params
```

2. **As an environment variable**:
```bash
export AWS_S3_ENCRYPTION_KEY_ID=arn:aws:kms:us-west-2:123456789012:key/12345678-1234-1234-1234-123456789012
```



## Further Reading

- [AWS Bedrock Batch Inference Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference.html)
- [LiteLLM Managed Batches](../proxy/managed_batches)
- [LiteLLM Authentication to Bedrock](https://docs.litellm.ai/docs/providers/bedrock#boto3---authentication)
