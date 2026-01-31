import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# AWS Bedrock - Rerank API

Use Bedrock's Rerank API in the Cohere `/rerank` format.

:::info Cost Tracking

✅ **Cost tracking is supported** for Bedrock Rerank API calls.

:::

## Supported Parameters

- `model` - the foundation model ARN
- `query` - the query to rerank against
- `documents` - the list of documents to rerank
- `top_n` - the number of results to return

## Usage

<Tabs>
<TabItem label="SDK" value="sdk">

```python
from litellm import rerank
import os 

os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""

response = rerank(
    model="bedrock/arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0", # provide the model ARN - get this here https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock/client/list_foundation_models.html
    query="hello",
    documents=["hello", "world"],
    top_n=2,
)

print(response)
```

</TabItem>
<TabItem label="PROXY" value="proxy">

### 1. Setup config.yaml

```yaml
model_list:
    - model_name: bedrock-rerank
      litellm_params:
        model: bedrock/arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0
        aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
        aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
        aws_region_name: os.environ/AWS_REGION_NAME
```

### 2. Start proxy server

```bash
litellm --config config.yaml

# RUNNING on http://0.0.0.0:4000
```

### 3. Test it! 

```bash
curl http://0.0.0.0:4000/rerank \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bedrock-rerank",
    "query": "What is the capital of the United States?",
    "documents": [
        "Carson City is the capital city of the American state of Nevada.",
        "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
        "Washington, D.C. is the capital of the United States.",
        "Capital punishment has existed in the United States since before it was a country."
    ],
    "top_n": 3


  }'
```

</TabItem>
</Tabs>

## Authentication

All standard Bedrock authentication methods are supported for rerank. See [Bedrock Authentication](./bedrock#boto3---authentication) for details.

