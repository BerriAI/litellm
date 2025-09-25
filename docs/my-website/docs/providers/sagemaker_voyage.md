# SageMaker Voyage AI Embeddings

Voyage AI embedding models deployed on AWS SageMaker, combining Voyage AI's high-quality embedding models with AWS infrastructure for production deployments.

## Quick Start

### Basic Usage

```python
from litellm import embedding
import os

# Set AWS credentials (or use IAM roles)
os.environ['AWS_ACCESS_KEY_ID'] = 'your-access-key'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'your-secret-key' 
os.environ['AWS_REGION_NAME'] = 'us-east-1'

response = embedding(
    model="sagemaker/voyage/voyage-3",
    input=["Sample text 1", "Sample text 2"],
    input_type="query"  # or "document" for retrieval tasks
)
print(response)
```

### With Custom SageMaker Endpoint

```python
response = embedding(
    model="sagemaker/voyage/voyage-3",
    input=["Sample text 1", "Sample text 2"],
    sagemaker_endpoint_name="my-custom-voyage-endpoint",
    aws_region_name="us-west-2",
    input_type="query"
)
```

## Prerequisites

### 1. AWS Setup

Before using SageMaker Voyage embeddings, you need:

1. **AWS Account** with appropriate permissions
2. **SageMaker Voyage Model Subscription** from AWS Marketplace
3. **Deployed SageMaker Endpoint** with a Voyage AI model

### 2. Model Subscription

Subscribe to Voyage AI models in AWS Marketplace:

1. Visit [AWS Marketplace](https://aws.amazon.com/marketplace)
2. Search for "Voyage AI" 
3. Subscribe to the desired model package (e.g., `voyage-3`, `voyage-code-3`)

### 3. Deploy SageMaker Endpoint

Deploy the subscribed model to a SageMaker endpoint. You can use:

- **AWS Console**: SageMaker → Model packages → Deploy
- **AWS CLI**: Using CloudFormation or CDK
- **Boto3**: Programmatically deploy endpoints

## Supported Models

All Voyage AI models available on AWS Marketplace are supported:

### Text Embedding Models

| Model Name | SageMaker Usage | Description |
|------------|----------------|-------------|
| voyage-3 | `sagemaker/voyage/voyage-3` | Latest general-purpose embedding model |
| voyage-3-lite | `sagemaker/voyage/voyage-3-lite` | Lighter version of voyage-3 |  
| voyage-3-large | `sagemaker/voyage/voyage-3-large` | Larger, higher-capacity version |
| voyage-code-3 | `sagemaker/voyage/voyage-code-3` | Optimized for code similarity |
| voyage-code-2 | `sagemaker/voyage/voyage-code-2` | Previous generation code model |
| voyage-finance-2 | `sagemaker/voyage/voyage-finance-2` | Finance domain-specific |
| voyage-law-2 | `sagemaker/voyage/voyage-law-2` | Legal domain-specific |
| voyage-multilingual-2 | `sagemaker/voyage/voyage-multilingual-2` | Multilingual support |
| voyage-2 | `sagemaker/voyage/voyage-2` | Previous generation general model |
| voyage-large-2 | `sagemaker/voyage/voyage-large-2` | Previous generation large model |
| voyage-large-2-instruct | `sagemaker/voyage/voyage-large-2-instruct` | Instruction-tuned variant |

### Multimodal Embedding Models

| Model Name | SageMaker Usage | Description |
|------------|----------------|-------------|
| voyage-multimodal-3 | `sagemaker/voyage/voyage-multimodal-3` | Text and image embeddings |

### Reranking Models

| Model Name | SageMaker Usage | Description |
|------------|----------------|-------------|
| rerank-2 | `sagemaker/voyage/rerank-2` | Latest reranking model |
| rerank-2-lite | `sagemaker/voyage/rerank-2-lite` | Lighter reranking model |
| rerank-lite-1 | `sagemaker/voyage/rerank-lite-1` | Previous generation reranker |

## Configuration Options

### Authentication

SageMaker Voyage supports all AWS authentication methods:

```python
# Method 1: Environment variables
os.environ['AWS_ACCESS_KEY_ID'] = 'your-key'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'your-secret'
os.environ['AWS_REGION_NAME'] = 'us-east-1'

# Method 2: Explicit parameters  
response = embedding(
    model="sagemaker/voyage/voyage-3",
    input=["text"],
    aws_access_key_id="your-key",
    aws_secret_access_key="your-secret", 
    aws_region_name="us-east-1"
)

# Method 3: IAM roles (recommended for production)
# No credentials needed - uses instance/container IAM role
response = embedding(
    model="sagemaker/voyage/voyage-3",
    input=["text"],
    aws_region_name="us-east-1"
)

# Method 4: AWS profiles
response = embedding(
    model="sagemaker/voyage/voyage-3", 
    input=["text"],
    aws_profile_name="my-profile"
)
```

### Voyage-Specific Parameters

All Voyage AI parameters are supported:

```python
response = embedding(
    model="sagemaker/voyage/voyage-3",
    input=["query text", "document text"], 
    input_type="query",  # "query" or "document" 
    truncation=True,     # Enable truncation
    encoding_format="float",  # "float" or "base64"
    dimensions=1024,     # Output dimensions (if supported)
)
```

### SageMaker-Specific Parameters

```python
response = embedding(
    model="sagemaker/voyage/voyage-3",
    input=["text"],
    # SageMaker specific
    sagemaker_endpoint_name="my-custom-endpoint",  # Override endpoint name
    aws_region_name="us-west-2",
    # AWS session parameters  
    aws_session_name="my-session",
    aws_role_name="my-role",
    aws_external_id="external-id"
)
```

## Usage Patterns

### Retrieval-Augmented Generation (RAG)

```python
# Embed documents for storage
documents = [
    "Python is a programming language",
    "Machine learning uses algorithms",
    "AWS provides cloud services"
]

doc_embeddings = embedding(
    model="sagemaker/voyage/voyage-3",
    input=documents,
    input_type="document"  # Important for documents
)

# Embed query for retrieval
query = "What is Python?"
query_embedding = embedding(
    model="sagemaker/voyage/voyage-3", 
    input=[query],
    input_type="query"  # Important for queries
)
```

### Code Similarity

```python
code_snippets = [
    "def hello_world(): print('Hello, World!')",
    "function helloWorld() { console.log('Hello, World!'); }",
    "class MyClass: pass"
]

code_embeddings = embedding(
    model="sagemaker/voyage/voyage-code-3",  # Code-specific model
    input=code_snippets,
    input_type="document"
)
```

### Multilingual Text

```python
multilingual_text = [
    "Hello world",           # English
    "Hola mundo",           # Spanish  
    "Bonjour le monde",     # French
    "こんにちは世界"         # Japanese
]

embeddings = embedding(
    model="sagemaker/voyage/voyage-multilingual-2",
    input=multilingual_text,
    input_type="query"
)
```

## Advanced Configuration

### Custom Endpoint Names

By default, the endpoint name is derived from the model name. You can override this:

```python
response = embedding(
    model="sagemaker/voyage/voyage-3",
    input=["text"],
    sagemaker_endpoint_name="voyage-3-production-v2"
)
```

### Cross-Region Deployment

```python
response = embedding(
    model="sagemaker/voyage/voyage-3",
    input=["text"], 
    aws_region_name="eu-west-1",  # Use European deployment
    sagemaker_endpoint_name="voyage-3-eu"
)
```

### Error Handling

```python
from litellm.llms.voyage.embedding.sagemaker_transformation import SageMakerVoyageError

try:
    response = embedding(
        model="sagemaker/voyage/voyage-3",
        input=["text"],
    )
except SageMakerVoyageError as e:
    print(f"SageMaker Voyage Error: {e.status_code} - {e.message}")
except Exception as e:
    print(f"General error: {e}")
```

## Cost Optimization

### Instance Types

Choose appropriate SageMaker instance types for your workload:

```yaml
# In your deployment configuration
InstanceType: ml.g5.xlarge    # GPU instances for better performance
# or
InstanceType: ml.c5.2xlarge   # CPU instances for cost optimization
```

### Batch Processing

Process multiple texts together for efficiency:

```python
# Efficient: Process in batches
large_batch = ["text1", "text2", ..., "text100"] 
embeddings = embedding(
    model="sagemaker/voyage/voyage-3",
    input=large_batch  # Process all at once
)

# Less efficient: Individual calls
# for text in large_batch:
#     embedding(model="sagemaker/voyage/voyage-3", input=[text])
```

## Monitoring and Logging

Enable detailed logging for debugging:

```python
import litellm
litellm.set_verbose = True

response = embedding(
    model="sagemaker/voyage/voyage-3",
    input=["text"],
)
# Will output detailed request/response information
```

## LiteLLM Proxy Usage

Configure SageMaker Voyage in your proxy config:

```yaml
model_list:
  - model_name: voyage-embeddings
    litellm_params:
      model: sagemaker/voyage/voyage-3
      aws_region_name: us-east-1
      input_type: query
  - model_name: voyage-code-embeddings  
    litellm_params:
      model: sagemaker/voyage/voyage-code-3
      aws_region_name: us-east-1
      sagemaker_endpoint_name: voyage-code-production
```

Then use via the proxy:

```bash
curl -X POST "http://localhost:4000/v1/embeddings" \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voyage-embeddings",
    "input": ["Hello world", "Test text"]
  }'
```

## Comparison with Direct Voyage AI

| Feature | Direct Voyage AI | SageMaker Voyage |
|---------|------------------|------------------|
| Authentication | API Key | AWS Credentials |
| Scaling | Managed by Voyage | Custom SageMaker scaling |
| Cost | Pay-per-use | Instance-based pricing |
| Latency | Varies by region | Consistent (your VPC) |
| Customization | Limited | Full SageMaker control |
| Data Privacy | Voyage's infrastructure | Your AWS account |

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   ```
   Error: Access denied
   ```
   - Verify AWS credentials have SageMaker permissions
   - Check IAM role policies include `sagemaker:InvokeEndpoint`

2. **Endpoint Not Found**
   ```
   Error: Could not find endpoint
   ```
   - Verify endpoint name is correct
   - Check endpoint is deployed and in service
   - Confirm region matches endpoint deployment

3. **Model Not Supported**
   ```
   Error: Model not supported
   ```
   - Verify model is subscribed in AWS Marketplace
   - Check model package is deployed to SageMaker

### Required IAM Permissions

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow", 
            "Action": [
                "sagemaker:InvokeEndpoint"
            ],
            "Resource": "arn:aws:sagemaker:*:*:endpoint/voyage-*"
        }
    ]
}
```

## Support

For issues specific to:
- **LiteLLM Integration**: Create an issue on [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- **SageMaker Deployment**: Consult [AWS SageMaker documentation](https://docs.aws.amazon.com/sagemaker/)
- **Voyage AI Models**: Visit [Voyage AI documentation](https://docs.voyageai.com/)