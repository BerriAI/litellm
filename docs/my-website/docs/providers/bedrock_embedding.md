# Bedrock Embedding

## Supported Embedding Models

| Provider | LiteLLM Route | AWS Documentation |
|----------|---------------|-------------------|
| Amazon Titan | `bedrock/amazon.*` | [Amazon Titan Embeddings](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html) |
| Cohere | `bedrock/cohere.*` | [Cohere Embeddings](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-cohere-embed.html) |
| TwelveLabs | `bedrock/us.twelvelabs.*` | [TwelveLabs](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-twelvelabs.html) |

## Async Invoke Support

LiteLLM supports AWS Bedrock's async-invoke feature for embedding models that require asynchronous processing, particularly useful for large media files (video, audio) or when you need to process embeddings in the background.

### Supported Models

| Provider | Async Invoke Route | Use Case |
|----------|-------------------|----------|
| TwelveLabs Marengo | `bedrock/async_invoke/us.twelvelabs.marengo-embed-2-7-v1:0` | Video, audio, image, and text embeddings |

### Required Parameters

When using async-invoke, you must provide:

| Parameter | Description | Required |
|-----------|-------------|----------|
| `output_s3_uri` | S3 URI where the embedding results will be stored | ✅ Yes |
| `input_type` | Type of input: `"text"`, `"image"`, `"video"`, or `"audio"` | ✅ Yes |
| `aws_region_name` | AWS region for the request | ✅ Yes |

### Usage

#### Basic Async Invoke

```python
from litellm import embedding

# Text embedding with async-invoke
response = embedding(
    model="bedrock/async_invoke/us.twelvelabs.marengo-embed-2-7-v1:0",
    input=["Hello world from LiteLLM async invoke!"],
    aws_region_name="us-east-1",
    input_type="text",
    output_s3_uri="s3://your-bucket/async-invoke-output/"
)

print(f"Job submitted! Invocation ARN: {response._hidden_params._invocation_arn}")
```

#### Video/Audio Embedding

```python
# Video embedding (requires async-invoke)
response = embedding(
    model="bedrock/async_invoke/us.twelvelabs.marengo-embed-2-7-v1:0",
    input=["s3://your-bucket/video.mp4"],  # S3 URL for video
    aws_region_name="us-east-1",
    input_type="video",
    output_s3_uri="s3://your-bucket/async-invoke-output/"
)

print(f"Video embedding job submitted! ARN: {response._hidden_params._invocation_arn}")
```

#### Image Embedding with Base64

```python
import base64

# Load and encode image
with open("image.jpg", "rb") as img_file:
    img_data = base64.b64encode(img_file.read()).decode('utf-8')
    img_base64 = f"data:image/jpeg;base64,{img_data}"

response = embedding(
    model="bedrock/async_invoke/us.twelvelabs.marengo-embed-2-7-v1:0",
    input=[img_base64],
    aws_region_name="us-east-1",
    input_type="image",
    output_s3_uri="s3://your-bucket/async-invoke-output/"
)
```

### Retrieving Job Information

#### Getting Job ID and Invocation ARN

The async-invoke response includes the invocation ARN in the hidden parameters:

```python
response = embedding(
    model="bedrock/async_invoke/us.twelvelabs.marengo-embed-2-7-v1:0",
    input=["Hello world"],
    aws_region_name="us-east-1",
    input_type="text",
    output_s3_uri="s3://your-bucket/async-invoke-output/"
)

# Access invocation ARN
invocation_arn = response._hidden_params._invocation_arn
print(f"Invocation ARN: {invocation_arn}")

# Extract job ID from ARN (last part after the last slash)
job_id = invocation_arn.split("/")[-1]
print(f"Job ID: {job_id}")
```

#### Checking Job Status

Use LiteLLM's `retrieve_batch` function to check if your job is still processing:

```python
from litellm import retrieve_batch

def check_async_job_status(invocation_arn, aws_region_name="us-east-1"):
    """Check the status of an async invoke job using LiteLLM batch API"""
    try:
        response = retrieve_batch(
            batch_id=invocation_arn,
            custom_llm_provider="bedrock",
            aws_region_name=aws_region_name
        )
        return response
    except Exception as e:
        print(f"Error checking job status: {e}")
        return None

# Check status
status = check_async_job_status(invocation_arn, "us-east-1")
if status:
    print(f"Job Status: {status.status}")
    print(f"Output Location: {status.output_file_id}")
```

**Note:** The actual embedding results are stored in S3. The `output_file_id` from the batch status can be used to locate the results file in your S3 bucket.

### Error Handling

#### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `ValueError: output_s3_uri cannot be empty` | Missing S3 output URI | Provide a valid S3 URI |
| `ValueError: Input type 'video' requires async_invoke route` | Using video/audio without async-invoke | Use `bedrock/async_invoke/` model prefix |
| `ValueError: input_type is required` | Missing input type parameter | Specify `input_type` parameter |

#### Example Error Handling

```python
try:
    response = embedding(
        model="bedrock/async_invoke/us.twelvelabs.marengo-embed-2-7-v1:0",
        input=["Hello world"],
        aws_region_name="us-east-1",
        input_type="text",
        output_s3_uri="s3://your-bucket/output/"  # Required for async-invoke
    )
    print("Job submitted successfully!")
    
except ValueError as e:
    if "output_s3_uri cannot be empty" in str(e):
        print("Error: Please provide a valid S3 output URI")
    elif "requires async_invoke route" in str(e):
        print("Error: Use async_invoke model for video/audio inputs")
    else:
        print(f"Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Best Practices

1. **Use async-invoke for large files**: Video and audio files are better processed asynchronously
2. **Use LiteLLM batch API**: Use `retrieve_batch()` instead of direct Bedrock API calls for status checking
3. **Monitor job status**: Check job status periodically using the batch API to know when results are ready
4. **Handle errors gracefully**: Implement proper error handling for network issues and job failures
5. **Set appropriate timeouts**: Consider the processing time for large files
6. **Use S3 for large inputs**: For video/audio, use S3 URLs instead of base64 encoding

### Limitations

- Async-invoke is currently only supported for TwelveLabs Marengo models
- Results are stored in S3 and must be retrieved separately using the output file ID
- Job status checking requires using LiteLLM's `retrieve_batch()` function
- No built-in polling mechanism in LiteLLM (must implement your own status checking loop)

### API keys
This can be set as env variables or passed as **params to litellm.embedding()**
```python
import os
os.environ["AWS_ACCESS_KEY_ID"] = ""        # Access key
os.environ["AWS_SECRET_ACCESS_KEY"] = ""    # Secret access key
os.environ["AWS_REGION_NAME"] = ""           # us-east-1, us-east-2, us-west-1, us-west-2
```

## Usage
### LiteLLM Python SDK
```python
from litellm import embedding
response = embedding(
    model="bedrock/amazon.titan-embed-text-v1",
    input=["good morning from litellm"],
)
print(response)
```

### LiteLLM Proxy Server

#### 1. Setup config.yaml
```yaml
model_list:
  - model_name: titan-embed-v1
    litellm_params:
      model: bedrock/amazon.titan-embed-text-v1
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: us-east-1
  - model_name: titan-embed-v2
    litellm_params:
      model: bedrock/amazon.titan-embed-text-v2:0
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: us-east-1
```

#### 2. Start Proxy 
```bash
litellm --config /path/to/config.yaml
```

#### 3. Use with OpenAI Python SDK
```python
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

response = client.embeddings.create(
    input=["good morning from litellm"],
    model="titan-embed-v1"
)
print(response)
```

#### 4. Use with LiteLLM Python SDK
```python
import litellm
response = litellm.embedding(
    model="titan-embed-v1", # model alias from config.yaml
    input=["good morning from litellm"],
    api_base="http://0.0.0.0:4000",
    api_key="anything"
)
print(response)
```

## Supported AWS Bedrock Embedding Models

| Model Name           | Usage                               | Supported Additional OpenAI params |
|----------------------|---------------------------------------------|-----|
| Titan Embeddings V2 | `embedding(model="bedrock/amazon.titan-embed-text-v2:0", input=input)` | [here](https://github.com/BerriAI/litellm/blob/f5905e100068e7a4d61441d7453d7cf5609c2121/litellm/llms/bedrock/embed/amazon_titan_v2_transformation.py#L59) |
| Titan Embeddings - V1 | `embedding(model="bedrock/amazon.titan-embed-text-v1", input=input)` | [here](https://github.com/BerriAI/litellm/blob/f5905e100068e7a4d61441d7453d7cf5609c2121/litellm/llms/bedrock/embed/amazon_titan_g1_transformation.py#L53)
| Titan Multimodal Embeddings | `embedding(model="bedrock/amazon.titan-embed-image-v1", input=input)` | [here](https://github.com/BerriAI/litellm/blob/f5905e100068e7a4d61441d7453d7cf5609c2121/litellm/llms/bedrock/embed/amazon_titan_multimodal_transformation.py#L28) |
| TwelveLabs Marengo Embed 2.7 | `embedding(model="bedrock/us.twelvelabs.marengo-embed-2-7-v1:0", input=input)` | Supports multimodal input (text, video, audio, image) |
| Cohere Embeddings - English | `embedding(model="bedrock/cohere.embed-english-v3", input=input)` | [here](https://github.com/BerriAI/litellm/blob/f5905e100068e7a4d61441d7453d7cf5609c2121/litellm/llms/bedrock/embed/cohere_transformation.py#L18)
| Cohere Embeddings - Multilingual | `embedding(model="bedrock/cohere.embed-multilingual-v3", input=input)` | [here](https://github.com/BerriAI/litellm/blob/f5905e100068e7a4d61441d7453d7cf5609c2121/litellm/llms/bedrock/embed/cohere_transformation.py#L18)

### Advanced - [Drop Unsupported Params](https://docs.litellm.ai/docs/completion/drop_params#openai-proxy-usage)

### Advanced - [Pass model/provider-specific Params](https://docs.litellm.ai/docs/completion/provider_specific_params#proxy-usage)