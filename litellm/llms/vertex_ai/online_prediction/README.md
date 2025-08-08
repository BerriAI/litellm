# Vertex AI Online Prediction

This module provides support for Vertex AI Online Prediction endpoints, enabling real-time inference on custom models deployed on Google Cloud.

## Features

- **Real-time Inference**: Low-latency predictions on deployed models
- **Custom Model Support**: Support for custom models deployed on Vertex AI
- **Batch and Single Predictions**: Support for both single and batch prediction requests
- **Authentication**: Support for various Google Cloud authentication methods
- **Error Handling**: Comprehensive error handling and retry logic
- **Logging**: Detailed logging for monitoring and debugging

## Architecture

```
Online Prediction Flow:
1. Client Request → LiteLLM Router
2. Router → Vertex AI Online Prediction Handler
3. Handler → Vertex AI Prediction Service
4. Response → Client
```

## API Endpoints Supported

- `POST /v1/projects/{project}/locations/{location}/endpoints/{endpoint}:predict`
- `POST /v1/projects/{project}/locations/{location}/endpoints/{endpoint}:rawPredict`

## Usage Examples

### Basic Online Prediction
```python
from litellm import completion

response = completion(
    model="vertex_ai/endpoints/1234567890123456789",  # Custom endpoint
    messages=[{"role": "user", "content": "Hello, world!"}],
    custom_llm_provider="vertex_ai",
    vertex_project="my-project",
    vertex_location="us-central1"
)
```

### Batch Online Prediction
```python
from litellm import batch_completion

responses = batch_completion(
    model="vertex_ai/endpoints/1234567890123456789",
    messages=[
        [{"role": "user", "content": "Hello, world!"}],
        [{"role": "user", "content": "How are you?"}]
    ],
    custom_llm_provider="vertex_ai"
)
```

## Configuration

### Environment Variables
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to service account key file
- `VERTEX_AI_PROJECT`: Default Google Cloud project ID
- `VERTEX_AI_LOCATION`: Default Vertex AI location

### Model Configuration
```yaml
model_list:
  - model_name: "my-custom-model"
    litellm_params:
      model: "vertex_ai/endpoints/1234567890123456789"
      vertex_project: "my-project"
      vertex_location: "us-central1"
      custom_llm_provider: "vertex_ai"
```

## Implementation Status

- [x] Basic structure and types
- [ ] Handler implementation
- [ ] Transformation logic
- [ ] Authentication integration
- [ ] Error handling
- [ ] Testing
- [ ] Documentation 