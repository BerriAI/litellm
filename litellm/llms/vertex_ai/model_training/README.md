# Vertex AI Supervised Fine-Tuning

This module provides support for Vertex AI Supervised Fine-Tuning, enabling users to fine-tune pre-trained models on custom datasets for specific tasks.

## Features

- **Supervised Fine-Tuning**: Fine-tune pre-trained models on custom datasets
- **Hyperparameter Optimization**: Automated hyperparameter tuning for optimal performance
- **Dataset Management**: Support for various dataset formats and validation
- **Model Evaluation**: Built-in evaluation metrics and validation
- **Cost Tracking**: Real-time cost monitoring and optimization
- **Model Deployment**: Automatic deployment of fine-tuned models to endpoints

## Architecture

```
Supervised Fine-Tuning Flow:
1. Base Model + Training Data → Vertex AI Fine-Tuning Job
2. Fine-Tuning Job → Fine-Tuned Model
3. Fine-Tuned Model → Model Registry
4. Model Registry → Endpoint Deployment
5. Endpoint → Online Prediction
```

## Supported Base Models

- **Gemini Models**: gemini-1.0-pro, gemini-2.0-flash, gemini-2.5-pro
- **Claude Models**: claude-3-opus, claude-3-sonnet, claude-3-haiku
- **Llama Models**: meta-llama/Llama-2-7b-chat, meta-llama/Llama-2-13b-chat
- **Mistral Models**: mistral-7b-instruct, mistral-large
- **Custom Models**: Any model available in Vertex AI Model Garden

## Usage Examples

### Basic Supervised Fine-Tuning
```python
from litellm import create_fine_tuning_job

job = create_fine_tuning_job(
    model="gemini-1.0-pro",  # Base model to fine-tune
    training_file="gs://my-bucket/training-data.jsonl",
    validation_file="gs://my-bucket/validation-data.jsonl",
    hyperparameters={
        "epoch_count": 3,
        "learning_rate_multiplier": 1.0,
        "adapter_size": "medium"
    },
    suffix="my-custom-model",  # Suffix for the fine-tuned model
    vertex_project="my-project",
    vertex_location="us-central1"
)
```

### Fine-Tuning with Custom Hyperparameters
```python
from litellm import create_fine_tuning_job

job = create_fine_tuning_job(
    model="claude-3-sonnet",
    training_file="gs://my-bucket/training-data.jsonl",
    hyperparameters={
        "epoch_count": 5,
        "learning_rate_multiplier": 0.5,
        "adapter_size": "large",
        "batch_size": 16
    },
    suffix="customer-support-model",
    vertex_project="my-project",
    vertex_location="us-central1"
)
```

### Monitor Fine-Tuning Progress
```python
from litellm import get_fine_tuning_job

job_status = get_fine_tuning_job(
    job_id="1234567890123456789",
    vertex_project="my-project",
    vertex_location="us-central1"
)

print(f"Status: {job_status.status}")
print(f"Progress: {job_status.progress}")
print(f"Fine-tuned model: {job_status.fine_tuned_model}")
```

## Dataset Format

### JSONL Format (Recommended)
```jsonl
{"messages": [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "The capital of France is Paris."}]}
{"messages": [{"role": "user", "content": "How do I make coffee?"}, {"role": "assistant", "content": "To make coffee, you need coffee grounds, hot water, and a brewing method..."}]}
```

### CSV Format
```csv
prompt,completion
"What is the capital of France?","The capital of France is Paris."
"How do I make coffee?","To make coffee, you need coffee grounds, hot water, and a brewing method..."
```

## Hyperparameters

### Core Hyperparameters
- **`epoch_count`**: Number of training epochs (1-10, default: 3)
- **`learning_rate_multiplier`**: Learning rate multiplier (0.1-10.0, default: 1.0)
- **`adapter_size`**: Size of the adapter ("small", "medium", "large", default: "medium")

### Advanced Hyperparameters
- **`batch_size`**: Training batch size (1-64, default: model-specific)
- **`warmup_steps`**: Number of warmup steps (0-1000, default: 0)
- **`weight_decay`**: Weight decay coefficient (0.0-0.1, default: 0.01)

## Configuration

### Environment Variables
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to service account key file
- `VERTEX_AI_PROJECT`: Default Google Cloud project ID
- `VERTEX_AI_LOCATION`: Default Vertex AI location
- `VERTEX_AI_STAGING_BUCKET`: Default staging bucket for training artifacts

### Fine-Tuning Configuration
```yaml
fine_tuning_config:
  model: "gemini-1.0-pro"
  training_file: "gs://my-bucket/training-data.jsonl"
  validation_file: "gs://my-bucket/validation-data.jsonl"
  hyperparameters:
    epoch_count: 3
    learning_rate_multiplier: 1.0
    adapter_size: "medium"
  suffix: "my-custom-model"
  vertex_project: "my-project"
  vertex_location: "us-central1"
```

## Implementation Status

- [x] Basic structure and types
- [ ] Fine-tuning job creation handler
- [ ] Job monitoring and status tracking
- [ ] Hyperparameter validation
- [ ] Dataset format validation
- [ ] Cost tracking and optimization
- [ ] Model deployment integration
- [ ] Testing
- [ ] Documentation

## Fine-Tuning Job States

- `JOB_STATE_QUEUED`: Job is queued for execution
- `JOB_STATE_PENDING`: Job is pending execution
- `JOB_STATE_RUNNING`: Job is currently running
- `JOB_STATE_SUCCEEDED`: Job completed successfully
- `JOB_STATE_FAILED`: Job failed
- `JOB_STATE_CANCELLING`: Job is being cancelled
- `JOB_STATE_CANCELLED`: Job was cancelled
- `JOB_STATE_ERROR`: Job encountered an error

## Cost Considerations

- **Base Model Costs**: Varies by model size and complexity
- **Training Compute**: Based on training duration and compute resources
- **Storage Costs**: Based on dataset size and model artifacts
- **Network Costs**: Based on data transfer between services

## Best Practices

1. **Dataset Quality**: Ensure high-quality, well-formatted training data
2. **Validation Split**: Always include a validation dataset for monitoring
3. **Hyperparameter Tuning**: Start with default values and tune based on results
4. **Cost Monitoring**: Monitor costs and use appropriate compute resources
5. **Model Evaluation**: Evaluate fine-tuned models before deployment
6. **Iterative Improvement**: Use results to improve dataset and hyperparameters

## Limitations

- **Model Compatibility**: Not all models support fine-tuning
- **Dataset Size**: Minimum and maximum dataset size requirements
- **Compute Resources**: Limited by available compute resources
- **Cost**: Fine-tuning can be expensive for large models
- **Time**: Fine-tuning can take hours to days depending on model size

## Integration with Online Prediction

Once fine-tuning is complete, the fine-tuned model can be deployed to an endpoint and used with the online prediction module:

```python
# Use fine-tuned model for predictions
response = completion(
    model="vertex_ai/endpoints/1234567890123456789",  # Fine-tuned model endpoint
    messages=[{"role": "user", "content": "Hello, world!"}],
    custom_llm_provider="vertex_ai",
    vertex_project="my-project",
    vertex_location="us-central1"
)
``` 