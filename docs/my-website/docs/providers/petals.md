# Petals
LiteLLM supports Claude-1, 1.2 and Claude-2.

## Usage
Ensure you add `petals/` as a prefix for all petals LLMs. This sets the custom_llm_provider to petals

```python
from litellm import completion

response = completion(
    model="petals/petals-team/StableBeluga2", 
    messages=[{ "content": "Hello, how are you?","role": "user"}]
)

print(response)
```

### Model Details

| Model Name       | Function Call                              | Required OS Variables                |
|------------------|--------------------------------------------|--------------------------------------|
| claude-instant-1  | `completion('claude-instant-1', messages)` | `os.environ['ANTHROPIC_API_KEY']`       |
| claude-instant-1.2  | `completion('claude-instant-1.2', messages)` | `os.environ['ANTHROPIC_API_KEY']`       |
| claude-2  | `completion('claude-2', messages)` | `os.environ['ANTHROPIC_API_KEY']`       |
