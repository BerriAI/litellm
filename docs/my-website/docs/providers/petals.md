# Petals
Petals: https://github.com/bigscience-workshop/petals

## Pre-Requisites
Ensure you have `petals` installed
```shell
pip install git+https://github.com/bigscience-workshop/petals
```

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

| Model Name       | Function Call                              |
|------------------|--------------------------------------------|
| StableBeluga | `completion('petals/petals-team/StableBeluga2', messages)` | 



