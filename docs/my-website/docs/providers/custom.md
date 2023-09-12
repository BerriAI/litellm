# Custom LLM API-Endpoints
LiteLLM supports Custom deploy api endpoints

LiteLLM Expects the following input and output for custom LLM API endpoints

### Model Details

For calls to your custom API base ensure:
* Set `api_base="your-api-base"`
* Add `custom/` as a prefix to the `model` param. If your API expects `meta-llama/Llama-2-13b-hf` set `model=custom/meta-llama/Llama-2-13b-hf`

| Model Name       | Function Call                              |
|------------------|--------------------------------------------|
| meta-llama/Llama-2-13b-hf  | `response = completion(model="custom/meta-llama/Llama-2-13b-hf", messages=messages, api_base="https://your-custom-inference-endpoint")` |
| meta-llama/Llama-2-13b-hf  | `response = completion(model="custom/meta-llama/Llama-2-13b-hf", messages=messages, api_base="https://api.autoai.dev/inference")` |

### Example Call to Custom LLM API using LiteLLM
```python
from litellm import completion
response = completion(
    model="custom/meta-llama/Llama-2-13b-hf", 
    messages= [{"content": "what is custom llama?", "role": "user"}],
    temperature=0.2,
    max_tokens=10,
    api_base="https://api.autoai.dev/inference",
    request_timeout=300,
)
print("got response\n", response)
```

#### Setting your Custom API endpoint

Inputs to your custom LLM api bases should follow this format:

```python
resp = requests.post(
    your-api_base, 
    json={
        'model': 'meta-llama/Llama-2-13b-hf', # model name
        'params': {
            'prompt': ["The capital of France is P"],
            'max_tokens': 32,
            'temperature': 0.7,
            'top_p': 1.0,
            'top_k': 40,
        }
    }
)
```

Outputs from your custom LLM api bases should follow this format:   
```python
{
    'data': [
        {
            'prompt': 'The capital of France is P',
            'output': [
                'The capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France'
            ],
            'params': {
                'temperature': 0.7, 
                'top_k': 40, 
                'top_p': 1
            }
        }
        ],
    'message': 'ok'
}
```