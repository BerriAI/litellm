# Mock Completion Responses - Save Testing Costs

Trying to test making LLM Completion calls without calling the LLM APIs ? 
Pass `mock_response` to `litellm.completion` and litellm will directly return the response without neededing the call the LLM API and spend $$ 

## Using `mock_response`

```python
from litellm import completion 

model = "gpt-3.5-turbo"
messages = [{"role":"user", "content":"Why is LiteLLM amazing?"}]

completion(model=model, messages=messages, mock_response="It's simple to use and easy to get started")
```