# DeepInfra
https://deepinfra.com/
## Sample Usage
```python
from litellm import completion
import os

os.environ['DEEPINFRA_API_KEY'] = ""
response = completion(
    model="deepinfra/meta-llama/Llama-2-70b-chat-hf", 
    messages=[{"role": "user", "content": "write code for saying hi from LiteLLM"}]
)
```

## Sample Usage - Streaming
```python
from litellm import completion
import os

os.environ['DEEPINFRA_API_KEY'] = ""
response = completion(
    model="deepinfra/meta-llama/Llama-2-70b-chat-hf", 
    messages=[{"role": "user", "content": "write code for saying hi from LiteLLM"}],
    stream=True
)

for chunk in response:
    print(chunk)
```

## Chat Models
| Model Name       | Function Call                        | Required OS Variables    |
|------------------|--------------------------------------|-------------------------|
| meta-llama/Llama-2-70b-chat-hf  | `completion(model="deepinfra/meta-llama/Llama-2-70b-chat-hf", messages)` | `os.environ['DEEPINFRA_API_KEY']` |
| meta-llama/Llama-2-7b-chat-hf  | `completion(model="deepinfra/meta-llama/Llama-2-7b-chat-hf", messages)` | `os.environ['DEEPINFRA_API_KEY']` |
| codellama/CodeLlama-34b-Instruct-hf | `completion(model="deepinfra/codellama/CodeLlama-34b-Instruct-hf", messages)` | `os.environ['DEEPINFRA_API_KEY']` |


