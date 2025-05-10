# Featherless AI
https://featherless.ai/

:::tip

**We support ALL Featherless AI models, just set `model=featherless_ai/<any-model-on-featherless>` as a prefix when sending litellm requests. For the complete supported model list, visit https://featherless.ai/models **

:::


## API Key
```python
# env variable
os.environ['FEATHERLESS_AI_API_KEY']
```

## Sample Usage
```python
from litellm import completion
import os

os.environ['FEATHERLESS_AI_API_KEY'] = ""
response = completion(
    model="featherless_ai/featherless-ai/Qwerky-72B", 
    messages=[{"role": "user", "content": "write code for saying hi from LiteLLM"}]
)
```

## Sample Usage - Streaming
```python
from litellm import completion
import os

os.environ['FEATHERLESS_AI_API_KEY'] = ""
response = completion(
    model="featherless_ai/featherless-ai/Qwerky-72B", 
    messages=[{"role": "user", "content": "write code for saying hi from LiteLLM"}],
    stream=True
)

for chunk in response:
    print(chunk)
```

## Chat Models
| Model Name                                 | Function Call                                                                                  |
|---------------------------------------------|-----------------------------------------------------------------------------------------------|
| featherless-ai/Qwerky-72B                   | `completion(model="featherless_ai/featherless-ai/Qwerky-72B", messages)`                      |
| featherless-ai/Qwerky-QwQ-32B               | `completion(model="featherless_ai/featherless-ai/Qwerky-QwQ-32B", messages)`                  |
| Qwen/Qwen2.5-72B-Instruct                   | `completion(model="featherless_ai/Qwen/Qwen2.5-72B-Instruct", messages)`                      |
| all-hands/openhands-lm-32b-v0.1             | `completion(model="featherless_ai/all-hands/openhands-lm-32b-v0.1", messages)`                |
| Qwen/Qwen2.5-Coder-32B-Instruct             | `completion(model="featherless_ai/Qwen/Qwen2.5-Coder-32B-Instruct", messages)`                |
| deepseek-ai/DeepSeek-V3-0324                | `completion(model="featherless_ai/deepseek-ai/DeepSeek-V3-0324", messages)`                   |
| mistralai/Mistral-Small-24B-Instruct-2501   | `completion(model="featherless_ai/mistralai/Mistral-Small-24B-Instruct-2501", messages)`      |
| mistralai/Mistral-Nemo-Instruct-2407        | `completion(model="featherless_ai/mistralai/Mistral-Nemo-Instruct-2407", messages)`           |
| ProdeusUnity/Stellar-Odyssey-12b-v0.0       | `completion(model="featherless_ai/ProdeusUnity/Stellar-Odyssey-12b-v0.0", messages)`          |
