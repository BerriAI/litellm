import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Galadriel
https://docs.galadriel.com/api-reference/chat-completion-API

LiteLLM supports all models on Galadriel.

## API Key
```python
import os 
os.environ['GALADRIEL_API_KEY'] = "your-api-key"
```

## Sample Usage
```python
from litellm import completion
import os

os.environ['GALADRIEL_API_KEY'] = ""
response = completion(
    model="galadriel/llama3.1", 
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
)
print(response)
```

## Sample Usage - Streaming
```python
from litellm import completion
import os

os.environ['GALADRIEL_API_KEY'] = ""
response = completion(
    model="galadriel/llama3.1", 
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
    stream=True
)

for chunk in response:
    print(chunk)
```


## Supported Models
### Serverless Endpoints
We support ALL Galadriel AI models, just set `galadriel/` as a prefix when sending completion requests

We support both the complete model name and the simplified name match. 

You can specify the model name either with the full name or with a simplified version e.g. `llama3.1:70b` 

| Model Name                                               | Simplified Name                  | Function Call                                           |
| -------------------------------------------------------- | -------------------------------- | ------------------------------------------------------- |
| neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8               | llama3.1 or llama3.1:8b          | `completion(model="galadriel/llama3.1", messages)`      |
| neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w4a16  | llama3.1:70b                     | `completion(model="galadriel/llama3.1:70b", messages)`  |
| neuralmagic/Meta-Llama-3.1-405B-Instruct-quantized.w4a16 | llama3.1:405b                    | `completion(model="galadriel/llama3.1:405b", messages)` |
| neuralmagic/Mistral-Nemo-Instruct-2407-quantized.w4a16   | mistral-nemo or mistral-nemo:12b | `completion(model="galadriel/mistral-nemo", messages)`  |

