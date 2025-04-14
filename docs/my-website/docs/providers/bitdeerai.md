import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Bitdeer AI 

:::info
**We support ALL Bitdeer AI models, just set `bitdeerai/` as a prefix when sending completion requests**
:::

## API Keys

```python 
import os 
os.environ["BITDEERAI_API_KEY"] = "your-api-key"
```
## Sample Usage

### chat
```python
import os
from litellm import completion 

os.environ["BITDEERAI_API_KEY"] = "your-api-key"

messages = [
    {
        "role":"system",
        "content":"You are a knowledgeable assistant. Provide concise and clear explanations to scientific questions."
    },
    {
        "role": "user",
        "content": "Can you explain the theory of evolution in simple terms?"
    }
]

completion(model="bitdeerai/OpenGVLab/InternVL2_5-78B-MPO", messages=messages)
```
### embedding
```python
import os
from litellm import embedding 

response = embedding(
    model="bitdeerai/BAAI/bge-m3", input=['The cat danced gracefully under the moonlight, its shadow twirling like a silent partner.']
)
print(response)
```
## Bitdeer AI Models
liteLLM supports `non-streaming` and `streaming` requests to all models on https://www.bitdeer.ai

Example Bitdeer AI Usage - Note: liteLLM supports all models deployed on Bitdeer AI


### LLMs models
| Model Name                                | Function Call                                                       |
|-------------------------------------------|---------------------------------------------------------------------|
| bitdeerai/deepseek-ai/DeepSeek-R1           | `completion('bitdeerai/deepseek-ai/DeepSeek-R1', messages)` |
| bitdeerai/deepseek-ai/DeepSeek-V3           | `completion('bitdeerai/deepseek-ai/DeepSeek-V3', messages)` |
| bitdeerai/Qwen/QwQ-32B                      | `completion('bitdeerai/Qwen/QwQ-32B', messages)` |
| bitdeerai/Qwen/Qwen2.5-VL-72B-Instruct      | `completion('bitdeerai/Qwen/Qwen2.5-VL-72B-Instruct', messages)` |
| bitdeerai/Qwen/Qwen2.5-Coder-32B-Instruct   | `completion('bitdeerai/Qwen/Qwen2.5-Coder-32B-Instruct', messages)` |
| bitdeerai/meta-llama/Llama-3.3-70B-Instruct | `completion('bitdeerai/meta-llama/Llama-3.3-70B-Instruct', messages)` |
| bitdeerai/OpenGVLab/InternVL2_5-78B-MPO     | `completion('bitdeerai/OpenGVLab/InternVL2_5-78B-MPO', messages)` |

### Embedding models
| Model Name                         | Function Call                                                     |
|-----------------------------------------------------|-------------------------------------------------------------------|
| bitdeerai/Alibaba-NLP/gte-Qwen2-7B-instruct         | `completion('bitdeerai/Alibaba-NLP/gte-Qwen2-7B-instruct', inputs)` |
| bitdeerai/BAAI/bge-m3                               | `completion('bitdeerai/BAAI/bge-m3', inputs)`                       |
| bitdeerai/BAAI/bge-large-en-v1.5                    | `completion('bitdeerai/BAAI/bge-m3', inputs)`                       |
| bitdeerai/intfloat/multilingual-e5-large-instruct   | `completion('bitdeerai/intfloat/multilingual-e5-large-instruct', inputs)` |


