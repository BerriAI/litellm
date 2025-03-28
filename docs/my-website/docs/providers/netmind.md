import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Netmind AI 
LiteLLM supports all models on Netmind AI. 

## API Keys

```python 
import os 
os.environ["NETMIND_API_KEY"] = "your-api-key"
```
## Sample Usage

### chat
```python
import os
from litellm import completion 

os.environ["NETMIND_API_KEY"] = "your-api-key"

messages = [{"role": "user", "content": "Write me a poem about the blue sky"}]

completion(model="netmind/meta-llama/Llama-3.3-70B-Instruct", messages=messages)
```
### embedding
```python
import os
from litellm import embedding 

response = embedding(
    model="netmind/nvidia/NV-Embed-v2", input=['I love programming.']
)
print(response)
```


## Netmind AI Models
liteLLM supports `non-streaming` and `streaming` requests to all models on https://www.netmind.ai/

Example Netmind Usage - Note: liteLLM supports all models deployed on Netmind


### LLMs models
| Model Name                                | Function Call                                                       |
|-------------------------------------------|---------------------------------------------------------------------|
| netmind/deepseek-ai/DeepSeek-R1           | `completion('netmind/deepseek-ai/DeepSeek-R1', messages)` |
| netmind/deepseek-ai/DeepSeek-V3           | `completion('netmind/deepseek-ai/DeepSeek-V3', messages)` |
| netmind/meta-llama/Llama-3.3-70B-Instruct | `completion('netmind/meta-llama/Llama-3.3-70B-Instruct', messages)` |
| netmind/meta-llama/Meta-Llama-3.1-405B    | `completion('netmind/meta-llama/Meta-Llama-3.1-405B', messages)` |
| netmind/Llama3.1-8B-Chinese-Chat          | `completion('netmind/Llama3.1-8B-Chinese-Chat ', messages)` |
| netmind/Qwen/Qwen2.5-72B-Instruct         | `completion('netmind/Qwen/Qwen2.5-72B-Instruct', messages)` |
| netmind/Qwen/QwQ-32B                      | `completion('netmind/Qwen/QwQ-32B', messages)` |
| netmind/deepseek-ai/Janus-Pro-7B          | `completion('netmind/deepseek-ai/Janus-Pro-7B', messages)` |

### Embedding models
| Model Name                         | Function Call                                                     |
|------------------------------------|-------------------------------------------------------------------|
| netmind/BAAI/bge-m3                | `completion('netmind/BAAI/bge-m3', inputs)`                       |
| netmind/nvidia/NV-Embed-v2         | `completion('netmind/nvidia/NV-Embed-v2', inputs)` |
| netmind/dunzhang/stella_en_1.5B_v5 | `completion('netmind/dunzhang/stella_en_1.5B_v5', inputs)` |



