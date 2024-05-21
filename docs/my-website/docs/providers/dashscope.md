# Dashscope
https://dashscope.console.aliyun.com/

**We support ALL Qwen models, just set `dashscope/` as a prefix when sending completion requests**

## API Key
```python
# env variable
os.environ['DASHSCOPE_API_KEY']
```

## Sample Usage
```python
from litellm import completion
import os

os.environ['DASHSCOPE_API_KEY'] = ""
response = completion(
    model="dashscope/qwen-turbo", 
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

os.environ['DASHSCOPE_API_KEY'] = ""
response = completion(
    model="dashscope/qwen-turbo", 
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
    stream=True
)

for chunk in response:
    print(chunk)
```


## Supported Models - ALL Qwen Models Supported!
We support ALL Qwen models, just set `dashscope/` as a prefix when sending completion requests

| Model Name               | Function Call                                                                                                                                                      |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| qwen-turbo | `completion(model="dashscope/qwen-turbo", messages)` | 
| qwen-plus | `completion(model="dashscope/qwen-plus", messages)` | 
| qwen-max | `completion(model="dashscope/qwen-max", messages)` | 
| qwen-max-longcontext | `completion(model="dashscope/qwen-max-longcontext", messages)` |
| qwen-vl-plus | `completion(model="dashscope/qwen-vl-plus", messages)` |  
| qwen-vl-max | `completion(model="dashscope/qwen-vl-max", messages)` |  
| qwen1.5-110b-chat | `completion(model="dashscope/qwen1.5-110b-chat", messages)` |  
| qwen1.5-72b-chat | `completion(model="dashscope/qwen1.5-72b-chat", messages)` |  
| qwen1.5-32b-chat | `completion(model="dashscope/qwen1.5-32b-chat", messages)` |  
| qwen1.5-14b-chat | `completion(model="dashscope/qwen1.5-14b-chat", messages)` |  
| qwen1.5-7b-chat | `completion(model="dashscope/qwen1.5-7b-chat", messages)` |  
| qwen1.5-1.8b-chat | `completion(model="dashscope/qwen1.5-1.8b-chat", messages)` |  
| qwen1.5-0.5b-chat | `completion(model="dashscope/qwen1.5-0.5b-chat", messages)` |  
| codeqwen1.5-7b-chat | `completion(model="dashscope/codeqwen1.5-7b-chat", messages)` |  
| qwen-72b-chat | `completion(model="dashscope/qwen-72b-chat", messages)` |  
| qwen-14b-chat | `completion(model="dashscope/qwen-14b-chat", messages)` |  
| qwen-7b-chat| `completion(model="dashscope/qwen-7b-chat", messages)` |  
| qwen-1.8b-longcontext-chat | `completion(model="dashscope/qwen-1.8b-longcontext-chat", messages)` |  
| qwen-1.8b-chat | `completion(model="dashscope/qwen-1.8b-chat", messages)` |  
```