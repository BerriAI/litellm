# Dashscope (Qwen API)
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


[DashScope Model List](https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope?spm=a2c4g.11186623.help-menu-2400256.d_2_8_0.1efd516e2tTXBn&scm=20140722.H_2833609._.OR_help-T_cn~zh-V_1#7f9c78ae99pwz)

| Model Name               | Function Call                                                                                                                                                      |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| qwen-turbo | `completion(model="dashscope/qwen-turbo", messages)` | 
| qwen-plus | `completion(model="dashscope/qwen-plus", messages)` | 
| qwen-max | `completion(model="dashscope/qwen-max", messages)` | 
| qwen-turbo-latest | `completion(model="dashscope/qwen-turbo-latest", messages)` | 
| qwen-plus-latest | `completion(model="dashscope/qwen-plus-latest", messages)` | 
| qwen-max-latest | `completion(model="dashscope/qwen-max-latest", messages)` | 
| qwen-vl-plus | `completion(model="dashscope/qwen-vl-plus", messages)` |  
| qwen-vl-max | `completion(model="dashscope/qwen-vl-max", messages)` |  
| qwq-32b | `completion(model="dashscope/qwq-32b", messages)` |  
| qwq-32b-preview | `completion(model="dashscope/qwq-32b-preview", messages)` |  
| qwen3-235b-a22b | `completion(model="dashscope/qwen3-235b-a22b", messages)` |  
| qwen3-32b | `completion(model="dashscope/qwen3-32b", messages)` |  
| qwen3-30b-a3b | `completion(model="dashscope/qwen3-30b-a3b", messages)` |
```