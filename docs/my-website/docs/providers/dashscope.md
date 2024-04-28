import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Alibaba Cloud DashScope
Qwen LLMs are supported on [Dashscope](https://dashscope.aliyun.com/)


## Pre-requisites
* `pip install dashscope --upgrade`
* Get API Key - https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key


### API KEYS
```python
import os 
os.environ["DASHSCOPE_API_KEY"] = "" # API key
```

### Completion - using api_key

```python
import litellm

# dashscope call
response = litellm.completion(
    model = "dashscope/<model name>",                   # dashscope/<model name> 
    api_key  = "",                                      # dashscope api key
    messages = [{"role": "user", "content": "good morning"}],
)
```

### Models
DashScope provides many language models, including Qwen models. https://help.aliyun.com/zh/dashscope/developer-reference/model-introduction. 

Use liteLLM to easily call models provided on DashScope.

Usage: Pass `model=dashscope/<Model ID>`

| Model Name           | Function Call                                                           | 
|----------------------|-------------------------------------------------------------------------|
| qwen-turbo           | `completion(model='dashscope/qwen-turbo', messages=messages)`           |
| qwen-plus            | `completion(model='dashscope/qwen-plus', messages=messages)`            |
| qwen-max             | `completion(model='dashscope/qwen-max', messages=messages)`             |
| qwen-max-longcontext | `completion(model='dashscope/qwen-max-longcontext', messages=messages)` |