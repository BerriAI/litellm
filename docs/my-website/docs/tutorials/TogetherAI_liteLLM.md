## liteLLM Together AI Tutorial
https://together.ai/



```python
!pip install litellm==0.1.371
```


```python
import os
from litellm import completion
os.environ["TOGETHER_AI_TOKEN"] = "" #@param
user_message = "Hello, whats the weather in San Francisco??"
messages = [{ "content": user_message,"role": "user"}]
```

## Calling togethercomputer/llama-2-70b-chat
https://api.together.xyz/playground/chat?model=togethercomputer%2Fllama-2-70b-chat


```python
model_name = "togethercomputer/llama-2-70b-chat"
response = completion(model=model_name, messages=messages, together_ai=True)
print(response)
```

    {'choices': [{'finish_reason': 'stop', 'index': 0, 'message': {'role': 'assistant', 'content': "\n\nI'm not able to provide real-time weather information. However, I can suggest"}}], 'created': 1691629657.9288375, 'model': 'togethercomputer/llama-2-70b-chat', 'usage': {'prompt_tokens': 9, 'completion_tokens': 17, 'total_tokens': 26}}


## With Streaming


```python
response = completion(model=model_name, messages=messages, together_ai=True, stream=True)
print(response)
for chunk in response:
  print(chunk['choices'][0]['delta']) # same as openai format
```

    <litellm.utils.CustomStreamWrapper object at 0x7ad005e93ee0>
    {'role': 'assistant', 'content': '\\n'}
    {'role': 'assistant', 'content': '\\n'}
    {'role': 'assistant', 'content': 'I'}
    {'role': 'assistant', 'content': 'm'}
    {'role': 'assistant', 'content': ' not'}
    {'role': 'assistant', 'content': ' able'}
    {'role': 'assistant', 'content': ' to'}
    {'role': 'assistant', 'content': ' provide'}
    {'role': 'assistant', 'content': ' real'}
    {'role': 'assistant', 'content': '-'}
    {'role': 'assistant', 'content': 'time'}
    {'role': 'assistant', 'content': ' weather'}
    {'role': 'assistant', 'content': ' information'}
    {'role': 'assistant', 'content': '.'}
    {'role': 'assistant', 'content': ' However'}
    {'role': 'assistant', 'content': ','}
    {'role': 'assistant', 'content': ' I'}
    {'role': 'assistant', 'content': ' can'}

