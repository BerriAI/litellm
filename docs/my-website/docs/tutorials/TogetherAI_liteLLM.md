# Llama2 Together AI Tutorial
https://together.ai/



```python
!pip install litellm
```


```python
import os
from litellm import completion
os.environ["TOGETHERAI_API_KEY"] = "" #@param
user_message = "Hello, whats the weather in San Francisco??"
messages = [{ "content": user_message,"role": "user"}]
```

## Calling Llama2 on TogetherAI
https://api.together.xyz/playground/chat?model=togethercomputer%2Fllama-2-70b-chat

```python
model_name = "together_ai/togethercomputer/llama-2-70b-chat"
response = completion(model=model_name, messages=messages)
print(response)
```

    {'choices': [{'finish_reason': 'stop', 'index': 0, 'message': {'role': 'assistant', 'content': "\n\nI'm not able to provide real-time weather information. However, I can suggest"}}], 'created': 1691629657.9288375, 'model': 'togethercomputer/llama-2-70b-chat', 'usage': {'prompt_tokens': 9, 'completion_tokens': 17, 'total_tokens': 26}}


LiteLLM handles the prompt formatting for Together AI's Llama2 models as well, converting your message to the 
`[INST] <your instruction> [/INST]` format required. 

[Implementation Code](https://github.com/BerriAI/litellm/blob/64f3d3c56ef02ac5544983efc78293de31c1c201/litellm/llms/prompt_templates/factory.py#L17)

## With Streaming


```python
response = completion(model=model_name, messages=messages, together_ai=True, stream=True)
print(response)
for chunk in response:
  print(chunk['choices'][0]['delta']) # same as openai format
```


## Use Llama2 variants with Custom Prompt Templates

Using a version of Llama2 on TogetherAI that needs custom prompt formatting? 

You can create a custom prompt template. 

Let's make one for `OpenAssistant/llama2-70b-oasst-sft-v10`!

The accepted template format is: [Reference](https://huggingface.co/OpenAssistant/llama2-70b-oasst-sft-v10)
```
"""
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
```

Let's register our custom prompt template: [Implementation Code](https://github.com/BerriAI/litellm/blob/64f3d3c56ef02ac5544983efc78293de31c1c201/litellm/llms/prompt_templates/factory.py#L77)
```python
import litellm 

litellm.register_prompt_template(
    model="OpenAssistant/llama2-70b-oasst-sft-v10",
    roles={"system":"<|im_start|>system", "assistant":"<|im_start|>assistant", "user":"<|im_start|>user"}, # tell LiteLLM how you want to map the openai messages to this model
    pre_message_sep= "\n",
    post_message_sep= "\n"
)
```

Let's use it! 

```python
from litellm import completion 

# set env variable 
os.environ["TOGETHERAI_API_KEY"] = ""

messages=[{"role":"user", "content": "Write me a poem about the blue sky"}]

completion(model="together_ai/OpenAssistant/llama2-70b-oasst-sft-v10", messages=messages)
```

**Complete Code**

```python
import litellm 
from litellm import completion

# set env variable 
os.environ["TOGETHERAI_API_KEY"] = ""

litellm.register_prompt_template(
    model="OpenAssistant/llama2-70b-oasst-sft-v10",
    roles={"system":"<|im_start|>system", "assistant":"<|im_start|>assistant", "user":"<|im_start|>user"}, # tell LiteLLM how you want to map the openai messages to this model
    pre_message_sep= "\n",
    post_message_sep= "\n"
)

messages=[{"role":"user", "content": "Write me a poem about the blue sky"}]

response = completion(model="together_ai/OpenAssistant/llama2-70b-oasst-sft-v10", messages=messages)

print(response)
```

**Output**
```json
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": ".\n\nThe sky is a canvas of blue,\nWith clouds that drift and move,",
        "role": "assistant",
        "logprobs": null
      }
    }
  ],
  "created": 1693941410.482018,
  "model": "OpenAssistant/llama2-70b-oasst-sft-v10",
  "usage": {
    "prompt_tokens": 7,
    "completion_tokens": 16,
    "total_tokens": 23
  },
  "litellm_call_id": "f21315db-afd6-4c1e-b43a-0b5682de4b06"
}
```
