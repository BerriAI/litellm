# Together AI 
LiteLLM supports all models on Together AI. 

### API KEYS

```python 
import os 
os.environ["TOGETHERAI_API_KEY"] = ""
```

### Sample Usage

```python
from litellm import completion 

# set env variable 
os.environ["TOGETHERAI_API_KEY"] = ""

messages = [{"role": "user", "content": "Write me a poem about the blue sky"}]

completion(model="togethercomputer/Llama-2-7B-32K-Instruct", messages=messages, custom_llm_provider="together_ai")
```

### Together AI Models
liteLLM supports `non-streaming` and `streaming` requests to all models on https://api.together.xyz/

Example TogetherAI Usage - Note: liteLLM supports all models deployed on TogetherAI

| Model Name                        | Function Call                                                          | Required OS Variables           |
|-----------------------------------|------------------------------------------------------------------------|---------------------------------|
| togethercomputer/llama-2-70b-chat  | `completion('togethercomputer/llama-2-70b-chat', messages)`            | `os.environ['TOGETHERAI_API_KEY']` |
| togethercomputer/LLaMA-2-13b-chat  | `completion('togethercomputer/LLaMA-2-13b-chat', messages)`            | `os.environ['TOGETHERAI_API_KEY']` |
| togethercomputer/code-and-talk-v1 | `completion('togethercomputer/code-and-talk-v1', messages)`           | `os.environ['TOGETHERAI_API_KEY']` |
| togethercomputer/creative-v1      | `completion('togethercomputer/creative-v1', messages)`                | `os.environ['TOGETHERAI_API_KEY']` |
| togethercomputer/yourmodel        | `completion('togethercomputer/yourmodel', messages)`                  | `os.environ['TOGETHERAI_API_KEY']` |


### Prompt Templates

Using a chat model on Together AI with it's own prompt format?

#### Using Llama2 Instruct models
If you're using Together AI's Llama2 variants( `model=togethercomputer/llama-2..-instruct`), LiteLLM can automatically translate between the OpenAI prompt format and the TogetherAI Llama2 one (`[INST]..[/INST]`). 

```python
from litellm import completion 

# set env variable 
os.environ["TOGETHERAI_API_KEY"] = ""

messages = [{"role": "user", "content": "Write me a poem about the blue sky"}]

completion(model="together_ai/togethercomputer/Llama-2-7B-32K-Instruct", messages=messages)
```

#### Using another model

You can create a custom prompt template on LiteLLM (and we [welcome PRs](https://github.com/BerriAI/litellm) to add them to the main repo 🤗)

Let's make one for `OpenAssistant/llama2-70b-oasst-sft-v10`!

The accepted template format is: [Reference](https://huggingface.co/OpenAssistant/llama2-70b-oasst-sft-v10-)
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

### Advanced Usage

Instead of using the `custom_llm_provider` arg to specify which provider you're using (e.g. together ai), you can just pass the provider name as part of the model name, and LiteLLM will parse it out. 

Expected format: <custom_llm_provider>/<model_name>

e.g. completion(model="together_ai/togethercomputer/Llama-2-7B-32K-Instruct", ...)

```python
from litellm import completion 

# set env variable 
os.environ["TOGETHERAI_API_KEY"] = ""

messages = [{"role": "user", "content": "Write me a poem about the blue sky"}]

completion(model="together_ai/togethercomputer/Llama-2-7B-32K-Instruct", messages=messages)
```