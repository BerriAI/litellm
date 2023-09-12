import Image from '@theme/IdealImage';

# Huggingface

LiteLLM supports Huggingface Inference Endpoints that uses the [text-generation-inference](https://github.com/huggingface/text-generation-inference) format. 

### API KEYS 
```python
import os 
os.environ["HUGGINGFACE_API_KEY"] = ""
```

### Models with Prompt Formatting
For models with special prompt templates (e.g. Llama2), we format the prompt to fit their template. 

**What if we don't support a model you need?**
You can also specify you're own custom prompt formatting, in case we don't have your model covered yet. 

**Does this mean you have to specify a prompt for all models?**
No. By default we'll concatenate your message content to make a prompt. 

**Default Prompt Template**
```python
def default_pt(messages):
    return " ".join(message["content"] for message in messages)
```

[Code for how prompt formats work in LiteLLM](https://github.com/BerriAI/litellm/blob/main/litellm/llms/prompt_templates/factory.py)

#### Models with Special Prompt Templates

| Model Name | Works for Models | Function Call | Required OS Variables |
| -------- | -------- | -------- | -------- |
| meta-llama/Llama-2-7b-chat | All meta-llama llama2 chat models| `completion(model='huggingface/meta-llama/Llama-2-7b', messages=messages, api_base="your_api_endpoint")` | `os.environ['HUGGINGFACE_API_KEY']` |
| tiiuae/falcon-7b-instruct | All falcon instruct models | `completion(model='huggingface/tiiuae/falcon-7b-instruct', messages=messages, api_base="your_api_endpoint")` | `os.environ['HUGGINGFACE_API_KEY']` |
| mosaicml/mpt-7b-chat | All mpt chat models | `completion(model='huggingface/mosaicml/mpt-7b-chat', messages=messages, api_base="your_api_endpoint")` | `os.environ['HUGGINGFACE_API_KEY']` |
| codellama/CodeLlama-34b-Instruct-hf | All codellama instruct models | `completion(model='huggingface/codellama/CodeLlama-34b-Instruct-hf', messages=messages, api_base="your_api_endpoint")` | `os.environ['HUGGINGFACE_API_KEY']` |
| WizardLM/WizardCoder-Python-34B-V1.0 | All wizardcoder models | `completion(model='huggingface/WizardLM/WizardCoder-Python-34B-V1.0', messages=messages, api_base="your_api_endpoint")` | `os.environ['HUGGINGFACE_API_KEY']` |
| Phind/Phind-CodeLlama-34B-v2 | All phind-codellama models | `completion(model='huggingface/Phind/Phind-CodeLlama-34B-v2', messages=messages, api_base="your_api_endpoint")` | `os.environ['HUGGINGFACE_API_KEY']` |

#### Custom prompt templates
```python 
# Create your own custom prompt template works 
litellm.register_prompt_template(
	    model="togethercomputer/LLaMA-2-7B-32K",
	    roles={
            "system": {
                "pre_message": "[INST] <<SYS>>\n",
                "post_message": "\n<</SYS>>\n [/INST]\n"
            },
            "user": {
                "pre_message": "[INST] ",
                "post_message": " [/INST]\n"
            }, 
            "assistant": {
                "post_message": "\n"
            }
        }
    )

def test_huggingface_custom_model():
    model = "huggingface/togethercomputer/LLaMA-2-7B-32K"
    response = completion(model=model, messages=messages, api_base="https://ecd4sb5n09bo4ei2.us-east-1.aws.endpoints.huggingface.cloud")
    print(response['choices'][0]['message']['content'])
    return response

test_huggingface_custom_model()
```

[Implementation Code](https://github.com/BerriAI/litellm/blob/c0b3da2c14c791a0b755f0b1e5a9ef065951ecbf/litellm/llms/huggingface_restapi.py#L52)

## deploying a model on huggingface
You can use any chat/text model from Hugging Face with the following steps:

* Copy your model id/url from Huggingface Inference Endpoints
    - [ ] Go to https://ui.endpoints.huggingface.co/
    - [ ] Copy the url of the specific model you'd like to use 
    <Image img={require('../../img/hf_inference_endpoint.png')} alt="HF_Dashboard" style={{ maxWidth: '50%', height: 'auto' }}/>
* Set it as your model name
* Set your HUGGINGFACE_API_KEY as an environment variable

Need help deploying a model on huggingface? [Check out this guide.](https://huggingface.co/docs/inference-endpoints/guides/create_endpoint)

## usage 

You need to tell LiteLLM when you're calling Huggingface.  


Do that by passing in the custom llm provider as part of the model name -  
completion(model="<custom_llm_provider>/<model_name>",...). 

Model name - `WizardLM/WizardCoder-Python-34B-V1.0`

Model id - `https://ji16r2iys9a8rjk2.us-east-1.aws.endpoints.huggingface.cloud`

```
import os 
from litellm import completion 

# Set env variables
os.environ["HUGGINGFACE_API_KEY"] = "huggingface_api_key"

messages = [{ "content": "There's a llama in my garden 😱 What should I do?","role": "user"}]

# model = <custom_llm_provider>/<model_id>
response = completion(model="huggingface/WizardLM/WizardCoder-Python-34B-V1.0", messages=messages, api_base="https://ji16r2iys9a8rjk2.us-east-1.aws.endpoints.huggingface.cloud")

print(response)
```

# output

Same as the OpenAI format, but also includes logprobs. [See the code](https://github.com/BerriAI/litellm/blob/b4b2dbf005142e0a483d46a07a88a19814899403/litellm/llms/huggingface_restapi.py#L115)

```
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "\ud83d\ude31\n\nComment: @SarahSzabo I'm",
        "role": "assistant",
        "logprobs": -22.697942825499993
      }
    }
  ],
  "created": 1693436637.38206,
  "model": "https://ji16r2iys9a8rjk2.us-east-1.aws.endpoints.huggingface.cloud",
  "usage": {
    "prompt_tokens": 14,
    "completion_tokens": 11,
    "total_tokens": 25
  }
}
```

# FAQ 
**Does this support stop sequences?**

Yes, we support stop sequences - and you can pass as many as allowed by Huggingface (or any provider!)

**How do you deal with repetition penalty?**

We map the presence penalty parameter in openai to the repetition penalty parameter on Huggingface. [See code](https://github.com/BerriAI/litellm/blob/b4b2dbf005142e0a483d46a07a88a19814899403/litellm/utils.py#L757). 

We welcome any suggestions for improving our Huggingface integration - Create an [issue](https://github.com/BerriAI/litellm/issues/new/choose)/[Join the Discord](https://discord.com/invite/wuPM9dRgDw)!