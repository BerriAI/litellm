import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Huggingface

LiteLLM supports the following types of Huggingface models:
* Text-generation-interface: [Here's all the models that use this format](https://huggingface.co/models?other=text-generation-inference).
* Conversational task: [Here's all the models that use this format](https://huggingface.co/models?pipeline_tag=conversational).
* Non TGI/Conversational-task LLMs

## Usage 

<a target="_blank" href="https://colab.research.google.com/github/BerriAI/litellm/blob/main/cookbook/LiteLLM_HuggingFace.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

You need to tell LiteLLM when you're calling Huggingface. 
This is done by adding the "huggingface/" prefix to `model`, example `completion(model="huggingface/<model_name>",...)`. 

<Tabs>
<TabItem value="tgi" label="Text-generation-interface (TGI)">

```python
import os 
from litellm import completion 

# [OPTIONAL] set env var
os.environ["HUGGINGFACE_API_KEY"] = "huggingface_api_key" 

messages = [{ "content": "There's a llama in my garden 😱 What should I do?","role": "user"}]

# e.g. Call 'WizardLM/WizardCoder-Python-34B-V1.0' hosted on HF Inference endpoints
response = completion(
  model="huggingface/WizardLM/WizardCoder-Python-34B-V1.0", 
  messages=messages, 
  api_base="https://my-endpoint.huggingface.cloud"
)

print(response)
```

</TabItem>
<TabItem value="conv" label="Conversational-task (BlenderBot, etc.)">

```python
import os 
from litellm import completion 

# [OPTIONAL] set env var
os.environ["HUGGINGFACE_API_KEY"] = "huggingface_api_key" 

messages = [{ "content": "There's a llama in my garden 😱 What should I do?","role": "user"}]

# e.g. Call 'facebook/blenderbot-400M-distill' hosted on HF Inference endpoints
response = completion(
  model="huggingface/facebook/blenderbot-400M-distill", 
  messages=messages, 
  api_base="https://my-endpoint.huggingface.cloud"
)

print(response)
```
</TabItem>
<TabItem value="none" label="Non TGI/Conversational-task LLMs">

```python
import os 
from litellm import completion 

# [OPTIONAL] set env var
os.environ["HUGGINGFACE_API_KEY"] = "huggingface_api_key" 

messages = [{ "content": "There's a llama in my garden 😱 What should I do?","role": "user"}]

# e.g. Call 'roneneldan/TinyStories-3M' hosted on HF Inference endpoints
response = completion(
  model="huggingface/roneneldan/TinyStories-3M", 
  messages=messages,
  api_base="https://p69xlsj6rpno5drq.us-east-1.aws.endpoints.huggingface.cloud",
)

print(response)
```
</TabItem>
</Tabs>

## Streaming

<a target="_blank" href="https://colab.research.google.com/github/BerriAI/litellm/blob/main/cookbook/LiteLLM_HuggingFace.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

You need to tell LiteLLM when you're calling Huggingface. 
This is done by adding the "huggingface/" prefix to `model`, example `completion(model="huggingface/<model_name>",...)`. 

```python
import os 
from litellm import completion 

# [OPTIONAL] set env var
os.environ["HUGGINGFACE_API_KEY"] = "huggingface_api_key" 

messages = [{ "content": "There's a llama in my garden 😱 What should I do?","role": "user"}]

# e.g. Call 'facebook/blenderbot-400M-distill' hosted on HF Inference endpoints
response = completion(
  model="huggingface/facebook/blenderbot-400M-distill", 
  messages=messages, 
  api_base="https://my-endpoint.huggingface.cloud",
  stream=True
)

print(response)
for chunk in response:
  print(chunk)
```

## Embedding
LiteLLM supports Huggingface's [text-embedding-inference](https://github.com/huggingface/text-embeddings-inference) format.
```python
from litellm import embedding
import os
os.environ['HUGGINGFACE_API_KEY'] = ""
response = embedding(
    model='huggingface/microsoft/codebert-base', 
    input=["good morning from litellm"]
)
```

## Advanced

### Setting API KEYS + API BASE
If required, you can set the api key + api base, set it in your os environment. [Code for how it's sent](https://github.com/BerriAI/litellm/blob/0100ab2382a0e720c7978fbf662cc6e6920e7e03/litellm/llms/huggingface_restapi.py#L25)

```python
import os 
os.environ["HUGGINGFACE_API_KEY"] = ""
os.environ["HUGGINGFACE_API_BASE"] = "" 
```

### Viewing Log probs

#### Using `decoder_input_details` - OpenAI `echo`
The `echo` param is supported by OpenAI Completions - Use `litellm.text_completion()` for this 
```python
from litellm import text_completion
response = text_completion(
    model="huggingface/bigcode/starcoder", 
    prompt="good morning", 
    max_tokens=10, logprobs=10,
    echo=True
)
 ```

#### Output
 ```json
{
   "id":"chatcmpl-3fc71792-c442-4ba1-a611-19dd0ac371ad",
   "object":"text_completion",
   "created":1698801125.936519,
   "model":"bigcode/starcoder",
   "choices":[
      {
         "text":", I'm going to make you a sand",
         "index":0,
         "logprobs":{
            "tokens":[
               "good",
               " morning",
               ",",
               " I",
               "'m",
               " going",
               " to",
               " make",
               " you",
               " a",
               " s",
               "and"
            ],
            "token_logprobs":[
               "None",
               -14.96875,
               -2.2285156,
               -2.734375,
               -2.0957031,
               -2.0917969,
               -0.09429932,
               -3.1132812,
               -1.3203125,
               -1.2304688,
               -1.6201172,
               -0.010292053
            ]
         },
         "finish_reason":"length"
      }
   ],
   "usage":{
      "completion_tokens":9,
      "prompt_tokens":2,
      "total_tokens":11
   }
}
```

### Models with Prompt Formatting
For models with special prompt templates (e.g. Llama2), we format the prompt to fit their template. 

#### Models with natively Supported Prompt Templates

| Model Name | Works for Models | Function Call | Required OS Variables |
| -------- | -------- | -------- | -------- |
| mistralai/Mistral-7B-Instruct-v0.1 | mistralai/Mistral-7B-Instruct-v0.1| `completion(model='huggingface/mistralai/Mistral-7B-Instruct-v0.1', messages=messages, api_base="your_api_endpoint")` | `os.environ['HUGGINGFACE_API_KEY']` |
| meta-llama/Llama-2-7b-chat | All meta-llama llama2 chat models| `completion(model='huggingface/meta-llama/Llama-2-7b', messages=messages, api_base="your_api_endpoint")` | `os.environ['HUGGINGFACE_API_KEY']` |
| tiiuae/falcon-7b-instruct | All falcon instruct models | `completion(model='huggingface/tiiuae/falcon-7b-instruct', messages=messages, api_base="your_api_endpoint")` | `os.environ['HUGGINGFACE_API_KEY']` |
| mosaicml/mpt-7b-chat | All mpt chat models | `completion(model='huggingface/mosaicml/mpt-7b-chat', messages=messages, api_base="your_api_endpoint")` | `os.environ['HUGGINGFACE_API_KEY']` |
| codellama/CodeLlama-34b-Instruct-hf | All codellama instruct models | `completion(model='huggingface/codellama/CodeLlama-34b-Instruct-hf', messages=messages, api_base="your_api_endpoint")` | `os.environ['HUGGINGFACE_API_KEY']` |
| WizardLM/WizardCoder-Python-34B-V1.0 | All wizardcoder models | `completion(model='huggingface/WizardLM/WizardCoder-Python-34B-V1.0', messages=messages, api_base="your_api_endpoint")` | `os.environ['HUGGINGFACE_API_KEY']` |
| Phind/Phind-CodeLlama-34B-v2 | All phind-codellama models | `completion(model='huggingface/Phind/Phind-CodeLlama-34B-v2', messages=messages, api_base="your_api_endpoint")` | `os.environ['HUGGINGFACE_API_KEY']` |


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

### Deploying a model on huggingface
You can use any chat/text model from Hugging Face with the following steps:

* Copy your model id/url from Huggingface Inference Endpoints
    - [ ] Go to https://ui.endpoints.huggingface.co/
    - [ ] Copy the url of the specific model you'd like to use 
    <Image img={require('../../img/hf_inference_endpoint.png')} alt="HF_Dashboard" style={{ maxWidth: '50%', height: 'auto' }}/>
* Set it as your model name
* Set your HUGGINGFACE_API_KEY as an environment variable

Need help deploying a model on huggingface? [Check out this guide.](https://huggingface.co/docs/inference-endpoints/guides/create_endpoint)


# output

Same as the OpenAI format, but also includes logprobs. [See the code](https://github.com/BerriAI/litellm/blob/b4b2dbf005142e0a483d46a07a88a19814899403/litellm/llms/huggingface_restapi.py#L115)

```json
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