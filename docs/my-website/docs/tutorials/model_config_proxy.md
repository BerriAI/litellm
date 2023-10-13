# Customize Prompt Templates on OpenAI-Compatible server 

**You will learn:** How to set a custom prompt template on our OpenAI compatible server. 

## Step 1: Start OpenAI Compatible server
Let's spin up a local OpenAI-compatible server, to call a deployed `codellama/CodeLlama-34b-Instruct-hf` model using Huggingface's [Text-Generation-Inference (TGI)](https://github.com/huggingface/text-generation-inference) format.

```shell
$ litellm --model huggingface/codellama/CodeLlama-34b-Instruct-hf --api_base https://my-endpoint.com

# OpenAI compatible server running on http://0.0.0.0/8000
```

By default this will follow [our formatting](https://github.com/BerriAI/litellm/blob/9932371f883c55fd0f3142f91d9c40279e8fe241/litellm/llms/prompt_templates/factory.py#L10) for CodeLlama (based on the [Huggingface's documentation](https://huggingface.co/blog/codellama#conversational-instructions)). 

```python
[INST] <<SYS>>
{{ system_prompt }}
<</SYS>>

{{ user_msg_1 }} [/INST] {{ model_answer_1 }} [INST] {{ user_msg_2 }} [/INST]
```

But this lacks BOS(`<s>`) and EOS(`</s>`) tokens.

So instead of using the LiteLLM default, let's use our own prompt template to use these in our messages. 

## Step 2: Create Custom Prompt Template

Our litellm server accepts prompt templates as part of a config file. You can save api keys, fallback models, prompt templates etc. in this config. [See a complete config file](../proxy_server.md)

For now, let's just create a simple config file with our prompt template, and tell our server about it. 

Create a file called `litellm_config.toml`, and here's how we'll format it:

```toml
[model.<your-model-name>]
// our prompt template
```

We want to add:
* BOS (`<s>`) tokens at the start of every System and Human message
* EOS (`</s>`) tokens at the end of every assistant message. 

This is what it looks like: 
```toml
[model."huggingface/codellama/CodeLlama-34b-Instruct-hf".prompt_template] 
MODEL_SYSTEM_MESSAGE_START_TOKEN = "<s>[INST]  <<SYS>>\n]" 
MODEL_SYSTEM_MESSAGE_END_TOKEN = "\n<</SYS>>\n [/INST]\n"

MODEL_USER_MESSAGE_START_TOKEN = "<s>[INST] " 
MODEL_USER_MESSAGE_END_TOKEN = " [/INST]\n"

MODEL_ASSISTANT_MESSAGE_START_TOKEN = ""
MODEL_ASSISTANT_MESSAGE_END_TOKEN = "</s>"
```

## Step 3: Save template

Let's save our custom template to our litellm server by running:
```shell
litellm --config -f ./litellm_config.toml 
```

LiteLLM will save a copy of this file in it's package, so it can persist these settings across restarts.