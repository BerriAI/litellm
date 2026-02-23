import Image from '@theme/IdealImage';

# Literal AI - Log, Evaluate, Monitor

[Literal AI](https://literalai.com) is a collaborative observability, evaluation and analytics platform for building production-grade LLM apps.

<Image img={require('../../img/literalai.png')} />

## Pre-Requisites

Ensure you have the `literalai` package installed:

```shell
pip install literalai litellm
```

## Quick Start

```python
import litellm
import os

os.environ["LITERAL_API_KEY"] = ""
os.environ['OPENAI_API_KEY']= ""
os.environ['LITERAL_BATCH_SIZE'] = "1" # You won't see logs appear until the batch is full and sent

litellm.success_callback = ["literalai"] # Log Input/Output to LiteralAI
litellm.failure_callback = ["literalai"] # Log Errors to LiteralAI

# openai call
response = litellm.completion(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Hi 👋 - i'm openai"}
  ]
)
```

## Multi Step Traces

This integration is compatible with the Literal AI SDK decorators, enabling conversation and agent tracing

```py
import litellm
from literalai import LiteralClient
import os

os.environ["LITERAL_API_KEY"] = ""
os.environ['OPENAI_API_KEY']= ""
os.environ['LITERAL_BATCH_SIZE'] = "1" # You won't see logs appear until the batch is full and sent

litellm.input_callback = ["literalai"] # Support other Literal AI decorators and prompt templates
litellm.success_callback = ["literalai"] # Log Input/Output to LiteralAI
litellm.failure_callback = ["literalai"] # Log Errors to LiteralAI

literalai_client = LiteralClient()

@literalai_client.run
def my_agent(question: str):
    # agent logic here
    response = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": question}
        ],
        metadata={"literalai_parent_id": literalai_client.get_current_step().id}
    )
    return response

my_agent("Hello world")

# Waiting to send all logs before exiting, not needed in a production server
literalai_client.flush()
```

Learn more about [Literal AI logging capabilities](https://docs.literalai.com/guides/logs).

## Bind a Generation to its Prompt Template

This integration works out of the box with prompts managed on Literal AI. This means that a specific LLM generation will be bound to its template.

Learn more about [Prompt Management](https://docs.literalai.com/guides/prompt-management#pull-a-prompt-template-from-literal-ai) on Literal AI.

## OpenAI Proxy Usage

If you are using the Lite LLM proxy, you can use the Literal AI OpenAI instrumentation to log your calls.

```py
from literalai import LiteralClient
from openai import OpenAI

client = OpenAI(
    api_key="anything",            # litellm proxy virtual key
    base_url="http://0.0.0.0:4000" # litellm proxy base_url
)

literalai_client = LiteralClient(api_key="")

# Instrument the OpenAI client
literalai_client.instrument_openai()

settings = {
    "model": "gpt-3.5-turbo", # model you want to send litellm proxy
    "temperature": 0,
    # ... more settings
}

response = client.chat.completions.create(
        messages=[
            {
                "content": "You are a helpful bot, you always reply in Spanish",
                "role": "system"
            },
            {
                "content": message.content,
                "role": "user"
            }
        ],
        **settings
    )

```
