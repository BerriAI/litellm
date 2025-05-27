---
title: "Maxim - Agent Observability"
description: "Maxim AI streamlines AI application development and deployment by applying traditional software best practices to non-deterministic AI workflows. An end to end Agent Simulation, Evaluation & Observability platform which help teams maintain quality, reliability, and speed throughout the AI application lifecycle."
---

![docs/my-website/img/maxim-litellm.png]()

You can integrate Maxim as an observability platform either using Maxim SDK or via LiteLLM Proxy

## **Integrate using LiteLLM Python SDK**

We will set up Maxim as a logger for LiteLLM to monitor your LLM calls and attach custom traces for advanced analytics.

### Pre-Requisites

Ensure you add the below packages in your requirements.txt file or install them separately using _pip install litellm\>=1.25.0_ and _pip install maxim-py\>=3.6.4_

```
litellm>=1.25.0
maxim-py>=3.6.4
```

### Set API Keys in Environment Variables

In Maxim set up your workspace & create the API keys - [Setting up your workspace](https://www.getmaxim.ai/docs/introduction/quickstart/setting-up-workspace).

Set the following environment variables in your environment or `.env` file

```
MAXIM_API_KEY=
MAXIM_LOG_REPO_ID=
OPENAI_API_KEY=
```

### Get Started

Using Maxim \<\> LiteLLM One Line Integration, you can easily start sending responses across providers to Maxim

```
import litellm
import os
from maxim import Maxim, Config, LoggerConfig
from maxim.logger.litellm import MaximLiteLLMTracer

logger = Maxim().logger()

# One-line integration: add Maxim tracer to LiteLLM callbacks
litellm.callbacks = [MaximLiteLLMTracer(logger)]
```

In the above example -

1. We first created a logger instance using Maxim SDK.
2. We then added Maxim as LiteLLM Logger using the imported _MaximLiteLLMTracer_ class.

Now we can make LLM calls using LiteLLM and check the received logs on Maxim

```
import os
from litellm import acompletion

response = await acompletion(
  model='openai/gpt-4o',
  api_key=os.getenv('OPENAI_API_KEY'),
  messages=[{'role': 'user', 'content': 'Hello, world!'}],
)
```

### Advance Tutorial

In order to attach a custom trace to your LiteLLM calls for advanced tracking, follow the below steps -

1. Import _TraceConfig_ class, it will help you instantiate a custom trace
2. Pass a unique _id_ & _name_ for this custom trace
3. Trigger a trace event with a unique id, it will create a trace section in your Maxim Log repository and hold the spans for this trace id.
4. Make a LiteLLM call and for visibility, provide the metadata with required keys.

```
from maxim.logger.logger import TraceConfig
import uuid

trace = logger.trace(TraceConfig(id=str(uuid.uuid4()), name='litellm-generation'))
trace.event(str(uuid.uuid4()), 'litellm-generation', 'litellm-generation', {})
# Attach trace to LiteLLM call using metadata
response = await acompletion(
  model='openai/gpt-4o',
  api_key=os.getenv('OPENAI_API_KEY'),
  messages=[{'role': 'user', 'content': 'What can you do for me!'}],
  metadata={'maxim': {'trace_id': trace.id, 'span_name': 'litellm-generation'}}
)

print(response.choices[0].message.content)
```

To quickly get started, please check this [Google Colab Notebook](https://colab.research.google.com/github/maximhq/maxim-cookbooks/blob/main/python/observability-online-eval/litellm/litellm-one-line-integration.ipynb)

Check the quick demo of what we just learnt —

![maxim-litellm-gif.gif](docs/my-website/img/maxim-litellm-gif.gif)

## Integrate using **LiteLLM Proxy: One-Line Integration**

You can integrate Maxim observability with your LiteLLM Proxy using the One Line Integration

### Pre-Requisites

Install _litellm[proxy]_ & _maxim-py -_ define them in your requirements.txt file so that they can be installed automatically on dockerisation.

```
pip install litellm[proxy]>=1.30.0 maxim-py==3.6.4 python-dotenv>=0.21.1
```

### Set the API Keys

You need to define the below API Keys in your `.env` file, as covered before, you can check [Setting up your workspace](https://www.getmaxim.ai/docs/introduction/quickstart/setting-up-workspace) to get started with Maxim Workspaces and creating API Keys.

```
MAXIM_API_KEY=
MAXIM_LOG_REPO_ID=
OPENAI_API_KEY=
```

### Defining the Tracer

In your project folder create a file _maxim_proxy\__[_tracer.py_](http://tracer.py) & create an instance of _MaximLiteLLMProxyTracer()_ which is coming from Maxim SDK

```
from maxim.logger.litellm_proxy import MaximLiteLLMProxyTracer

# This single object wires up all LiteLLM traffic to Maxim
litellm_handler = MaximLiteLLMProxyTracer()
```

At present your project structure should look like this -

```
.
├── config.yml
├── maxim_proxy_tracer.py
├── requirements.txt
├── .env
└── (optional) Dockerfile & docker-compose.yml
```

In config.yml, we need to point LiteLLM callback to the tracer from Maxim -

```
litellm_settings:
  callbacks: maxim_proxy_tracer.litellm_handler
```

### Run the Proxy

Start the Proxy via LiteLLM CLI

```
litellm --port 8000 --config config.yml
```

## Support & Talk to Founders

- Check the [Maxim Documentation](https://getmaxim.ai/docs)
- Maxim Github [Link](https://github.com/maximhq)