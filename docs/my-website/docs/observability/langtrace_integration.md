import Image from '@theme/IdealImage';

# 🔥 Langtrace AI
Monitor, evaluate & improve your LLM apps

## Pre-Requisites
Make an account on [Langtrace AI](https://langtrace.ai/login)

## Quick Start
Use just 2 lines of code, to instantly log your responses **across all providers** with langtrace

```python
litellm.callbacks = ["langtrace"]
langtrace.init()
```

```python
import litellm
import os
from langtrace_python_sdk import langtrace


os.environ["LANGTRACE_API_KEY"] = "<your-api-key>"

# LLM API Keys
os.environ['OPENAI_API_KEY']="<openai-api-key>"

# set langtrace as a callback, litellm will send the data to langtrace
litellm.callbacks = ["langtrace"]

#  init langtrace
langtrace.init()

# openai call
response = completion(
    model="gpt-4o",
    messages=[
        {"content": "respond only in Yoda speak.", "role": "system"},
        {"content": "Hello, how are you?", "role": "user"},
    ],
)
print(response)
```