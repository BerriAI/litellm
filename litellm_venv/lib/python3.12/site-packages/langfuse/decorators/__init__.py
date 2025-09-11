"""Integrate Langfuse Tracing into your LLM applications with the Langfuse Python SDK using the `@observe()` decorator.

*Simple example (decorator + openai integration)*

```python
from langfuse.decorators import observe
from langfuse.openai import openai # OpenAI integration

@observe()
def story():
    return openai.chat.completions.create(
        model="gpt-3.5-turbo",
        max_tokens=100,
        messages=[
          {"role": "system", "content": "You are a great storyteller."},
          {"role": "user", "content": "Once upon a time in a galaxy far, far away..."}
        ],
    ).choices[0].message.content

@observe()
def main():
    return story()

main()
```

See [docs](https://langfuse.com/docs/sdk/python/decorators) for more information.
"""

from .langfuse_decorator import langfuse_context, observe, LangfuseDecorator

__all__ = ["langfuse_context", "observe", "LangfuseDecorator"]
