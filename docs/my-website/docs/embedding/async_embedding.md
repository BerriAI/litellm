# litellm.aembedding()

LiteLLM provides an asynchronous version of the `embedding` function called `aembedding`
### Usage
```python
from litellm import aembedding
import asyncio

async def test_get_response():
    response = await aembedding('text-embedding-ada-002', input=["good morning from litellm"])
    return response

response = asyncio.run(test_get_response())
print(response)
```