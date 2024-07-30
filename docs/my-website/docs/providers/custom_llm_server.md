# Custom API Server (Custom Format)

Call your custom torch-serve / internal LLM APIs via LiteLLM

:::info

For calling an openai-compatible endpoint, [go here](./openai_compatible.md)
:::

## Quick Start 

```python
import litellm
from litellm import CustomLLM, completion, get_llm_provider


class MyCustomLLM(CustomLLM):
    def completion(self, *args, **kwargs) -> litellm.ModelResponse:
        return litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello world"}],
            mock_response="Hi!",
        )  # type: ignore

litellm.custom_provider_map = [ # 👈 KEY STEP - REGISTER HANDLER
        {"provider": "my-custom-llm", "custom_handler": my_custom_llm}
    ]

resp = completion(
        model="my-custom-llm/my-fake-model",
        messages=[{"role": "user", "content": "Hello world!"}],
    )

assert resp.choices[0].message.content == "Hi!"
```

## OpenAI Proxy Usage

1. Setup your `custom_handler.py` file 

```python
import litellm
from litellm import CustomLLM, completion, get_llm_provider


class MyCustomLLM(CustomLLM):
    def completion(self, *args, **kwargs) -> litellm.ModelResponse:
        return litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello world"}],
            mock_response="Hi!",
        )  # type: ignore

    async def acompletion(self, *args, **kwargs) -> litellm.ModelResponse:
        return litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello world"}],
            mock_response="Hi!",
        )  # type: ignore


my_custom_llm = MyCustomLLM()
```

2. Add to `config.yaml` 

In the config below, we pass

python_filename: `custom_handler.py`
custom_handler_instance_name: `my_custom_llm`. This is defined in Step 1

custom_handler: `custom_handler.my_custom_llm`

```yaml
model_list:
  - model_name: "test-model"             
    litellm_params:
      model: "openai/text-embedding-ada-002"
  - model_name: "my-custom-model"
    litellm_params:
      model: "my-custom-llm/my-model"

litellm_settings:
  custom_provider_map:
  - {"provider": "my-custom-llm", "custom_handler": custom_handler.my_custom_llm}
```

```bash
litellm --config /path/to/config.yaml
```

3. Test it! 

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "model": "my-custom-model",
    "messages": [{"role": "user", "content": "Say \"this is a test\" in JSON!"}],
}'
```

Expected Response

```
{
    "id": "chatcmpl-06f1b9cd-08bc-43f7-9814-a69173921216",
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "Hi!",
                "role": "assistant",
                "tool_calls": null,
                "function_call": null
            }
        }
    ],
    "created": 1721955063,
    "model": "gpt-3.5-turbo",
    "object": "chat.completion",
    "system_fingerprint": null,
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30
    }
}
```

## Custom Handler Spec

```python
from litellm.types.utils import GenericStreamingChunk, ModelResponse
from typing import Iterator, AsyncIterator
from litellm.llms.base import BaseLLM

class CustomLLMError(Exception):  # use this for all your exceptions
    def __init__(
        self,
        status_code,
        message,
    ):
        self.status_code = status_code
        self.message = message
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs

class CustomLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__()

    def completion(self, *args, **kwargs) -> ModelResponse:
        raise CustomLLMError(status_code=500, message="Not implemented yet!")

    def streaming(self, *args, **kwargs) -> Iterator[GenericStreamingChunk]:
        raise CustomLLMError(status_code=500, message="Not implemented yet!")

    async def acompletion(self, *args, **kwargs) -> ModelResponse:
        raise CustomLLMError(status_code=500, message="Not implemented yet!")

    async def astreaming(self, *args, **kwargs) -> AsyncIterator[GenericStreamingChunk]:
        raise CustomLLMError(status_code=500, message="Not implemented yet!")
```
