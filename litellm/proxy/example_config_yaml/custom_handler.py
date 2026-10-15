from functools import partial
from typing import Final

from litellm import CustomLLM
from litellm.chat_completions.dispatch import ChatResult, forward_completion


class MyCustomLLM(CustomLLM):
    def completion(self, *args, **kwargs) -> ChatResult:
        completion_call: Final[partial[ChatResult]] = forward_completion()
        return completion_call(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello world"}],
            mock_response="Hi!",
        )

    async def acompletion(self, *args, **kwargs) -> ChatResult:
        completion_call: Final[partial[ChatResult]] = forward_completion()
        return completion_call(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello world"}],
            mock_response="Hi!",
        )


my_custom_llm: Final = MyCustomLLM()
