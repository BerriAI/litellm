from collections.abc import Coroutine
from typing import Final, Literal, overload

from litellm.llms.mistral.chat.transformation import MistralConfig
from litellm.llms.mistral.common_utils import with_mistral_tool_call_ids
from litellm.types.llms.openai import AllMessageValues


class VertexAIMistralConfig(MistralConfig):
    @property
    def custom_llm_provider(self) -> str:
        return "vertex_ai"

    @overload
    def _transform_messages(
        self,
        messages: list[AllMessageValues],  # mutable-ok: mirrors override contract
        model: str,
        is_async: Literal[True],
    ) -> Coroutine[object, object, list[AllMessageValues]]: ...  # mutable-ok: mirrors override contract

    @overload
    def _transform_messages(
        self,
        messages: list[AllMessageValues],  # mutable-ok: mirrors override contract
        model: str,
        is_async: Literal[False] = False,
    ) -> list[AllMessageValues]: ...  # mutable-ok: mirrors override contract

    def _transform_messages(
        self,
        messages: list[AllMessageValues],  # mutable-ok: mirrors override contract
        model: str,
        is_async: bool = False,
    ) -> (
        list[AllMessageValues]  # mutable-ok: mirrors override contract
        | Coroutine[object, object, list[AllMessageValues]]
    ):
        mistral_messages: Final = [with_mistral_tool_call_ids(message) for message in messages]
        if is_async:
            return super()._transform_messages(mistral_messages, model, True)
        return super()._transform_messages(mistral_messages, model, False)
