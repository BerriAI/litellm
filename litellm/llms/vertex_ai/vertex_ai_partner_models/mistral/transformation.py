from collections.abc import Coroutine, Mapping
from typing import Final, Literal, overload

from litellm.litellm_core_utils.core_helpers import as_object_list
from litellm.llms.mistral.chat.transformation import MistralConfig
from litellm.llms.mistral.common_utils import with_mistral_tool_call_ids
from litellm.types.llms.openai import AllMessageValues


def _part_text(part: object) -> str | None:
    match part:
        case {"type": "text", "text": str(text)}:
            return text
        case _:
            return None


def _several_part_texts(message: Mapping[str, object]) -> tuple[str, ...] | None:
    parts: Final = as_object_list(message.get("content"))
    if parts is None or len(parts) < 2:
        return None
    texts: Final = tuple(text for part in parts if (text := _part_text(part)) is not None)
    return texts if len(texts) == len(parts) else None


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

    @staticmethod
    def _with_flattened_content(message: AllMessageValues, keeps_reasoning: bool) -> AllMessageValues:
        texts: Final = _several_part_texts(message)
        if texts is None:
            return MistralConfig._with_flattened_content(message, keeps_reasoning)
        return MistralConfig._with_text_parts(message, texts)
