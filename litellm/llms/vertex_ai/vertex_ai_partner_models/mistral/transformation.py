from collections.abc import Coroutine, Mapping, Sequence
from typing import Final, Literal, overload

from litellm.litellm_core_utils.core_helpers import as_object_list
from litellm.llms.mistral.chat.transformation import MistralConfig
from litellm.llms.mistral.common_utils import with_mistral_tool_call_ids
from litellm.types.llms.openai import AllMessageValues, ChatCompletionTextObject

_PARAMS_VERTEX_AI_ADDS: Final = ("presence_penalty", "frequency_penalty", "n")


def _part_text(part: object) -> str | None:
    match part:
        case {"type": "text", "text": str(text)}:
            return text
        case _:
            return None


def _bare_text_part(part: object) -> object:
    text: Final = _part_text(part)
    return part if text is None else ChatCompletionTextObject(type="text", text=text)


def _content_parts(message: Mapping[str, object]) -> Sequence[object] | None:
    return as_object_list(message.get("content"))


def _has_several_text_parts(message: Mapping[str, object]) -> bool:
    parts: Final = _content_parts(message)
    return parts is not None and len(parts) >= 2 and all(_part_text(part) is not None for part in parts)


class VertexAIMistralConfig(MistralConfig):
    @property
    def custom_llm_provider(self) -> str:
        return "vertex_ai"

    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: mirrors override contract
        return [*super().get_supported_openai_params(model), *_PARAMS_VERTEX_AI_ADDS]

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
        mistral_messages: Final = [
            self._with_bare_text_parts(with_mistral_tool_call_ids(message)) for message in messages
        ]
        if is_async:
            return super()._transform_messages(mistral_messages, model, True)
        return super()._transform_messages(mistral_messages, model, False)

    @staticmethod
    def _with_bare_text_parts(message: AllMessageValues) -> AllMessageValues:
        parts: Final = _content_parts(message)
        if parts is None:
            return message
        return MistralConfig._with_content_parts(message, tuple(map(_bare_text_part, parts)))

    @staticmethod
    def _with_flattened_content(message: AllMessageValues) -> AllMessageValues:
        if _has_several_text_parts(message):
            return message
        return MistralConfig._with_flattened_content(message)
