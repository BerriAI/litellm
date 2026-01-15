"""
Translate from OpenAI's `/v1/chat/completions` to VLLM's `/v1/chat/completions`
"""

from typing import Any, Coroutine, List, Literal, Optional, Tuple, Union, cast, overload

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    _get_image_mime_type_from_url,
)
from litellm.litellm_core_utils.prompt_templates.factory import _parse_mime_type
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionFileObject,
    ChatCompletionVideoObject,
    ChatCompletionVideoUrlObject,
)

from ....utils import _remove_additional_properties, _remove_strict_from_schema
from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class HostedVLLMChatConfig(OpenAIGPTConfig):
    def get_supported_openai_params(self, model: str) -> List[str]:
        params = super().get_supported_openai_params(model)
        params.append("reasoning_effort")
        return params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        _tools = non_default_params.pop("tools", None)
        if _tools is not None:
            # remove 'additionalProperties' from tools
            _tools = _remove_additional_properties(_tools)
            # remove 'strict' from tools
            _tools = _remove_strict_from_schema(_tools)
        if _tools is not None:
            non_default_params["tools"] = _tools
        return super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("HOSTED_VLLM_API_BASE")  # type: ignore
        dynamic_api_key = (
            api_key or get_secret_str("HOSTED_VLLM_API_KEY") or "fake-api-key"
        )  # vllm does not require an api key
        return api_base, dynamic_api_key

    def _is_video_file(self, content_item: ChatCompletionFileObject) -> bool:
        """
        Check if the file is a video

        - format: video/<extension>
        - file_data: base64 encoded video data
        - file_id: infer mp4 from extension
        """
        file = content_item.get("file", {})
        format = file.get("format")
        file_data = file.get("file_data")
        file_id = file.get("file_id")
        if content_item.get("type") != "file":
            return False
        if format and format.startswith("video/"):
            return True
        elif file_data:
            mime_type = _parse_mime_type(file_data)
            if mime_type and mime_type.startswith("video/"):
                return True
        elif file_id:
            mime_type = _get_image_mime_type_from_url(file_id)
            if mime_type and mime_type.startswith("video/"):
                return True
        return False

    def _convert_file_to_video_url(
        self, content_item: ChatCompletionFileObject
    ) -> ChatCompletionVideoObject:
        file = content_item.get("file", {})
        file_id = file.get("file_id")
        file_data = file.get("file_data")

        if file_id:
            return ChatCompletionVideoObject(
                type="video_url", video_url=ChatCompletionVideoUrlObject(url=file_id)
            )
        elif file_data:
            return ChatCompletionVideoObject(
                type="video_url", video_url=ChatCompletionVideoUrlObject(url=file_data)
            )
        raise ValueError("file_id or file_data is required")

    @overload
    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: Literal[True]
    ) -> Coroutine[Any, Any, List[AllMessageValues]]:
        ...

    @overload
    def _transform_messages(
        self,
        messages: List[AllMessageValues],
        model: str,
        is_async: Literal[False] = False,
    ) -> List[AllMessageValues]:
        ...

    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: bool = False
    ) -> Union[List[AllMessageValues], Coroutine[Any, Any, List[AllMessageValues]]]:
        """
        Support translating video files from file_id or file_data to video_url
        """
        for message in messages:
            if message["role"] == "user":
                message_content = message.get("content")
                if message_content and isinstance(message_content, list):
                    replaced_content_items: List[
                        Tuple[int, ChatCompletionFileObject]
                    ] = []
                    for idx, content_item in enumerate(message_content):
                        if content_item.get("type") == "file":
                            content_item = cast(ChatCompletionFileObject, content_item)
                            if self._is_video_file(content_item):
                                replaced_content_items.append((idx, content_item))
                    for idx, content_item in replaced_content_items:
                        message_content[idx] = self._convert_file_to_video_url(
                            content_item
                        )
        if is_async:
            return super()._transform_messages(
                messages, model, is_async=cast(Literal[True], True)
            )
        else:
            return super()._transform_messages(
                messages, model, is_async=cast(Literal[False], False)
            )
