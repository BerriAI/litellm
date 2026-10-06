"""
Handles transforming requests for `bedrock/invoke/{nova} models`

Inherits from `AmazonConverseConfig`

Nova + Invoke API Tutorial: https://docs.aws.amazon.com/nova/latest/userguide/using-invoke-api.html
"""

from collections.abc import Callable, Mapping, Sequence
from functools import reduce
from typing import TYPE_CHECKING, Final, TypeVar

import httpx
from pydantic import TypeAdapter, ValidationError

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.types.llms.bedrock import (
    BedrockInvokeNovaRequest,
    CachePointBlock,
    ContentBlock,
    MessageBlock,
    SystemContentBlock,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

from ..converse_transformation import AmazonConverseConfig
from .base_invoke_transformation import AmazonInvokeConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer

_CachePointCarrier = TypeVar("_CachePointCarrier", SystemContentBlock, ContentBlock)
_INJECTION_POINTS: Final = TypeAdapter(tuple[Mapping[str, object], ...])


def _without_tool_config_injection_points(optional_params: Mapping[str, object]) -> dict[str, object]:
    """InvokeModel has no tool caching, and a ``tool_config`` point the Converse transform
    placed would credit the gateway for a cachePoint this request cannot carry.
    """
    raw_points: Final = optional_params.get("cache_control_injection_points")
    if raw_points is None:
        return dict(optional_params)
    try:
        points = _INJECTION_POINTS.validate_python(raw_points)
    except ValidationError:
        return dict(optional_params)
    return {
        **optional_params,
        "cache_control_injection_points": [point for point in points if point.get("location") != "tool_config"],
    }


def _system_block_with_cache_point(block: SystemContentBlock, cache_point: CachePointBlock) -> SystemContentBlock:
    return {**block, "cachePoint": cache_point}


def _content_block_with_cache_point(block: ContentBlock, cache_point: CachePointBlock) -> ContentBlock:
    return {**block, "cachePoint": cache_point}


def _inline_block_cache_points(
    blocks: Sequence[_CachePointCarrier],
    with_cache_point: Callable[[_CachePointCarrier, CachePointBlock], _CachePointCarrier],
) -> list[_CachePointCarrier]:
    def attach(inlined: tuple[_CachePointCarrier, ...], block: _CachePointCarrier) -> tuple[_CachePointCarrier, ...]:
        cache_point: Final = block.get("cachePoint")
        if cache_point is None or len(block) != 1:
            return (*inlined, block)
        anchor: Final = next((index for index in reversed(range(len(inlined))) if "text" in inlined[index]), None)
        if anchor is None:
            return inlined
        return (*inlined[:anchor], with_cache_point(inlined[anchor], cache_point), *inlined[anchor + 1 :])

    return list(reduce(attach, blocks, ()))


class AmazonInvokeNovaConfig(AmazonInvokeConfig, AmazonConverseConfig):
    """
    Config for sending `nova` requests to `/bedrock/invoke/`
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_supported_openai_params(self, model: str) -> list:
        return AmazonConverseConfig.get_supported_openai_params(self, model)

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        return AmazonConverseConfig.map_openai_params(self, non_default_params, optional_params, model, drop_params)

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict[str, object],
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        _transformed_nova_request: Final = AmazonConverseConfig.transform_request(
            self,
            model=model,
            messages=messages,
            optional_params=_without_tool_config_injection_points(optional_params),
            litellm_params=litellm_params,
            headers=headers,
        )
        _bedrock_invoke_nova_request: Final = self._inline_cache_points(
            BedrockInvokeNovaRequest(**_transformed_nova_request)
        )
        self._remove_empty_system_messages(_bedrock_invoke_nova_request)
        bedrock_invoke_nova_request: Final = self._filter_allowed_fields(_bedrock_invoke_nova_request)
        return bedrock_invoke_nova_request

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: Logging,
        request_data: dict,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        return AmazonConverseConfig.transform_response(
            self,
            model,
            raw_response,
            model_response,
            logging_obj,
            request_data,
            messages,
            optional_params,
            litellm_params,
            encoding,
            api_key,
            json_mode,
        )

    @staticmethod
    def _inline_cache_points(request: BedrockInvokeNovaRequest) -> BedrockInvokeNovaRequest:
        """InvokeModel takes ``cachePoint`` as a key of the text block it caches: it rejects the
        standalone ``{"cachePoint": ...}`` blocks Converse accepts and the key on image, toolUse,
        and toolResult blocks, so a point behind one of those moves back to the last text block.
        """
        return {
            **request,
            "system": _inline_block_cache_points(request.get("system", []), _system_block_with_cache_point),
            "messages": [
                MessageBlock(
                    role=message["role"],
                    content=_inline_block_cache_points(message["content"], _content_block_with_cache_point),
                )
                for message in request.get("messages", [])
            ],
        }

    def _filter_allowed_fields(self, bedrock_invoke_nova_request: BedrockInvokeNovaRequest) -> dict:
        """
        Filter out fields that are not allowed in the `BedrockInvokeNovaRequest` dataclass.
        """
        allowed_fields: Final = set(BedrockInvokeNovaRequest.__annotations__.keys())
        return {k: v for k, v in bedrock_invoke_nova_request.items() if k in allowed_fields}

    def _remove_empty_system_messages(self, bedrock_invoke_nova_request: BedrockInvokeNovaRequest) -> None:
        """
        In-place remove empty `system` messages from the request.

        /bedrock/invoke/ does not allow empty `system` messages.
        """
        _system_message: Final = bedrock_invoke_nova_request.get("system", None)
        if isinstance(_system_message, list) and len(_system_message) == 0:
            bedrock_invoke_nova_request.pop("system", None)
