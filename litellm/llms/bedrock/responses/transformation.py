"""Amazon Bedrock Runtime - native OpenAI Responses API.

AWS serves the OpenAI models on ``bedrock-runtime`` through an OpenAI-compatible
surface at ``https://bedrock-runtime.{region}.{dns_suffix}/openai/v1/responses``,
alongside Converse. Without this config the ``bedrock`` provider has no Responses
config at all, so ``/v1/responses`` falls back to the Chat Completions bridge and
the request is translated into Converse, which rejects Responses-only parameters
such as ``prompt_cache_key`` with a 400 and never sees reasoning items.

Payloads and SSE follow the OpenAI Responses spec, so this inherits
OpenAIResponsesAPIConfig and overrides only the endpoint URL, authentication, the
Codex history-item normalization the endpoint requires, and the tool filter below.

Tools: bedrock-runtime runs no server-side tools, so it rejects Codex's default
``web_search`` tool with "web search is not supported for this request". The
Converse bridge dropped that tool silently (Converse has no web search either),
so this config drops every tool type the endpoint rejects the same way. The
supported set is the one bedrock-runtime's own validation error names.

Parity with the Converse bridge on what it used to accept: ``background`` never
reached Converse (the bridge answered synchronously), while bedrock-runtime rejects
it with "The background parameter is not supported.", so it is dropped here. The
bridge also downloaded ``input_image`` http(s) URLs for Converse, while
bedrock-runtime only accepts ``data:`` and ``s3://`` image URLs, so remote image
URLs are fetched and inlined as data URIs before the request is signed.

Auth: Bearer token (litellm_params.api_key or the standard AWS_BEARER_TOKEN_BEDROCK)
when present; otherwise AWS SigV4 (service "bedrock") over the standard credential
chain, signed via BaseAWSLLM._sign_request once the body is final.

Model IDs: bedrock-runtime serves these models only through a cross-Region
inference profile, so the model is named ``us.openai.gpt-5.6-sol`` or
``global.openai.gpt-5.6-sol``; there is no in-Region form.
"""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from types import MappingProxyType
from typing import Final

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.prompt_templates.image_handling import (
    async_convert_url_to_base64,
    convert_url_to_base64,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.responses.codex_compat import drop_unsupported_tools, normalize_codex_input_items
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock.common_utils import (
    BedrockError,
    bedrock_supports_openai_responses,
)
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import ResponseInputParam, ResponsesAPIOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

BEDROCK_RUNTIME_OPENAI_RESPONSES_PATH: Final = "/openai/v1/responses"
BEDROCK_RUNTIME_OPENAI_BASE_SUFFIXES: Final = (
    "/openai/v1/responses",
    "/v1/responses",
    "/responses",
    "/openai/v1",
    "/v1",
)
BEDROCK_RUNTIME_SUPPORTED_RESPONSE_TOOL_TYPES: Final = frozenset(
    {"function", "mcp", "custom", "apply_patch", "namespace", "tool_search", "computer"}
)
BEDROCK_RUNTIME_UNSUPPORTED_RESPONSE_PARAMS: Final = frozenset({"background"})
REMOTE_IMAGE_URL_SCHEMES: Final = ("http://", "https://")
IMAGE_BLOCK_KEYS: Final = ("content", "output")
IMAGE_BLOCK_TYPES: Final = frozenset({"input_image", "computer_screenshot"})


def resolve_bedrock_bearer_token(api_key: str | None) -> str | None:
    return api_key or get_secret_str("AWS_BEARER_TOKEN_BEDROCK")


def _remote_image_url(block: object) -> str | None:
    if not isinstance(block, dict) or block.get("type") not in IMAGE_BLOCK_TYPES:
        return None
    image_url: Final = block.get("image_url")
    if not isinstance(image_url, str) or not image_url.startswith(REMOTE_IMAGE_URL_SCHEMES):
        return None
    return image_url


def _blocks_under(value: object) -> "tuple[object, ...]":
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, dict):
        return (value,)
    return ()


def _image_blocks(item: object) -> "tuple[object, ...]":
    """The blocks of ``item`` that can carry an image: its content and tool output lists, or a screenshot output dict."""
    if not isinstance(item, dict):
        return ()
    return tuple(block for key in IMAGE_BLOCK_KEYS for block in _blocks_under(item.get(key)))


def collect_remote_image_urls(input: "str | ResponseInputParam") -> "tuple[str, ...]":
    """The distinct http(s) image URLs in message content, tool output lists, and computer screenshots, in first-seen order."""
    if not isinstance(input, list):
        return ()
    return tuple(
        dict.fromkeys(
            url for item in input for block in _image_blocks(item) if (url := _remote_image_url(block)) is not None
        )
    )


def _inline_block(block: object, inlined: "Mapping[str, str]") -> object:
    url: Final = _remote_image_url(block)
    if url is None or not isinstance(block, dict):
        return block
    return {**block, "image_url": inlined[url]}  # mutable-ok: outgoing JSON request item


def _inline_value(value: object, inlined: "Mapping[str, str]") -> object:
    if isinstance(value, list):
        return [_inline_block(block, inlined) for block in value]  # mutable-ok: outgoing JSON request item
    return _inline_block(value, inlined)


def _inline_item(item: object, inlined: "Mapping[str, str]") -> object:
    if not isinstance(item, dict):
        return item
    inlined_fields: Final = {  # mutable-ok: outgoing JSON request item
        key: _inline_value(item[key], inlined) for key in IMAGE_BLOCK_KEYS if isinstance(item.get(key), (list, dict))
    }
    if not inlined_fields:
        return item
    return {**item, **inlined_fields}  # mutable-ok: same


def inline_remote_image_urls(
    input: "str | ResponseInputParam", inlined: "Mapping[str, str]"
) -> "str | ResponseInputParam":
    """``input`` with every http(s) image URL replaced by its entry in ``inlined``."""
    if not isinstance(input, list) or not inlined:
        return input
    items: Final = [_inline_item(item, inlined) for item in input]  # mutable-ok: downstream narrows on isinstance(list)
    return items  # pyright: ignore[reportReturnType]  # items keep the caller's input union


class BedrockOpenAIResponsesConfig(BaseAWSLLM, OpenAIResponsesAPIConfig):
    """Responses API config for the OpenAI models on the bedrock-runtime endpoint."""

    def __init__(
        self,
        fetch_image: "Callable[[str], str]" = convert_url_to_base64,
        async_fetch_image: "Callable[[str], Awaitable[str]]" = async_convert_url_to_base64,
    ) -> None:
        super().__init__()
        self.fetch_image = fetch_image
        self.async_fetch_image = async_fetch_image

    @classmethod
    def for_model(cls, model: str | None) -> "BedrockOpenAIResponsesConfig | None":
        """This config when ``model`` is served on the OpenAI Responses surface, else ``None``.

        The capability decision lives here rather than in the shared dispatch so that
        onboarding a model, or changing how the signal is read, stays inside the
        Bedrock adapter. ``None`` leaves the caller's existing behaviour untouched --
        chat-only Bedrock models keep the Chat Completions bridge.
        """
        if not bedrock_supports_openai_responses(model, litellm.model_cost):
            return None
        return cls()

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.BEDROCK

    def get_error_class(
        self, error_message: str, status_code: int, headers: dict[str, object] | httpx.Headers
    ) -> BaseLLMException:
        # The OpenAI base builds a blank response, dropping x-amzn-RequestId.
        return BedrockError(status_code=status_code, message=error_message, headers=headers)

    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: dict,  # mutable-ok: signature fixed by the BaseResponsesAPIConfig override contract
    ) -> str:
        region: Final = self._get_aws_region_name(optional_params=litellm_params, model=None)
        override: Final = (
            api_base
            or litellm_params.get("aws_bedrock_runtime_endpoint")
            or get_secret_str("AWS_BEDROCK_RUNTIME_ENDPOINT")
        )
        # Partition-aware: bedrock-runtime is amazonaws.com.cn in China, and other
        # suffixes in GovCloud/ISO, so defer to the shared endpoint builder.
        host: Final = (
            override or self._select_default_endpoint_url(endpoint_type="runtime", aws_region_name=region)
        ).rstrip("/")
        base: Final = next(
            (host[: -len(suffix)] for suffix in BEDROCK_RUNTIME_OPENAI_BASE_SUFFIXES if host.endswith(suffix)),
            host,
        )
        return f"{base}{BEDROCK_RUNTIME_OPENAI_RESPONSES_PATH}"

    def supports_native_file_search(self) -> bool:
        return False

    def validate_environment(
        self,
        headers: dict,  # mutable-ok: signature fixed by the BaseResponsesAPIConfig override contract
        model: str,
        litellm_params: GenericLiteLLMParams | None,
    ) -> dict:  # mutable-ok: signature fixed by the BaseResponsesAPIConfig override contract
        api_key: Final = litellm_params.api_key if litellm_params is not None else None
        bearer: Final = resolve_bedrock_bearer_token(api_key)
        if not bearer:
            return headers
        return {**headers, "Authorization": f"Bearer {bearer}"}  # mutable-ok: dict return per the contract

    def sign_request(
        self,
        headers: dict,  # mutable-ok: signature fixed by the BaseResponsesAPIConfig override contract
        optional_params: dict,  # mutable-ok: same
        request_data: dict,  # mutable-ok: same
        api_base: str,
        api_key: str | None = None,
        model: str | None = None,
        stream: bool | None = None,
        fake_stream: bool | None = None,
    ) -> "tuple[dict, bytes | None]":  # mutable-ok: signature fixed by the override contract
        if resolve_bedrock_bearer_token(api_key):
            # Bedrock API keys are Bearer credentials; SigV4 on top would be wrong.
            return headers, None
        return self._sign_request(
            service_name="bedrock",
            headers=headers,
            optional_params=optional_params,
            request_data=request_data,
            api_base=api_base,
            model=model,
            stream=stream,
            fake_stream=fake_stream,
        )

    def map_openai_params(
        self,
        response_api_optional_params: ResponsesAPIOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:  # mutable-ok: signature fixed by the override contract
        mapped: Final = super().map_openai_params(
            response_api_optional_params=response_api_optional_params, model=model, drop_params=drop_params
        )
        unsupported: Final = tuple(sorted(BEDROCK_RUNTIME_UNSUPPORTED_RESPONSE_PARAMS & mapped.keys()))
        if unsupported:
            verbose_logger.warning(
                "Bedrock Runtime Responses API: dropping unsupported parameter(s) %s that the endpoint rejects.",
                unsupported,
            )
        params: Final = {  # mutable-ok: outgoing JSON request params
            key: value for key, value in mapped.items() if key not in unsupported
        }
        tools: Final = params.get("tools")
        if not isinstance(tools, list):
            return params
        kept, dropped_types = drop_unsupported_tools(tools, BEDROCK_RUNTIME_SUPPORTED_RESPONSE_TOOL_TYPES)
        if not dropped_types:
            return params
        verbose_logger.warning(
            "Bedrock Runtime Responses API: dropping unsupported tool type(s) %s (supported: %s).",
            list(dropped_types),
            sorted(BEDROCK_RUNTIME_SUPPORTED_RESPONSE_TOOL_TYPES),
        )
        without_tools: Final = {key: value for key, value in params.items() if key != "tools"}
        if not kept:
            return without_tools
        return {**without_tools, "tools": list(kept)}

    def transform_responses_api_request(
        self,
        model: str,
        input: "str | ResponseInputParam",
        response_api_optional_request_params: dict,  # mutable-ok: signature fixed by the override contract
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: same
    ) -> dict:  # mutable-ok: same
        inlined: Final = MappingProxyType({url: self.fetch_image(url) for url in collect_remote_image_urls(input)})
        return self._transform_inlined_request(
            model=model,
            input=inline_remote_image_urls(input, inlined),
            response_api_optional_request_params=response_api_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    async def async_transform_responses_api_request(
        self,
        model: str,
        input: "str | ResponseInputParam",
        response_api_optional_request_params: dict,  # mutable-ok: signature fixed by the override contract
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: same
    ) -> dict:  # mutable-ok: same
        remote_urls: Final = collect_remote_image_urls(input)
        data_uris: Final = await asyncio.gather(*(self.async_fetch_image(url) for url in remote_urls))
        return self._transform_inlined_request(
            model=model,
            input=inline_remote_image_urls(input, MappingProxyType(dict(zip(remote_urls, data_uris, strict=True)))),
            response_api_optional_request_params=response_api_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    def _transform_inlined_request(
        self,
        model: str,
        input: "str | ResponseInputParam",
        response_api_optional_request_params: dict,  # mutable-ok: signature fixed by the override contract
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: same
    ) -> dict:  # mutable-ok: same
        normalized_input, rewritten_types = normalize_codex_input_items(input)
        if rewritten_types:
            verbose_logger.warning(
                "Bedrock Runtime Responses API: rewrote Codex input item type(s) %s that the endpoint rejects.",
                rewritten_types,
            )
        return super().transform_responses_api_request(
            model=model,
            input=normalized_input,
            response_api_optional_request_params=response_api_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )
