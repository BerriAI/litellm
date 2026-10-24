from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter

import litellm
from litellm.litellm_core_utils.url_utils import ensure_api_route
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.llms.gemini.chat.transformation import GoogleAIStudioGeminiConfig
from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig
from litellm.llms.opencode.common_utils import go_api_base, go_api_key, go_endpoint, go_headers, go_model_info
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging


class OpenCodeGoChatConfig(OpenAILikeChatConfig):
    @property
    def custom_llm_provider(self) -> str:
        return "opencode_go"

    def _get_openai_compatible_provider_info(
        self,
        api_base: str | None,
        api_key: str | None,
    ) -> tuple[str, str | None]:
        return go_api_base(api_base), go_api_key(api_key)

    def validate_environment(
        self,
        headers: Mapping[str, object],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, object]:  # mutable-ok: protocol handlers mutate the returned request and headers
        return go_headers(headers, litellm_params, api_key, model)

    def get_models(
        self, api_key: str | None = None, api_base: str | None = None
    ) -> list[str]:  # mutable-ok: provider discovery requires a list
        response: Final = litellm.module_level_client.get(  # pyright: ignore[reportUnknownMemberType]  # legacy HTTP wrapper has untyped query parameters
            url=ensure_api_route(
                (api_base or get_secret_str("OPENCODE_GO_API_BASE") or "https://opencode.ai/inference/go")
                .rstrip("/")
                .removesuffix("/v1"),
                "v1/models",
            ),
            headers={"Authorization": f"Bearer {go_api_key(api_key)}"} if go_api_key(api_key) else {},
        )
        response.raise_for_status()
        data: Final = TypeAdapter(dict[str, object]).validate_json(response.content)
        models: Final = TypeAdapter(tuple[Mapping[str, object], ...]).validate_python(data["data"])
        return [f"opencode_go/{entry['id']}" for entry in models if isinstance(entry.get("id"), str)]


class OpenCodeGoMessagesChatConfig(AnthropicConfig):
    @property
    def custom_llm_provider(self) -> str:
        return "opencode_go"

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        return ensure_api_route(go_api_base(api_base, model).rstrip("/").removesuffix("/v1"), "v1/messages")

    def validate_environment(
        self,
        headers: Mapping[str, object],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, object]:  # mutable-ok: protocol handlers mutate the returned request and headers
        return go_headers({"anthropic-version": "2023-06-01", **headers}, litellm_params, api_key, model)

    def transform_request(
        self,
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        headers: Mapping[str, object],
    ) -> dict[str, object]:  # mutable-ok: protocol handlers mutate the returned request and headers
        max_output: Final[int | None] = TypeAdapter[int | None](int | None).validate_python(
            go_model_info(model).get("max_output_tokens")
        )
        defaults: Final = {"max_tokens": max_output} if max_output else {}
        raw: Final[object] = super().transform_request(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # legacy Anthropic transformer accepts untyped mappings
            model,
            list(messages),
            {**defaults, **optional_params},
            dict(litellm_params),
            dict(headers),
        )
        return TypeAdapter(dict[str, object]).validate_python(raw)


class OpenCodeGoGeminiChatConfig(GoogleAIStudioGeminiConfig):
    @property
    def supports_stream_param_in_request_body(self) -> bool:
        return False

    def get_model_response_iterator_for_request(
        self,
        streaming_response: Iterator[str] | AsyncIterator[str] | ModelResponse,
        sync_stream: bool,
        json_mode: bool | None,
        optional_params: Mapping[str, object],
        logging_obj: "Logging | None" = None,
    ) -> object:
        from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import ModelResponseIterator

        if logging_obj is None:
            raise ValueError("OpenCode Go Gemini streaming requires the request logging context")
        iterator: Final = ModelResponseIterator(
            streaming_response=streaming_response, sync_stream=sync_stream, logging_obj=logging_obj
        )
        return iter(iterator) if sync_stream else iterator

    @property
    def custom_llm_provider(self) -> str:
        return "opencode_go"

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        base: Final = ensure_api_route(
            go_api_base(api_base, model).rstrip("/").removesuffix("/v1beta"), "v1beta/models"
        )
        action: Final = "streamGenerateContent?alt=sse" if stream else "generateContent"
        return f"{base}/{model}:{action}"

    def validate_environment(
        self,
        headers: Mapping[str, object] | None,
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | Mapping[str, object] | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: Gemini handler requires mutable string headers
        return TypeAdapter(dict[str, str]).validate_python(
            go_headers(headers or {}, litellm_params, api_key if isinstance(api_key, str) else None, model)
        )

    def transform_request(
        self,
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        headers: Mapping[str, object],
    ) -> dict[str, object]:  # mutable-ok: protocol handlers mutate the returned request and headers
        from litellm.llms.vertex_ai.gemini.transformation import (
            _transform_request_body,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # shared legacy Gemini wire-format builder
        )

        cached_content: Final = optional_params.get("cached_content")
        body: Final = _transform_request_body(
            messages=list(messages),
            model=model,
            optional_params=dict(optional_params),
            custom_llm_provider="gemini",
            litellm_params=dict(litellm_params),
            cached_content=cached_content if isinstance(cached_content, str) else None,
        )
        return TypeAdapter(dict[str, object]).validate_python(body)


def go_chat_config(model: str) -> BaseConfig:
    match go_endpoint(model):
        case "/v1/messages":
            return OpenCodeGoMessagesChatConfig()
        case "/v1beta/models":
            return OpenCodeGoGeminiChatConfig()
        case _:
            return OpenCodeGoChatConfig()
