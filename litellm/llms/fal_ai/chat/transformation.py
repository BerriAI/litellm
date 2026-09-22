"""
Support for `/v1/chat/completions` on Fal AI model endpoints, e.g. fal-ai/moondream3-preview/query.

These endpoints are not OpenAI-compatible: the request body is a flat ``{"prompt", "image_url"}``
object and the response is ``{"output", "reasoning", "finish_reason", "usage_info"}``.
"""

import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter

from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Message, ModelResponse, Usage

if TYPE_CHECKING:
    import tiktoken

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

DEFAULT_BASE_URL: Final[str] = "https://fal.run"
PROVIDER_PREFIX: Final[str] = "fal_ai/"
PASSTHROUGH_PARAMS: Final[frozenset[str]] = frozenset(("reasoning", "temperature", "top_p"))
REASONING_DISABLED_EFFORTS: Final[frozenset[str]] = frozenset(("none", "minimal"))
REASONING_ENABLED_EFFORTS: Final[frozenset[str]] = frozenset(("low", "medium", "high"))


class _FalUsage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    input_tokens: int
    output_tokens: int


class _FalChatResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    output: str
    usage_info: _FalUsage
    reasoning: str | None = None
    finish_reason: str | None = None


_CHAT_RESPONSE: Final = TypeAdapter(_FalChatResponse)


class FalAIError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: dict | httpx.Headers | None = None,  # mutable-ok: BaseLLMException header contract
    ) -> None:
        super().__init__(status_code=status_code, message=message, headers=headers)


def _image_part_url(part: Mapping[str, object]) -> str | None:
    image_url: Final = part.get("image_url")
    if isinstance(image_url, str):
        return image_url
    if isinstance(image_url, Mapping):
        url: Final = image_url.get("url")
        return url if isinstance(url, str) else None
    return None


def _prompt_and_image(messages: Sequence[AllMessageValues]) -> tuple[str, str]:
    if len(messages) != 1 or messages[0].get("role") != "user":
        raise FalAIError(
            status_code=400,
            message="fal_ai chat completions accept exactly one user message; system prompts and multi-turn history are not supported",
        )
    content: Final = messages[0].get("content")
    if isinstance(content, str):
        if not content:
            raise FalAIError(status_code=400, message="fal_ai chat completions require text in the user message")
        raise FalAIError(
            status_code=400,
            message="fal_ai chat completions require exactly one image_url content part in the user message",
        )
    parts: Final[tuple[Mapping[str, object], ...]] = (
        tuple(part for part in content if isinstance(part, Mapping)) if isinstance(content, Sequence) else ()
    )
    prompt: Final = "\n".join(
        text for part in parts if part.get("type") == "text" and isinstance((text := part.get("text")), str) and text
    )
    image_urls: Final = tuple(
        url for part in parts if part.get("type") == "image_url" and (url := _image_part_url(part)) is not None
    )
    if not prompt:
        raise FalAIError(status_code=400, message="fal_ai chat completions require text in the user message")
    if len(image_urls) != 1:
        raise FalAIError(
            status_code=400,
            message="fal_ai chat completions require exactly one image_url content part in the user message",
        )
    return prompt, image_urls[0]


class FalAIChatConfig(BaseConfig):
    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return api_key or get_secret_str("FAL_AI_API_KEY")

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str:
        return (api_base or get_secret_str("FAL_AI_API_BASE") or DEFAULT_BASE_URL).rstrip("/")

    def get_supported_openai_params(self, model: str) -> list:  # mutable-ok: inherited contract returns a list
        return list(("reasoning_effort", "temperature", "top_p"))  # mutable-ok: inherited contract returns a list

    def _map_reasoning_effort(self, value: object, model: str, drop_params: bool) -> bool | None:
        if isinstance(value, str) and value in REASONING_DISABLED_EFFORTS:
            return False
        if isinstance(value, str) and value in REASONING_ENABLED_EFFORTS:
            return True
        if drop_params:
            return None
        raise FalAIError(status_code=400, message=f"Unsupported reasoning_effort {value!r} for {model}")

    def _translate_param(self, param: str, value: object, model: str, drop_params: bool) -> tuple[str, object] | None:
        if param in ("temperature", "top_p"):
            return param, value
        if param == "reasoning_effort":
            reasoning: Final = self._map_reasoning_effort(value, model, drop_params)
            return ("reasoning", reasoning) if reasoning is not None else None
        return None

    def map_openai_params(
        self,
        non_default_params: dict,  # mutable-ok: inherited contract
        optional_params: dict,  # mutable-ok: inherited contract
        model: str,
        drop_params: bool,
    ) -> dict:  # mutable-ok: inherited contract returns a dict
        mapped: Final = {  # mutable-ok: intermediate translation map, folded into the returned dict
            translated[0]: translated[1]
            for param, value in non_default_params.items()
            if (translated := self._translate_param(param, value, model, drop_params)) is not None
        }
        return {**optional_params, **mapped}  # mutable-ok: inherited contract returns a dict

    def validate_environment(
        self,
        headers: dict,  # mutable-ok: inherited contract
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: inherited contract
        optional_params: dict,  # mutable-ok: inherited contract
        litellm_params: dict,  # mutable-ok: inherited contract
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:  # mutable-ok: inherited contract returns a dict
        final_api_key: Final = self.get_api_key(api_key)
        if not final_api_key:
            raise ValueError("FAL_AI_API_KEY is not set")
        return {  # mutable-ok: inherited contract returns a dict
            "content-type": "application/json",
            **(headers or {}),  # mutable-ok: empty default for the inherited contract's headers
            "Authorization": f"Key {final_api_key}",
        }

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,  # mutable-ok: inherited contract
        litellm_params: dict,  # mutable-ok: inherited contract
        stream: bool | None = None,
    ) -> str:
        return f"{self.get_api_base(api_base)}/{model.removeprefix(PROVIDER_PREFIX)}"

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: inherited contract
        optional_params: dict,  # mutable-ok: inherited contract
        litellm_params: dict,  # mutable-ok: inherited contract
        headers: dict,  # mutable-ok: inherited contract
    ) -> dict:  # mutable-ok: inherited contract returns a dict
        if optional_params.get("stream"):
            raise FalAIError(status_code=400, message="fal_ai chat completions do not support streaming")
        prompt, image_url = _prompt_and_image(messages)
        return {  # mutable-ok: JSON request body
            "prompt": prompt,
            "image_url": image_url,
            **{  # mutable-ok: JSON request body
                key: value for key, value in optional_params.items() if key in PASSTHROUGH_PARAMS and value is not None
            },
        }

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: "LiteLLMLoggingObj",
        request_data: dict,  # mutable-ok: inherited contract
        messages: list[AllMessageValues],  # mutable-ok: inherited contract
        optional_params: dict,  # mutable-ok: inherited contract
        litellm_params: dict,  # mutable-ok: inherited contract
        encoding: "tiktoken.Encoding | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        try:
            completion_response: Final = _CHAT_RESPONSE.validate_json(raw_response.content)
        except ValueError:
            raise FalAIError(
                status_code=422,
                message=f"fal_ai returned an unexpected response body: {raw_response.text}",
                headers=raw_response.headers,
            )

        message: Final = Message(
            content=completion_response.output,
            role="assistant",
            reasoning_content=completion_response.reasoning,
        )
        model_response.choices[0].message = message  # rebind-ok: ModelResponse populated in place per contract
        model_response.choices[0].finish_reason = map_finish_reason(  # rebind-ok: same contract
            completion_response.finish_reason or "stop"
        )
        model_response.created = int(time.time())  # rebind-ok: same contract
        model_response.model = model  # rebind-ok: same contract
        model_response.usage = Usage(  # rebind-ok: same contract
            prompt_tokens=completion_response.usage_info.input_tokens,
            completion_tokens=completion_response.usage_info.output_tokens,
            total_tokens=completion_response.usage_info.input_tokens + completion_response.usage_info.output_tokens,
        )
        return model_response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict | httpx.Headers,  # mutable-ok: inherited contract
    ) -> BaseLLMException:
        return FalAIError(status_code=status_code, message=error_message, headers=headers)
