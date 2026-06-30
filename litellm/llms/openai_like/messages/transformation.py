from typing import Any, Optional

from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.types.router import GenericLiteLLMParams

DEFAULT_ANTHROPIC_API_VERSION = "2023-06-01"


class OpenAILikeAnthropicMessagesConfig(AnthropicMessagesConfig):
    """
    Forwards Anthropic /v1/messages requests to an OpenAI-compatible server that
    also natively exposes the Anthropic Messages API, with no translation.

    Opted into per deployment via ``model_info.supported_endpoints`` containing
    ``"/v1/messages"``. The inbound Anthropic payload (system, cache_control,
    thinking, tools, ...) is forwarded essentially unchanged to
    ``{api_base}/v1/messages``, so Anthropic-only features that the
    Anthropic->OpenAI translation would otherwise drop are preserved. Response
    parsing and streaming are inherited from the native Anthropic config.
    """

    def validate_anthropic_messages_environment(
        self,
        headers: dict[str, str],
        model: str,
        messages: list[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> tuple[dict[str, str], Optional[str]]:
        present = {key.lower() for key in headers}
        needs_auth = bool(api_key) and "authorization" not in present and "x-api-key" not in present
        defaults: dict[str, str] = {
            **({"authorization": f"Bearer {api_key}"} if needs_auth else {}),
            **({"anthropic-version": DEFAULT_ANTHROPIC_API_VERSION} if "anthropic-version" not in present else {}),
            **({"content-type": "application/json"} if "content-type" not in present else {}),
        }
        return {**headers, **defaults}, api_base

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        if not api_base:
            raise ValueError("api_base is required to forward Anthropic /v1/messages to a native endpoint")
        base = api_base.rstrip("/")
        if base.endswith("/v1/messages"):
            return base
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        return f"{base}/v1/messages"

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: list[dict],
        anthropic_messages_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> dict:
        if anthropic_messages_optional_request_params.get("max_tokens") is None:
            raise ValueError("max_tokens is required for the Anthropic /v1/messages API")
        return {
            "model": model,
            "messages": messages,
            **anthropic_messages_optional_request_params,
        }
