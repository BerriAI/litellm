from typing import Any, Optional

from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)

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
        combined = {**headers, **defaults}
        normalized = {
            ("anthropic-beta" if key.lower() == "anthropic-beta" else key): value for key, value in combined.items()
        }
        merged = self._update_headers_with_anthropic_beta(
            headers=normalized,
            optional_params=optional_params,
        )
        return merged, api_base

    def should_filter_anthropic_beta_headers(self) -> bool:
        return False

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
