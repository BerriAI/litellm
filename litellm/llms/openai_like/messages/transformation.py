from typing import Any, Final

import litellm
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.llms.openai_like.json_loader import SimpleProviderConfig
from litellm.secret_managers.main import get_secret_str

DEFAULT_ANTHROPIC_API_VERSION: Final = "2023-06-01"


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
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> tuple[dict[str, str], str | None]:
        present: Final = {key.lower() for key in headers}
        needs_auth: Final = bool(api_key) and "authorization" not in present and "x-api-key" not in present
        defaults: Final[dict[str, str]] = {
            **({"authorization": f"Bearer {api_key}"} if needs_auth else {}),
            **({"anthropic-version": DEFAULT_ANTHROPIC_API_VERSION} if "anthropic-version" not in present else {}),
            **({"content-type": "application/json"} if "content-type" not in present else {}),
        }
        combined: Final = {**headers, **defaults}
        normalized: Final = {
            ("anthropic-beta" if key.lower() == "anthropic-beta" else key): value for key, value in combined.items()
        }
        merged: Final = self._update_headers_with_anthropic_beta(
            headers=normalized,
            optional_params=optional_params,
        )
        return merged, api_base

    def should_filter_anthropic_beta_headers(self) -> bool:
        return False

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        if not api_base:
            raise ValueError("api_base is required to forward Anthropic /v1/messages to a native endpoint")
        base = api_base.rstrip("/")
        if base.endswith("/v1/messages"):
            return base
        base = base.removesuffix("/v1")
        return f"{base}/v1/messages"


class JSONProviderAnthropicMessagesConfig(OpenAILikeAnthropicMessagesConfig):
    """
    Provider-level native Anthropic Messages passthrough for JSON-configured
    OpenAI-compatible providers whose ``supported_endpoints`` in providers.json
    includes ``"/v1/messages"``. Resolves the api key and api base from the
    provider's configured env vars, then forwards the Anthropic payload
    untranslated like ``OpenAILikeAnthropicMessagesConfig``.
    """

    def __init__(self, provider: SimpleProviderConfig):
        super().__init__()
        self._provider = provider

    @property
    def custom_llm_provider(self) -> str | None:
        return self._provider.slug

    def should_strip_billing_metadata(self) -> bool:
        return True

    def _resolve_api_key(self, api_key: str | None) -> str | None:
        return api_key or get_secret_str(self._provider.api_key_env) or litellm.api_key

    def _resolve_api_base(self, api_base: str | None) -> str:
        env_api_base: Final = get_secret_str(self._provider.api_base_env) if self._provider.api_base_env else None
        return api_base or env_api_base or self._provider.base_url

    def validate_anthropic_messages_environment(
        self,
        headers: dict[str, str],
        model: str,
        messages: list[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> tuple[dict[str, str], str | None]:
        return super().validate_anthropic_messages_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=self._resolve_api_key(api_key),
            api_base=api_base,
        )

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        return super().get_complete_url(
            api_base=self._resolve_api_base(api_base),
            api_key=api_key,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params,
            stream=stream,
        )
