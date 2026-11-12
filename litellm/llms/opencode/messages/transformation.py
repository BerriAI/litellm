from collections.abc import Mapping, Sequence

from pydantic import TypeAdapter

from litellm.litellm_core_utils.url_utils import ensure_api_route
from litellm.llms.anthropic.messages.native.transformation import AnthropicMessagesConfig
from litellm.llms.opencode.common_utils import go_api_base, go_headers


class OpenCodeGoMessagesConfig(AnthropicMessagesConfig):
    @property
    def custom_llm_provider(self) -> str:
        return "opencode_go"

    def should_strip_billing_metadata(self) -> bool:
        return True

    def validate_anthropic_messages_environment(
        self,
        headers: Mapping[str, object],
        model: str,
        messages: Sequence[object],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> tuple[dict[str, object], str | None]:  # mutable-ok: native Messages handler mutates validated headers
        headers_out, base = super().validate_anthropic_messages_environment(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # validate legacy headers before returning them
            headers=go_headers(headers, litellm_params, api_key, model),
            model=model,
            messages=list(messages),
            optional_params=dict(optional_params),
            litellm_params=dict(litellm_params),
            api_base=go_api_base(api_base, model),
        )
        return TypeAdapter(dict[str, object]).validate_python(headers_out), base

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
