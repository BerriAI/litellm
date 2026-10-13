from typing import Final

from litellm.llms.anthropic.pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.llms.databricks.common_utils import DATABRICKS_ANTHROPIC_MESSAGES_PATH, DatabricksBase
from litellm.types.router import GenericLiteLLMParams


class DatabricksAnthropicMessagesConfig(DatabricksBase, AnthropicMessagesConfig):
    """
    Forwards /v1/messages for Claude serving endpoints to the workspace's native
    ``/serving-endpoints/anthropic/v1/messages`` route, so Anthropic-only request
    fields and response fields (``safeguards`` and ``safeguard_results``) survive
    instead of being bridged through chat completions.
    """

    @property
    def custom_llm_provider(self) -> str | None:
        return "databricks"

    def validate_anthropic_messages_environment(
        self,
        headers: dict[str, str],  # mutable-ok: mirrors the base signature
        model: str,
        messages: list[object],  # mutable-ok: mirrors the base signature
        optional_params: dict[str, object],  # mutable-ok: mirrors the base signature
        litellm_params: dict[str, object],  # mutable-ok: mirrors the base signature
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> tuple[dict[str, str], str | None]:  # mutable-ok: mirrors the base signature
        messages_api_base, auth_headers = self.databricks_validate_environment(  # pyright: ignore[reportUnknownMemberType]  # the shared helper takes a bare dict
            api_key=api_key,
            api_base=api_base,
            endpoint_type="anthropic_messages",
            custom_endpoint=False,
            headers=headers,
            custom_user_agent=self.configured_user_agent(litellm_params),
        )
        workspace_api_base: Final = messages_api_base.removesuffix(DATABRICKS_ANTHROPIC_MESSAGES_PATH)
        normalized_headers: Final = {
            ("content-type" if key.lower() == "content-type" else key): value for key, value in auth_headers.items()
        }
        finalized_headers: Final[dict[str, str]] = self._finalize_messages_headers(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # the shared helper returns a bare dict  # mutable-ok: becomes the outbound headers
            headers=normalized_headers,
            optional_params=optional_params,
            messages=messages,
        )
        return finalized_headers, workspace_api_base

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict[str, object],  # mutable-ok: mirrors the base signature
        litellm_params: dict[str, object],  # mutable-ok: mirrors the base signature
        stream: bool | None = None,
    ) -> str:
        resolved_api_base: Final = self._get_api_base(api_base).rstrip("/")
        if resolved_api_base.endswith(DATABRICKS_ANTHROPIC_MESSAGES_PATH):
            return resolved_api_base
        return f"{resolved_api_base}{DATABRICKS_ANTHROPIC_MESSAGES_PATH}"

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: list[dict[str, object]],  # mutable-ok: mirrors the base signature
        anthropic_messages_optional_request_params: dict[str, object],  # mutable-ok: mirrors the base signature
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],  # mutable-ok: mirrors the base signature
    ) -> dict[str, object]:  # mutable-ok: mirrors the base signature
        anthropic_messages_request: Final[dict[str, object]] = super().transform_anthropic_messages_request(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # the shared base returns a bare dict  # mutable-ok: becomes the outbound request body
            model=model,
            messages=messages,
            anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        self._normalize_system_role_messages(anthropic_messages_request, model=model)  # pyright: ignore[reportUnknownMemberType]  # the shared helper takes a bare dict
        self._remove_scope_from_cache_control(anthropic_messages_request)  # pyright: ignore[reportUnknownMemberType]  # the shared helper takes a bare dict
        return anthropic_messages_request
