"""Direct-connect to SAP AI Core foundation-model deployments on the Anthropic `/v1/messages` surface.

The orchestration config (`GenAIHubOrchestrationConfig`) POSTs `{deployment_url}/v2/completion`
and cannot reach a foundation-model deployment's native `/invoke` endpoint. This config sends the
AWS Bedrock InvokeModel Anthropic body (which preserves `cache_control` for prompt caching) straight
to `{deployment_url}/invoke` (streaming: `/invoke-with-response-stream`), authed with a SAP bearer
token plus `AI-Resource-Group` instead of SigV4. The deployment URL is auto-discovered from the
model name via `GET /lm/deployments`, or pinned explicitly through `api_base`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation import (
    AmazonAnthropicClaudeMessagesConfig,
)

from ..chat.handler import GenAIHubOrchestrationError
from ..direct_connect import (
    _canonical_anthropic_model,
    allowlist_anthropic_invoke_body,
    build_sap_auth,
    get_token_creator,
    resolve_sap_deployment_url,
    resource_group_from_params,
    without_resource_group,
)
from ..submode import split_sap_submode

if TYPE_CHECKING:
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    from litellm.types.router import GenericLiteLLMParams

    from ..direct_connect import TokenCreatorFactory

_INVOKE_PATH: Final = "/invoke"
_INVOKE_STREAM_PATH: Final = "/invoke-with-response-stream"
_EVENT_STREAM_ACCEPT: Final = "application/vnd.amazon.eventstream"


class SapDeploymentAnthropicMessagesConfig(AmazonAnthropicClaudeMessagesConfig):
    """Reuse the Bedrock invoke body/response/stream-decoder transforms; swap auth, URL and signing for SAP."""

    def __init__(
        self,
        *,
        http_client: HTTPHandler | None = None,
        token_creator_factory: TokenCreatorFactory = get_token_creator,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # base __init__ is untyped (**kwargs)
        self._http_client: Final = http_client
        self._token_creator_factory: Final = token_creator_factory

    @property
    def custom_llm_provider(self) -> str | None:
        return "sap"

    def validate_anthropic_messages_environment(
        self,
        headers: dict[str, str],  # mutable-ok: litellm base transform override signature
        model: str,
        messages: list[object],  # mutable-ok: litellm base transform override signature
        optional_params: dict[str, object],  # mutable-ok: litellm base transform override signature
        litellm_params: dict[str, object],  # mutable-ok: litellm base transform override signature
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> tuple[dict[str, str], str | None]:  # mutable-ok: litellm base transform override signature
        try:
            sap_headers, base_url, resource_group = build_sap_auth(
                api_key, self._token_creator_factory, headers, resource_group_from_params(litellm_params)
            )
        except ValueError as err:
            raise GenAIHubOrchestrationError(status_code=400, message=str(err))
        _, bare_model = split_sap_submode(model)
        deployment_url: Final = resolve_sap_deployment_url(
            base_url=base_url,
            resource_group=resource_group,
            model=bare_model,
            headers=sap_headers,
            http_client=self._http_client,
            api_base=api_base,
        )
        return sap_headers, deployment_url

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict[str, object],  # mutable-ok: litellm base transform override signature
        litellm_params: dict[str, object],  # mutable-ok: litellm base transform override signature
        stream: bool | None = None,
    ) -> str:
        if not api_base:
            raise GenAIHubOrchestrationError(
                status_code=500,
                message=f"No SAP AI Core deployment URL resolved for model '{model}'.",
            )
        suffix: Final = _INVOKE_STREAM_PATH if stream else _INVOKE_PATH
        return f"{api_base}{suffix}"

    def sign_request(
        self,
        headers: dict[str, str],  # mutable-ok: litellm base transform override signature
        optional_params: dict[str, object],  # mutable-ok: litellm base transform override signature
        request_data: dict[str, object],  # mutable-ok: litellm base transform override signature
        api_base: str,
        api_key: str | None = None,
        model: str | None = None,
        stream: bool | None = None,
        fake_stream: bool | None = None,
    ) -> tuple[dict[str, str], bytes | None]:  # mutable-ok: litellm base transform override signature
        if stream:
            return {**headers, "accept": _EVENT_STREAM_ACCEPT}, None  # mutable-ok: SAP request headers dict
        return headers, None

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: list[dict],  # mutable-ok: litellm base transform override signature
        anthropic_messages_optional_request_params: dict,  # mutable-ok: litellm base transform override signature
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: litellm base transform override signature
    ) -> dict:  # mutable-ok: litellm base transform override signature
        body: Final = super().transform_anthropic_messages_request(
            model=_canonical_anthropic_model(model),
            messages=messages,
            anthropic_messages_optional_request_params=without_resource_group(
                anthropic_messages_optional_request_params
            ),
            litellm_params=litellm_params,
            headers=headers,
        )
        return allowlist_anthropic_invoke_body(body)
