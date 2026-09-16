"""Direct-connect to SAP AI Core foundation-model deployments on the `/chat/completions` surface.

The mirror of `SapDeploymentAnthropicMessagesConfig` for chat completions: it reuses the Bedrock
Anthropic invoke body/response/stream transforms and swaps auth, URL and signing for SAP. The one
structural difference from the messages config is that chat's `validate_environment` returns headers
only, so the deployment URL is resolved in `get_complete_url` (which receives `api_base` and
`api_key`) rather than alongside the headers. The pinned-`api_base` path needs neither a token nor
discovery there; only the unpinned discovery fallback rebuilds auth to run the `GET /lm/deployments`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from litellm.llms.bedrock.chat.invoke_transformations.anthropic_claude3_transformation import (
    AmazonAnthropicClaudeConfig,
)
from litellm.types.llms.openai import AllMessageValues

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

    from ..direct_connect import TokenCreatorFactory

_INVOKE_PATH: Final = "/invoke"
_INVOKE_STREAM_PATH: Final = "/invoke-with-response-stream"
_EVENT_STREAM_ACCEPT: Final = "application/vnd.amazon.eventstream"


class SapDeploymentAnthropicChatConfig(AmazonAnthropicClaudeConfig):
    """Reuse the Bedrock invoke chat transforms; swap auth, URL and signing for SAP direct connect."""

    def __init__(
        self,
        *,
        http_client: HTTPHandler | None = None,
        token_creator_factory: TokenCreatorFactory = get_token_creator,
    ) -> None:
        super().__init__()
        self._http_client: Final = http_client
        self._token_creator_factory: Final = token_creator_factory

    @property
    def custom_llm_provider(self) -> str | None:
        return "sap"

    def get_supported_openai_params(self, model: str) -> list[str]:
        supported: Final[list[str]] = super().get_supported_openai_params(_canonical_anthropic_model(model))
        return supported

    def map_openai_params(
        self,
        non_default_params: dict[str, object],  # mutable-ok: litellm base transform override signature
        optional_params: dict[str, object],  # mutable-ok: litellm base transform override signature
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:
        mapped: Final[dict[str, object]] = super().map_openai_params(
            non_default_params, optional_params, _canonical_anthropic_model(model), drop_params
        )
        return mapped

    def validate_environment(
        self,
        headers: dict,  # mutable-ok: litellm base transform override signature
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: litellm base transform override signature
        optional_params: dict,  # mutable-ok: litellm base transform override signature
        litellm_params: dict,  # mutable-ok: litellm base transform override signature
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:  # mutable-ok: litellm base transform override signature
        sap_headers, _, _ = build_sap_auth(
            api_key, self._token_creator_factory, headers, resource_group_from_params(litellm_params)
        )
        return sap_headers

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,  # mutable-ok: litellm base transform override signature
        litellm_params: dict,  # mutable-ok: litellm base transform override signature
        stream: bool | None = None,
    ) -> str:
        suffix: Final = _INVOKE_STREAM_PATH if stream else _INVOKE_PATH
        if api_base:
            return f"{api_base}{suffix}"
        sap_headers, base_url, resource_group = build_sap_auth(
            api_key, self._token_creator_factory, resource_group=resource_group_from_params(litellm_params)
        )
        _, bare_model = split_sap_submode(model)
        deployment_url: Final = resolve_sap_deployment_url(
            base_url=base_url,
            resource_group=resource_group,
            model=bare_model,
            headers=sap_headers,
            http_client=self._http_client,
            api_base=None,
        )
        return f"{deployment_url}{suffix}"

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: litellm base transform override signature
        optional_params: dict,  # mutable-ok: litellm base transform override signature
        litellm_params: dict,  # mutable-ok: litellm base transform override signature
        headers: dict,  # mutable-ok: litellm base transform override signature
    ) -> dict:  # mutable-ok: litellm base transform override signature
        body: Final = super().transform_request(
            model=_canonical_anthropic_model(model),
            messages=messages,
            optional_params=without_resource_group(optional_params),
            litellm_params=litellm_params,
            headers=headers,
        )
        return allowlist_anthropic_invoke_body(body)

    def sign_request(
        self,
        headers: dict,  # mutable-ok: litellm base transform override signature
        optional_params: dict,  # mutable-ok: litellm base transform override signature
        request_data: dict,  # mutable-ok: litellm base transform override signature
        api_base: str,
        api_key: str | None = None,
        model: str | None = None,
        stream: bool | None = None,
        fake_stream: bool | None = None,
    ) -> tuple[dict, bytes | None]:  # mutable-ok: litellm base transform override signature
        if stream:
            return {**headers, "accept": _EVENT_STREAM_ACCEPT}, None  # mutable-ok: SAP request headers dict
        return headers, None
