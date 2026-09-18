"""Direct-connect to SAP AI Core OpenAI (GPT) foundation-model deployments on `/chat/completions`.

The GPT sibling of `SapDeploymentAnthropicChatConfig`. Where the Anthropic config wraps the Bedrock
invoke transforms (Claude on SAP is served by an ``aws-bedrock`` executable at ``/invoke``), GPT on SAP
is served by an Azure OpenAI executable that speaks the OpenAI chat-completions contract at
``{deployment_url}/chat/completions?api-version=...``. So this config wraps `OpenAIGPTConfig`, whose
`get_supported_openai_params` is broad, and swaps auth and URL for SAP. Wrapping the OpenAI config
(not `AzureOpenAIConfig`, whose `transform_response`/`validate_environment` defer to the OpenAI SDK) is
what routes the request through the generic httpx flow and keeps native params flowing through instead
of being narrowed the way orchestration narrows them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final
from urllib.parse import urlencode

from litellm.llms.openai.chat.gpt_5_transformation import OpenAIGPT5Config, is_gpt_reasoning_series_name
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.types.llms.openai import AllMessageValues

from ..direct_connect import (
    allowlist_openai_chat_body,
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

_CHAT_COMPLETIONS_PATH: Final = "/chat/completions"
_DEFAULT_API_VERSION: Final = "2024-12-01-preview"


class SapDeploymentOpenAIChatConfig(OpenAIGPTConfig):
    """Reuse the OpenAI chat transforms and SSE streaming; swap auth and URL for SAP direct connect."""

    def __init__(
        self,
        *,
        http_client: HTTPHandler | None = None,
        token_creator_factory: TokenCreatorFactory = get_token_creator,
        gpt5_config: OpenAIGPT5Config | None = None,
    ) -> None:
        super().__init__()
        self._http_client: Final = http_client
        self._token_creator_factory: Final = token_creator_factory
        self._gpt5_config: Final = gpt5_config if gpt5_config is not None else OpenAIGPT5Config()

    @property
    def custom_llm_provider(self) -> str | None:
        return "sap"

    def get_supported_openai_params(self, model: str) -> list[str]:
        _, bare_model = split_sap_submode(model)
        if is_gpt_reasoning_series_name(bare_model):
            reasoning: Final[list[str]] = self._gpt5_config.get_supported_openai_params(bare_model)
            return reasoning
        supported: Final[list[str]] = super().get_supported_openai_params(bare_model)
        return supported

    def map_openai_params(
        self,
        non_default_params: dict[str, object],  # mutable-ok: litellm base transform override signature
        optional_params: dict[str, object],  # mutable-ok: litellm base transform override signature
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:
        _, bare_model = split_sap_submode(model)
        if is_gpt_reasoning_series_name(bare_model):
            reasoning: Final[dict[str, object]] = self._gpt5_config.map_openai_params(
                non_default_params, optional_params, bare_model, drop_params
            )
            return reasoning
        mapped: Final[dict[str, object]] = super().map_openai_params(
            non_default_params, optional_params, bare_model, drop_params
        )
        return mapped

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: litellm base transform override signature
        optional_params: dict,  # mutable-ok: litellm base transform override signature
        litellm_params: dict,  # mutable-ok: litellm base transform override signature
        headers: dict,  # mutable-ok: litellm base transform override signature
    ) -> dict:  # mutable-ok: litellm base transform override signature
        _, bare_model = split_sap_submode(model)
        body: Final = super().transform_request(bare_model, messages, without_resource_group(optional_params), litellm_params, headers)
        return allowlist_openai_chat_body(body, frozenset(self.get_supported_openai_params(bare_model)))

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
        api_version: Final = self._resolve_api_version(litellm_params)
        if api_base:
            return self._chat_url(api_base, api_version)
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
        return self._chat_url(deployment_url, api_version)

    @staticmethod
    def _resolve_api_version(litellm_params: dict) -> str:  # mutable-ok: litellm base transform override signature
        value: Final = litellm_params.get("api_version")
        return value if isinstance(value, str) and value else _DEFAULT_API_VERSION

    @staticmethod
    def _chat_url(base: str, api_version: str) -> str:
        params: Final = {"api-version": api_version}  # mutable-ok: urlencode query params
        return f"{base}{_CHAT_COMPLETIONS_PATH}?{urlencode(params)}"
