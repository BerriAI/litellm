"""Direct-connect to SAP AI Core OpenAI (GPT) foundation-model deployments on `/v1/responses`.

The Responses-API sibling of `SapDeploymentOpenAIChatConfig`. GPT on SAP is served by an Azure OpenAI
executable that speaks the OpenAI contract, so this config wraps `OpenAIResponsesAPIConfig` (reusing its
request/response/streaming transforms) and swaps auth and URL for SAP direct connect. The native Azure
responses config appends `/openai/responses` to an account-root base; SAP's discovered deployment URL is
already the full per-deployment path, so the wire path is `{deployment_url}/responses?api-version=...`,
mirroring the chat path's `{deployment_url}/chat/completions?api-version=...` exactly.

Only GPT deployments reach this config: the responses dispatch returns it for GPT models and `None` for
Claude, which keeps flowing through the chat-completions bridge (SAP Claude runs on a Bedrock invoke
executable with no native Responses surface). Because no SAP model pins `api_base`, the deployment URL is
resolved by discovery. The base `get_complete_url` receives neither `model` nor `api_key`, so the bare
model and its dependencies are injected at construction: `validate_environment` (which has the SAP service
key in `litellm_params`) runs discovery and memoizes the URL, and `get_complete_url` reads it back by the
same `(base_url, resource_group, model)` key the shared memo uses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final
from urllib.parse import urlencode

from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.types.utils import LlmProviders

from ..direct_connect import (
    allowlist_openai_responses_body,
    build_sap_auth,
    get_token_creator,
    resolve_sap_deployment_url,
    resource_group_from_params,
)
from ..submode import split_sap_submode

if TYPE_CHECKING:
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    from litellm.types.llms.openai import ResponseInputParam
    from litellm.types.router import GenericLiteLLMParams

    from ..direct_connect import TokenCreatorFactory

_RESPONSES_PATH: Final = "/responses"
_DEFAULT_API_VERSION: Final = "2024-12-01-preview"


class SapDeploymentOpenAIResponsesConfig(OpenAIResponsesAPIConfig):
    """Reuse the OpenAI responses transforms and SSE streaming; swap auth and URL for SAP direct connect.

    `model` is the bare backend model name (`gpt-5.5`, `deployment/` and `sap/` prefixes already stripped),
    injected at construction because the base `get_complete_url` receives no model.
    """

    def __init__(
        self,
        *,
        model: str,
        http_client: HTTPHandler | None = None,
        token_creator_factory: TokenCreatorFactory = get_token_creator,
    ) -> None:
        super().__init__()
        self._model: Final = model
        self._http_client: Final = http_client
        self._token_creator_factory: Final = token_creator_factory

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.SAP_GENERATIVE_AI_HUB

    def validate_environment(self, headers: dict, model: str, litellm_params: GenericLiteLLMParams | None) -> dict:
        params: Final = litellm_params if litellm_params is not None else _empty_litellm_params()
        sap_headers, base_url, resource_group = build_sap_auth(
            params.api_key, self._token_creator_factory, headers, resource_group_from_params(params.model_dump())
        )
        resolve_sap_deployment_url(
            base_url=base_url,
            resource_group=resource_group,
            model=self._model,
            headers=sap_headers,
            http_client=self._http_client,
            api_base=params.api_base,
        )
        return sap_headers

    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: dict,  # mutable-ok: litellm base transform override signature
    ) -> str:
        api_version: Final = self._resolve_api_version(litellm_params)
        if api_base:
            return self._responses_url(api_base, api_version)
        _, base_url, resource_group = build_sap_auth(
            _api_key_from(litellm_params),
            self._token_creator_factory,
            resource_group=resource_group_from_params(litellm_params),
        )
        deployment_url: Final = resolve_sap_deployment_url(
            base_url=base_url,
            resource_group=resource_group,
            model=self._model,
            headers={},
            http_client=self._http_client,
            api_base=None,
        )
        return self._responses_url(deployment_url, api_version)

    def transform_responses_api_request(
        self,
        model: str,
        input: str | ResponseInputParam,  # mutable-ok: litellm base transform override signature
        response_api_optional_request_params: dict,  # mutable-ok: litellm base transform override signature
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: litellm base transform override signature
    ) -> dict:  # mutable-ok: litellm request-body dict contract
        _, bare_model = split_sap_submode(model)
        body: Final = super().transform_responses_api_request(
            model=bare_model,
            input=input,
            response_api_optional_request_params=response_api_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        return allowlist_openai_responses_body(body, frozenset(self.get_supported_openai_params(bare_model)))

    @staticmethod
    def _resolve_api_version(litellm_params: dict) -> str:  # mutable-ok: litellm base transform override signature
        value: Final = litellm_params.get("api_version")
        return value if isinstance(value, str) and value else _DEFAULT_API_VERSION

    @staticmethod
    def _responses_url(base: str, api_version: str) -> str:
        params: Final = {"api-version": api_version}  # mutable-ok: urlencode query params
        return f"{base}{_RESPONSES_PATH}?{urlencode(params)}"


def _empty_litellm_params() -> GenericLiteLLMParams:
    from litellm.types.router import GenericLiteLLMParams

    return GenericLiteLLMParams()


def _api_key_from(litellm_params: dict) -> str | None:  # mutable-ok: litellm base transform override signature
    value: Final = litellm_params.get("api_key")
    return value if isinstance(value, str) and value else None
