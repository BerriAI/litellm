"""
Amazon Bedrock Mantle - Responses API backend.

Mantle serves Responses on two upstream paths: gpt frontier models (gpt-5.5 /
gpt-5.4) on `/openai/v1/responses`, and everything else that supports Responses
(e.g. gpt-oss) on the standard `/v1/responses`. The gate picks the path per
model and injects it via `use_openai_path`. Payloads and SSE follow the OpenAI
Responses spec, so this config inherits OpenAIResponsesAPIConfig and overrides
only the endpoint URL and authentication.

Auth: Bearer token (BEDROCK_MANTLE_API_KEY or the standard
AWS_BEARER_TOKEN_BEDROCK, or litellm_params.api_key) when present; otherwise
AWS SigV4 (service name "bedrock") using the standard credential chain (IAM
role / access key / profile / web identity), signed via the shared
BaseAWSLLM._sign_request after the request body is finalized.
"""

from typing import Any, Dict, List, Optional

import litellm
from litellm._logging import verbose_logger
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock_mantle.common_utils import (
    MANTLE_HOST_RE,
    BedrockMantleAuthMixin,
)
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

# Checked longest/most-specific first so a full endpoint URL collapses to host
# in one pass and the appended path never doubles.
_BASE_SUFFIXES_TO_STRIP = (
    "/openai/v1/responses",
    "/v1/responses",
    "/responses",
    "/openai/v1",
    "/v1",
)

# Per Bedrock Mantle Responses API validation errors.
_BEDROCK_MANTLE_SUPPORTED_RESPONSE_TOOL_TYPES = frozenset({"function", "mcp", "custom", "namespace", "tool_search"})

_BEDROCK_MANTLE_SUPPORTED_SERVICE_TIERS = frozenset({"auto", "default"})


class BedrockMantleResponsesAPIConfig(BedrockMantleAuthMixin, OpenAIResponsesAPIConfig):
    def __init__(
        self,
        aws_signer: Optional[BaseAWSLLM] = None,
        use_openai_path: bool = True,
    ):
        super().__init__()
        self._aws_signer = aws_signer or BaseAWSLLM()
        self.use_openai_path = use_openai_path

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.BEDROCK_MANTLE

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        region = self._resolve_region({**litellm_params, "api_base": api_base})
        base = api_base or get_secret_str("BEDROCK_MANTLE_API_BASE") or f"https://bedrock-mantle.{region}.api.aws"
        base = base.rstrip("/")
        for suffix in _BASE_SUFFIXES_TO_STRIP:
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        # For the standard Mantle host (including the default-region base that
        # responses/main.py auto-injects into litellm_params.api_base), pin to the
        # single resolved region so aws_region_name wins; preserve custom proxy hosts.
        if MANTLE_HOST_RE.match(base):
            base = f"https://bedrock-mantle.{region}.api.aws"
        path = "/openai/v1/responses" if self.use_openai_path else "/v1/responses"
        return f"{base}{path}"

    def validate_environment(self, headers: dict, model: str, litellm_params: Optional[GenericLiteLLMParams]) -> dict:
        litellm_params = litellm_params or GenericLiteLLMParams()
        bearer = self._resolve_bearer_token(litellm_params.api_key)
        if bearer:
            headers["Authorization"] = f"Bearer {bearer}"
        if litellm_params.aws_bedrock_project_id:
            headers["OpenAI-Project"] = litellm_params.aws_bedrock_project_id
        return headers

    def supports_native_file_search(self) -> bool:
        return False

    def supports_native_websocket(self) -> bool:
        return False

    @staticmethod
    def _filter_unsupported_tools(tools: List[Any]) -> List[Any]:
        """Keep only tool types Mantle's Responses API accepts."""
        kept: List[Any] = []
        dropped_types: List[str] = []
        for tool in tools:
            if not isinstance(tool, dict):
                kept.append(tool)
                continue
            tool_type = tool.get("type")
            if tool_type in _BEDROCK_MANTLE_SUPPORTED_RESPONSE_TOOL_TYPES:
                kept.append(tool)
            else:
                dropped_types.append(str(tool_type))

        if dropped_types:
            verbose_logger.warning(
                "Bedrock Mantle Responses API: dropping unsupported tool type(s) %s (supported: %s).",
                sorted(set(dropped_types)),
                sorted(_BEDROCK_MANTLE_SUPPORTED_RESPONSE_TOOL_TYPES),
            )

        return kept

    @staticmethod
    def _handle_unsupported_service_tier(params: dict, drop_params: bool) -> dict:
        service_tier = params.get("service_tier")
        if service_tier is None or service_tier in _BEDROCK_MANTLE_SUPPORTED_SERVICE_TIERS:
            return params
        if not drop_params:
            raise litellm.utils.UnsupportedParamsError(
                status_code=400,
                message=(
                    f"bedrock_mantle does not support service_tier={service_tier!r}; the Bedrock Mantle "
                    "Responses API only accepts 'auto' or 'default'. Set `drop_params: true` (litellm_settings "
                    "or this deployment's litellm_params) to have LiteLLM drop it, or remove service_tier from "
                    "the client (Codex CLI sends it when a speed tier is set in ~/.codex/config.toml)."
                ),
            )
        verbose_logger.warning(
            "Bedrock Mantle Responses API: dropping unsupported service_tier %r (supported: %s).",
            service_tier,
            sorted(_BEDROCK_MANTLE_SUPPORTED_SERVICE_TIERS),
        )
        return {key: value for key, value in params.items() if key != "service_tier"}

    def map_openai_params(
        self,
        response_api_optional_params: ResponsesAPIOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        params = self._handle_unsupported_service_tier(
            super().map_openai_params(
                response_api_optional_params=response_api_optional_params,
                model=model,
                drop_params=drop_params,
            ),
            drop_params=drop_params,
        )

        tools = params.get("tools")
        if not tools:
            return params

        tools_list = tools if isinstance(tools, list) else [tools]
        filtered = self._filter_unsupported_tools(tools_list)
        if filtered:
            params["tools"] = filtered
        else:
            params.pop("tools", None)

        return params
