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

import re
from typing import Any, Dict, List, Optional, Tuple

from botocore.exceptions import (
    CredentialRetrievalError,
    NoCredentialsError,
    PartialCredentialsError,
    ProfileNotFound,
)

from litellm._logging import verbose_logger
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

BEDROCK_MANTLE_DEFAULT_REGION = "us-east-1"

# Checked longest/most-specific first so a full endpoint URL collapses to host
# in one pass and the appended path never doubles.
_BASE_SUFFIXES_TO_STRIP = (
    "/openai/v1/responses",
    "/v1/responses",
    "/responses",
    "/openai/v1",
    "/v1",
)

# Standard Mantle host: https://bedrock-mantle.<region>.api.aws (group 1 = region).
_MANTLE_HOST_RE = re.compile(
    r"^https?://bedrock-mantle\.([^/.]+)\.api\.aws", re.IGNORECASE
)

# Per Bedrock Mantle Responses API validation errors.
_BEDROCK_MANTLE_SUPPORTED_RESPONSE_TOOL_TYPES = frozenset(
    {"function", "mcp", "custom", "namespace", "tool_search"}
)


class BedrockMantleResponsesAPIConfig(OpenAIResponsesAPIConfig):
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

    @staticmethod
    def _resolve_region(params: dict) -> str:
        region = params.get("aws_region_name")
        if region:
            return region
        base = params.get("api_base") or get_secret_str("BEDROCK_MANTLE_API_BASE")
        if base:
            match = _MANTLE_HOST_RE.match(base.rstrip("/"))
            if match:
                return match.group(1)
        return (
            get_secret_str("BEDROCK_MANTLE_REGION")
            or get_secret_str("AWS_REGION_NAME")
            or get_secret_str("AWS_REGION")
            or BEDROCK_MANTLE_DEFAULT_REGION
        )

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        region = self._resolve_region({**litellm_params, "api_base": api_base})
        base = (
            api_base
            or get_secret_str("BEDROCK_MANTLE_API_BASE")
            or f"https://bedrock-mantle.{region}.api.aws"
        )
        base = base.rstrip("/")
        for suffix in _BASE_SUFFIXES_TO_STRIP:
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        # For the standard Mantle host (including the default-region base that
        # responses/main.py auto-injects into litellm_params.api_base), pin to the
        # single resolved region so aws_region_name wins; preserve custom proxy hosts.
        if _MANTLE_HOST_RE.match(base):
            base = f"https://bedrock-mantle.{region}.api.aws"
        path = "/openai/v1/responses" if self.use_openai_path else "/v1/responses"
        return f"{base}{path}"

    def validate_environment(
        self, headers: dict, model: str, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key = (
            litellm_params.api_key
            or get_secret_str("BEDROCK_MANTLE_API_KEY")
            or get_secret_str("AWS_BEARER_TOKEN_BEDROCK")
        )
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
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
            verbose_logger.debug(
                "Bedrock Mantle Responses API: dropping unsupported tool type(s) "
                "%s (supported: %s).",
                sorted(set(dropped_types)),
                sorted(_BEDROCK_MANTLE_SUPPORTED_RESPONSE_TOOL_TYPES),
            )

        return kept

    def map_openai_params(
        self,
        response_api_optional_params: ResponsesAPIOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        params = super().map_openai_params(
            response_api_optional_params=response_api_optional_params,
            model=model,
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

    def sign_request(
        self,
        headers: dict,
        optional_params: dict,
        request_data: dict,
        api_base: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        stream: Optional[bool] = None,
        fake_stream: Optional[bool] = None,
    ) -> Tuple[dict, Optional[bytes]]:
        bearer = (
            api_key
            or get_secret_str("BEDROCK_MANTLE_API_KEY")
            or get_secret_str("AWS_BEARER_TOKEN_BEDROCK")
        )
        if not bearer:
            # SigV4 path. Pin the credential-scope region to the region of the actual
            # signing URL (api_base, already region-resolved by get_complete_url) so the
            # SigV4 scope and the URL host can never disagree. Resolve from api_base first,
            # then fall back to the regular precedence. Also drop any caller Authorization
            # so _sign_request's restore-original-Authorization step cannot override the
            # SigV4 header.
            optional_params = {
                **optional_params,
                "aws_region_name": self._resolve_region(
                    {**optional_params, "api_base": api_base}
                ),
            }
            headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}
        try:
            return self._aws_signer._sign_request(
                service_name="bedrock",
                headers=headers,
                optional_params=optional_params,
                request_data=request_data,
                api_base=api_base,
                api_key=bearer,
                model=model,
                stream=stream,
                fake_stream=fake_stream,
            )
        except (
            NoCredentialsError,
            PartialCredentialsError,
            ProfileNotFound,
            CredentialRetrievalError,
        ) as e:
            raise ValueError(
                "Bedrock Mantle auth failed: no Bearer token and no usable AWS "
                "credentials. Set BEDROCK_MANTLE_API_KEY (or AWS_BEARER_TOKEN_BEDROCK) "
                "or pass api_key for Bearer auth, or provide AWS credentials "
                "(IAM role / access key / profile / web identity) for SigV4."
            ) from e
