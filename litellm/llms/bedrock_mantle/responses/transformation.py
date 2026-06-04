"""
Amazon Bedrock Mantle - Responses API backend.

gpt-5.5 / gpt-5.4 on Mantle are exposed ONLY on the `/openai/v1/responses`
path (not the standard `/v1/responses`). Payloads and SSE follow the OpenAI
Responses spec, so this config inherits OpenAIResponsesAPIConfig and overrides
only the endpoint URL and Bearer authentication.

Auth: AWS Bedrock API key as Bearer token (BEDROCK_MANTLE_API_KEY or the
standard AWS_BEARER_TOKEN_BEDROCK), or AWS SigV4 (IAM) credentials.
"""

import re
from typing import Optional, Tuple

from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

BEDROCK_MANTLE_DEFAULT_REGION = "us-east-2"

_MANTLE_HOST_PATTERN = re.compile(r"bedrock-mantle\.([a-z0-9-]+)\.api\.aws")

# Checked longest/most-specific first so a full endpoint URL collapses to host
# in one pass and the appended path never doubles.
_BASE_SUFFIXES_TO_STRIP = (
    "/openai/v1/responses",
    "/v1/responses",
    "/responses",
    "/openai/v1",
    "/v1",
)


class BedrockMantleResponsesAPIConfig(OpenAIResponsesAPIConfig, BaseAWSLLM):
    def __init__(self):
        super().__init__()
        BaseAWSLLM.__init__(self)

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.BEDROCK_MANTLE

    def _explicit_region(self, aws_region_name: Optional[str]) -> Optional[str]:
        return (
            aws_region_name
            or get_secret_str("BEDROCK_MANTLE_REGION")
            or get_secret_str("AWS_REGION")
        )

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        explicit_region = self._explicit_region(litellm_params.get("aws_region_name"))
        base = (
            api_base
            or get_secret_str("BEDROCK_MANTLE_API_BASE")
            or f"https://bedrock-mantle.{explicit_region or BEDROCK_MANTLE_DEFAULT_REGION}.api.aws"
        )
        # get_llm_provider injects a default Mantle host whose region segment is
        # resolved without aws_region_name, so it can disagree with the SigV4
        # signing region and the endpoint rejects the request. When the caller
        # gave an explicit region, pin the host's region segment to it.
        if explicit_region:
            base = _MANTLE_HOST_PATTERN.sub(
                f"bedrock-mantle.{explicit_region}.api.aws", base
            )
        base = base.rstrip("/")
        for suffix in _BASE_SUFFIXES_TO_STRIP:
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        return f"{base}/openai/v1/responses"

    def _use_sigv4(
        self,
        api_key: Optional[str],
        aws_region_name: Optional[str],
        aws_access_key_id: Optional[str],
    ) -> bool:
        bearer_key = (
            api_key
            or get_secret_str("BEDROCK_MANTLE_API_KEY")
            or get_secret_str("AWS_BEARER_TOKEN_BEDROCK")
        )
        if bearer_key:
            return False
        return any(
            [
                aws_region_name,
                aws_access_key_id,
                get_secret_str("AWS_ROLE_NAME"),
                get_secret_str("AWS_ROLE_ARN"),
                get_secret_str("AWS_WEB_IDENTITY_TOKEN"),
                get_secret_str("AWS_WEB_IDENTITY_TOKEN_FILE"),
                get_secret_str("AWS_PROFILE_NAME"),
                get_secret_str("AWS_ACCESS_KEY_ID"),
                get_secret_str("AWS_REGION"),
                get_secret_str("AWS_REGION_NAME"),
                get_secret_str("BEDROCK_MANTLE_REGION"),
            ]
        )

    def validate_environment(
        self, headers: dict, model: str, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        litellm_params = litellm_params or GenericLiteLLMParams()
        if self._use_sigv4(
            api_key=litellm_params.api_key,
            aws_region_name=litellm_params.aws_region_name,
            aws_access_key_id=litellm_params.aws_access_key_id,
        ):
            headers.setdefault("Content-Type", "application/json")
            return headers
        api_key = (
            litellm_params.api_key
            or get_secret_str("BEDROCK_MANTLE_API_KEY")
            or get_secret_str("AWS_BEARER_TOKEN_BEDROCK")
        )
        if not api_key:
            raise ValueError(
                "Bedrock Mantle API key or AWS IAM credentials are required. "
                "Set BEDROCK_MANTLE_API_KEY, AWS_BEARER_TOKEN_BEDROCK, or pass "
                "api_key; or configure AWS credentials for SigV4 "
                "(aws_region_name, aws_access_key_id, aws_role_name, "
                "aws_web_identity_token, or aws_profile_name)."
            )
        headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def sign_request(
        self,
        headers: dict,
        optional_params: dict,
        request_data: dict,
        api_base: str,
        model: Optional[str] = None,
        stream: Optional[bool] = None,
        fake_stream: Optional[bool] = None,
    ) -> Tuple[dict, Optional[bytes]]:
        if not self._use_sigv4(
            api_key=optional_params.get("api_key"),
            aws_region_name=optional_params.get("aws_region_name"),
            aws_access_key_id=optional_params.get("aws_access_key_id"),
        ):
            return headers, None
        sign_params = dict(optional_params)
        host_match = _MANTLE_HOST_PATTERN.search(api_base)
        if host_match:
            sign_params["aws_region_name"] = host_match.group(1)
        return self._sign_request(
            service_name="bedrock",
            headers=headers,
            optional_params=sign_params,
            request_data=request_data,
            api_base=api_base,
        )

    def supports_native_file_search(self) -> bool:
        return False

    def supports_native_websocket(self) -> bool:
        return False
