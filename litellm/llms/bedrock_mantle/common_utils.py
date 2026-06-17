"""
Shared auth and region resolution for the Amazon Bedrock Mantle backends.

Mantle authenticates with a Bearer token when one is available
(litellm_params.api_key, BEDROCK_MANTLE_API_KEY, or the standard
AWS_BEARER_TOKEN_BEDROCK); otherwise it falls back to AWS SigV4 (service
"bedrock") over the standard credential chain (IAM role / access key / profile /
web identity). The Chat Completions and Responses backends share this behaviour
through BedrockMantleAuthMixin so the two paths can never drift apart.
"""

import re
from typing import Optional, Tuple

from botocore.exceptions import (
    CredentialRetrievalError,
    NoCredentialsError,
    PartialCredentialsError,
    ProfileNotFound,
)

from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.secret_managers.main import get_secret_str

BEDROCK_MANTLE_DEFAULT_REGION = "us-east-1"

# Standard Mantle host: https://bedrock-mantle.<region>.api.aws (group 1 = region).
MANTLE_HOST_RE = re.compile(
    r"^https?://bedrock-mantle\.([^/.]+)\.api\.aws", re.IGNORECASE
)


class BedrockMantleAuthMixin:
    _aws_signer: BaseAWSLLM

    @staticmethod
    def _resolve_bearer_token(api_key: Optional[str]) -> Optional[str]:
        return (
            api_key
            or get_secret_str("BEDROCK_MANTLE_API_KEY")
            or get_secret_str("AWS_BEARER_TOKEN_BEDROCK")
        )

    @staticmethod
    def _resolve_region(params: dict) -> str:
        region = params.get("aws_region_name")
        if region:
            BaseAWSLLM._validate_aws_region_name(region)
            return region
        base = params.get("api_base") or get_secret_str("BEDROCK_MANTLE_API_BASE")
        if base:
            match = MANTLE_HOST_RE.match(base.rstrip("/"))
            if match:
                return match.group(1)
        return (
            get_secret_str("BEDROCK_MANTLE_REGION")
            or get_secret_str("AWS_REGION_NAME")
            or get_secret_str("AWS_REGION")
            or BEDROCK_MANTLE_DEFAULT_REGION
        )

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
        bearer = self._resolve_bearer_token(api_key)
        if not bearer:
            # SigV4 path. Pin the credential-scope region to the region of the actual
            # signing URL (api_base is already region-resolved) so the SigV4 scope and
            # the URL host can never disagree. Also drop any caller Authorization so
            # _sign_request's restore-original-Authorization step cannot override the
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
