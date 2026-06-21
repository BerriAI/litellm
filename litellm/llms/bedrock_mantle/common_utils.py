"""Shared auth, region resolution, and routing helpers for the Amazon Bedrock Mantle provider.

Mantle authenticates with a Bearer token when one is available
(litellm_params.api_key, BEDROCK_MANTLE_API_KEY, or the standard
AWS_BEARER_TOKEN_BEDROCK); otherwise it falls back to AWS SigV4 (service
"bedrock") over the standard credential chain (IAM role / access key / profile /
web identity). The Chat Completions and Responses backends share this behaviour
through BedrockMantleAuthMixin so the two paths can never drift apart.

The two routing helpers (mantle_supports_responses, mantle_base_segment) are
pure functions of (model, model_cost) so they can be unit-tested without patching
global state.
"""

import re
from typing import Tuple

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
    def _resolve_bearer_token(api_key: str | None) -> str | None:
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
        api_key: str | None = None,
        model: str | None = None,
        stream: bool | None = None,
        fake_stream: bool | None = None,
    ) -> Tuple[dict, bytes | None]:
        bearer = self._resolve_bearer_token(api_key)
        if not bearer:
            # Pin the credential-scope region to the region of the actual signing URL
            # so the SigV4 scope and URL host can never disagree, even when a stale
            # api_base and aws_region_name point at different regions.
            host_match = MANTLE_HOST_RE.match(api_base.rstrip("/"))
            optional_params = {
                **optional_params,
                "aws_region_name": (
                    host_match.group(1)
                    if host_match
                    else self._resolve_region({**optional_params, "api_base": api_base})
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


def mantle_supports_responses(model: str | None, model_cost: dict) -> bool:
    """Whether a Bedrock Mantle model can serve the native Responses API.

    Purely data-driven from the model's price-map capability signal -- either
    /v1/responses in supported_endpoints, or mode=responses -- both overridable
    via register_model and proxy model_info, so onboarding a model is a JSON
    change, never a code change. There is deliberately NO model-name match here:
    capability is per-model, not per-family (openai.gpt-oss-120b supports
    Responses while openai.gpt-oss-safeguard-120b does not, despite sharing the
    gpt-oss substring), so a substring gate would be wrong. A model absent from
    model_cost simply has no signal and returns False (chat-completions emulation).
    """
    entry = model_cost.get(f"bedrock_mantle/{model}", {})
    if "/v1/responses" in (entry.get("supported_endpoints") or []):
        return True
    return entry.get("mode") == "responses"


def mantle_base_segment(model: str | None, model_cost: dict) -> str:
    """Return the base path segment for a Bedrock Mantle model's OpenAI surface.

    Data-driven from the model's price-map use_openai_responses_path flag
    (overridable via register_model / proxy model_info). Per the AWS model cards,
    gpt-5.x and the google gemma-4-* family carry that flag and are served on the
    /openai/v1 base (.../openai/v1/responses and .../openai/v1/chat/completions);
    every other model including gpt-oss uses the standard /v1 base. The segment is
    the base for the model's whole OpenAI-compatible surface, so both the chat and
    responses configs derive from it -- there is no separate model-name rule.
    """
    entry = model_cost.get(f"bedrock_mantle/{model}", {})
    return "openai/v1" if entry.get("use_openai_responses_path") is True else "v1"
