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
from collections.abc import Mapping
from functools import partial
from typing import Final, Literal
from urllib.parse import urlparse

from litellm.litellm_core_utils.optional_imports import ensure_optional_import
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM, SignsRequestsWithAWS
from litellm.llms.bedrock.common_utils import AmazonBedrockGlobalConfig, get_bedrock_cross_region_inference_regions
from litellm.secret_managers.main import get_secret_str

BEDROCK_MANTLE_DEFAULT_REGION: Final = "us-east-1"
BEDROCK_REGIONS: Final = frozenset(AmazonBedrockGlobalConfig().get_all_regions())
BEDROCK_GEO_PROFILES: Final = frozenset(get_bedrock_cross_region_inference_regions())

MANTLE_AUTH_FAILED: Final = (
    "Bedrock Mantle auth failed: no Bearer token and no usable AWS "
    "credentials. Set BEDROCK_MANTLE_API_KEY (or AWS_BEARER_TOKEN_BEDROCK) "
    "or pass api_key for Bearer auth, or provide AWS credentials "
    "(IAM role / access key / profile / web identity) for SigV4."
)

# Standard Mantle host: https://bedrock-mantle.<region>.api.aws (group 1 = region).
MANTLE_HOST_RE: Final = re.compile(r"^https?://bedrock-mantle\.([^/.]+)\.api\.aws(?=/|$)", re.IGNORECASE)


def resolve_mantle_bearer_token(api_key: str | None) -> str | None:
    return api_key or get_secret_str("BEDROCK_MANTLE_API_KEY") or get_secret_str("AWS_BEARER_TOKEN_BEDROCK")


def split_mantle_region_prefix(model: str) -> tuple[str | None, str]:
    head, sep, tail = model.partition("/")
    if sep and head in BEDROCK_REGIONS:
        return head, tail
    return None, model


def mantle_catalog_id(model: str) -> str:
    """``us-east-1/us.openai.gpt-6.1-sol`` -> ``openai.gpt-6.1-sol``, for cost-map lookups only."""
    model_id: Final = split_mantle_region_prefix(model)[1]
    geo, sep, profiled_id = model_id.partition(".")
    return profiled_id if sep and geo in BEDROCK_GEO_PROFILES else model_id


def mantle_routing_id(model: str, model_cost: Mapping[str, object]) -> str:
    """The id whose ``bedrock_mantle/`` row decides routing: an exactly registered geo id first, else the bare id."""
    model_id: Final = split_mantle_region_prefix(model)[1]
    return model_id if f"bedrock_mantle/{model_id}" in model_cost else mantle_catalog_id(model_id)


def is_bedrock_runtime_host(api_base: str | None) -> bool:
    """bedrock-runtime, a VPC endpoint for it included, accepts fewer Responses tools and params than Mantle."""
    host: Final = urlparse(api_base).hostname if api_base else None
    return host is not None and any(label.startswith("bedrock-runtime") for label in host.split("."))


def aws_error_envelope_type(body: object) -> str | None:
    """bedrock-runtime answers a path it does not serve with a 200 ``{"Output": {"__type": ...}}`` body."""
    output: Final = body.get("Output") if isinstance(body, Mapping) else None
    error_type: Final = output.get("__type") if isinstance(output, Mapping) else None
    return error_type if isinstance(error_type, str) else None


def aws_error_envelope_message(error_type: str) -> str:
    return (
        f"Bedrock answered HTTP 200 with {error_type} and no completion, so the request URL names an operation "
        "it does not serve. Check the deployment's api_base and model id."
    )


def resolve_mantle_region(params: Mapping[str, object]) -> str:
    region: Final = params.get("aws_region_name")
    if isinstance(region, str) and region:
        BaseAWSLLM.validate_aws_region_name(region)
        return region
    api_base: Final = params.get("api_base")
    base: Final = (api_base if isinstance(api_base, str) else None) or get_secret_str("BEDROCK_MANTLE_API_BASE")
    if base:
        match: Final = MANTLE_HOST_RE.match(base.rstrip("/"))
        if match:
            return match.group(1)
    return (
        get_secret_str("BEDROCK_MANTLE_REGION")
        or get_secret_str("AWS_REGION_NAME")
        or get_secret_str("AWS_REGION")
        or BEDROCK_MANTLE_DEFAULT_REGION
    )


class BedrockMantleAuthMixin(SignsRequestsWithAWS):
    _aws_signer: BaseAWSLLM

    @staticmethod
    def _resolve_bearer_token(api_key: str | None) -> str | None:
        return resolve_mantle_bearer_token(api_key)

    @staticmethod
    def _resolve_region(params: Mapping[str, object]) -> str:
        return resolve_mantle_region(params)

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
    ) -> tuple[dict, bytes | None]:
        bearer: Final = self._resolve_bearer_token(api_key)
        sign: Final[partial[tuple[dict[str, str | bytes], bytes | None]]] = partial(
            self._aws_signer._sign_request,
            service_name="bedrock",
            request_data=request_data,
            api_base=api_base,
            api_key=bearer,
            model=model,
            stream=stream,
            fake_stream=fake_stream,
        )
        if bearer:
            return sign(headers=headers, optional_params=optional_params)
        ensure_optional_import("botocore")
        from botocore.exceptions import (
            CredentialRetrievalError,
            NoCredentialsError,
            PartialCredentialsError,
            ProfileNotFound,
        )

        # Pin the credential-scope region to the region of the actual signing URL
        # so the SigV4 scope and URL host can never disagree, even when a stale
        # api_base and aws_region_name point at different regions.
        host_match: Final = MANTLE_HOST_RE.match(api_base.rstrip("/"))
        optional_params = {
            **optional_params,
            "aws_region_name": (
                host_match.group(1) if host_match else self._resolve_region({**optional_params, "api_base": api_base})
            ),
        }
        headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}
        try:
            return sign(headers=headers, optional_params=optional_params)
        except (
            NoCredentialsError,
            PartialCredentialsError,
            ProfileNotFound,
            CredentialRetrievalError,
        ) as e:
            raise ValueError(MANTLE_AUTH_FAILED) from e


def is_mantle_claude_model(model: str) -> bool:
    return "claude" in model.lower()


def mantle_health_check_mode(model: str) -> Literal["anthropic_messages"] | None:
    return "anthropic_messages" if is_mantle_claude_model(model) else None


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
    entry: Final = model_cost.get(f"bedrock_mantle/{mantle_routing_id(model, model_cost)}", {}) if model else {}
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
    entry: Final = model_cost.get(f"bedrock_mantle/{mantle_routing_id(model, model_cost)}", {}) if model else {}
    return "openai/v1" if entry.get("use_openai_responses_path") is True else "v1"
