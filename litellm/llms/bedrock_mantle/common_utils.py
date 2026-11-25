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

from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.optional_imports import ensure_optional_import
from litellm.litellm_core_utils.prompt_templates.json_schema_utils import drop_schema_keywords
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM, SignsRequestsWithAWS
from litellm.llms.bedrock.common_utils import AmazonBedrockGlobalConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams

BEDROCK_MANTLE_DEFAULT_REGION: Final = "us-east-1"
BEDROCK_REGIONS: Final = frozenset(AmazonBedrockGlobalConfig().get_all_regions())

MANTLE_AUTH_FAILED: Final = (
    "Bedrock Mantle auth failed: no Bearer token and no usable AWS "
    "credentials. Set BEDROCK_MANTLE_API_KEY (or AWS_BEARER_TOKEN_BEDROCK) "
    "or pass api_key for Bearer auth, or provide AWS credentials "
    "(IAM role / access key / profile / web identity) for SigV4."
)


class BedrockMantleAuthError(ValueError):
    def __init__(self) -> None:
        super().__init__(MANTLE_AUTH_FAILED)


# Standard Mantle host: https://bedrock-mantle.<region>.api.aws (group 1 = region).
MANTLE_HOST_RE: Final = re.compile(r"^https?://bedrock-mantle\.([^/.]+)\.api\.aws(?=/|$)", re.IGNORECASE)


def resolve_mantle_bearer_token(api_key: str | None) -> str | None:
    return api_key or get_secret_str("BEDROCK_MANTLE_API_KEY") or get_secret_str("AWS_BEARER_TOKEN_BEDROCK")


def split_mantle_region_prefix(model: str) -> tuple[str | None, str]:
    head, sep, tail = model.partition("/")
    if sep and head in BEDROCK_REGIONS:
        return head, tail
    return None, model


def mantle_url_region(api_base: str, params: Mapping[str, object]) -> str:
    host_match: Final = MANTLE_HOST_RE.match(api_base.rstrip("/"))
    return host_match.group(1) if host_match else resolve_mantle_region({**params, "api_base": api_base})


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
        headers: dict[str, object],
        optional_params: dict[str, object],
        request_data: dict[str, object],
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
        optional_params = {**optional_params, "aws_region_name": mantle_url_region(api_base, optional_params)}
        headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}
        try:
            return sign(headers=headers, optional_params=optional_params)
        except (NoCredentialsError, PartialCredentialsError, ProfileNotFound) as e:
            raise BedrockMantleAuthError() from e
        except CredentialRetrievalError as e:
            raise ValueError(MANTLE_AUTH_FAILED) from e


def is_mantle_claude_model(model: str) -> bool:
    return "claude" in model.lower()


_GEMMA_4_REJECTED_SCHEMA_KEYWORDS: Final = frozenset(
    {
        "$dynamicAnchor",
        "$dynamicRef",
        "$recursiveAnchor",
        "$recursiveRef",
        "contains",
        "dependencies",
        "dependentRequired",
        "dependentSchemas",
        "else",
        "if",
        "maxContains",
        "maxProperties",
        "minContains",
        "minProperties",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
        "uniqueItems",
    }
)


def _declared_all_json_schema_keywords_support(
    litellm_params: Mapping[str, object] | GenericLiteLLMParams,
) -> bool | None:
    match litellm_params:
        case GenericLiteLLMParams(model_info={"supports_all_json_schema_keywords": bool(declared)}):
            return declared
        case {"model_info": {"supports_all_json_schema_keywords": bool(declared)}}:
            return declared
        case _:
            return None


def mantle_supports_all_json_schema_keywords(
    model: str,
    region: str,
    litellm_params: Mapping[str, object] | GenericLiteLLMParams,
    model_cost: Mapping[str, object],
) -> bool:
    declared: Final = _declared_all_json_schema_keywords_support(litellm_params)
    if declared is not None:
        return declared
    bare_model: Final = split_mantle_region_prefix(model)[1]
    listed: Final = next(
        (
            flag
            for key in (f"bedrock_mantle/{region}/{bare_model}", f"bedrock_mantle/{bare_model}")
            if (flag := _listed_all_json_schema_keywords_support(model_cost.get(key))) is not None
        ),
        None,
    )
    return not bare_model.startswith("google.gemma-4") if listed is None else listed


def _listed_all_json_schema_keywords_support(row: object) -> bool | None:
    match as_str_mapping(row):
        case {"supports_all_json_schema_keywords": bool(listed)}:
            return listed
        case _:
            return None


def drop_schema_keywords_gemma_4_rejects(schema: Mapping[str, object]) -> Mapping[str, object]:
    """Mantle's gemma-4 models answer 400 "The task request was rejected by the target" to a tool,
    ``response_format`` or ``text.format`` schema carrying any of these keywords, which every other
    Mantle family accepts, and Claude Code's ``AskUserQuestion`` tool sends ``propertyNames``."""
    return drop_schema_keywords(schema, _GEMMA_4_REJECTED_SCHEMA_KEYWORDS)


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
    entry: Final = model_cost.get(f"bedrock_mantle/{split_mantle_region_prefix(model)[1]}", {}) if model else {}
    if "/v1/responses" in (entry.get("supported_endpoints") or []):
        return True
    return entry.get("mode") == "responses"


def mantle_base_segment(model: str | None, model_cost: Mapping[str, Mapping[str, object]]) -> str:
    """Return the base path segment for a Bedrock Mantle model's OpenAI surface.

    Data-driven from the model's price-map use_openai_responses_path flag
    (overridable via register_model / proxy model_info). Per the AWS model cards,
    gpt-5.x and the google gemma-4-* family carry that flag and are served on the
    /openai/v1 base (.../openai/v1/responses and .../openai/v1/chat/completions);
    every other model including gpt-oss uses the standard /v1 base. The segment is
    the base for the model's whole OpenAI-compatible surface, so both the chat and
    responses configs derive from it -- there is no separate model-name rule.
    """
    entry: Final[Mapping[str, object]] = (
        model_cost.get(f"bedrock_mantle/{split_mantle_region_prefix(model)[1]}", {}) if model else {}
    )
    return "openai/v1" if entry.get("use_openai_responses_path") is True else "v1"
