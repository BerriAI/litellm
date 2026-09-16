from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from typing import Final

from models import LiteLLMParamsBody, ModelMode

LIVE_PROVIDER_REQUIRED: Final[ContextVar[bool]] = ContextVar("live_provider_required", default=False)

DEFAULT_BEDROCK_REGION: Final = "us-east-1"
BEDROCK_CROSS_REGION_PREFIX: Final = "us."
BEDROCK_EDGE_MODELS: Final = frozenset(
    {
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "us.anthropic.claude-sonnet-5",
        "us.anthropic.claude-opus-4-7",
    }
)
ENV_REFERENCE_PREFIX: Final = "os.environ/"


def bedrock_region(declared: str | None, model: str) -> str | None:
    """The region whose edge mount a deployment belongs to, or None when the
    harness cannot know it.

    Most Bedrock deployments declare `os.environ/AWS_REGION`, which the proxy
    resolves from its own environment. The run pod does not share that
    environment, so the harness genuinely does not know the region. A `us.`
    inference profile fans out across the US regions and is reachable from any
    of them, so the default entry point is correct for those whatever the proxy
    resolved; anything else keeps its direct path rather than being sent to a
    region the model may not exist in."""
    if declared is None:
        return DEFAULT_BEDROCK_REGION
    if not declared.startswith(ENV_REFERENCE_PREFIX):
        return declared
    return DEFAULT_BEDROCK_REGION if model.startswith(BEDROCK_CROSS_REGION_PREFIX) else None


def bedrock_mount(params: LiteLLMParamsBody) -> str | None:
    """The edge mount a Bedrock deployment belongs to, or None.

    The allowlist mirrors the runner role's IAM policy, which names its models
    one by one. A model outside it would be re-signed with an identity that
    cannot invoke it and come back 403 from Bedrock, so an unlisted model keeps
    its direct path and loses only caching. Adding a model is a policy edit in
    litellm-ops and a line here."""
    route: Final = params.model.partition("/")[2]
    model: Final = route.partition("/")[2] or route
    if model not in BEDROCK_EDGE_MODELS:
        return None
    region: Final = bedrock_region(params.aws_region_name, model)
    return None if region is None else f"bedrock/{region}"


def route_bedrock(
    params: LiteLLMParamsBody, base_for: Callable[[str], str | None], mode: ModelMode | None,
) -> LiteLLMParamsBody:
    """Deployments that carry their own AWS identity stay off the edge. The edge
    re-signs with the run pod's role, so routing an `aws_role_name` deployment
    would quietly replace the very assume-role chain that test exists to prove."""
    if mode is not None or params.aws_role_name is not None or params.aws_access_key_id is not None:
        return params
    if params.api_base is not None or params.aws_bedrock_runtime_endpoint is not None:
        return params
    mount: Final = bedrock_mount(params)
    if mount is None:
        return params
    base: Final = base_for(mount)
    if base is None:
        return params
    return params.model_copy(update={"aws_bedrock_runtime_endpoint": base})


def route_cache_model(
    params: LiteLLMParamsBody, base_for: Callable[[str], str | None], *, enabled: bool, mode: ModelMode | None = None,
) -> LiteLLMParamsBody:
    if not enabled or LIVE_PROVIDER_REQUIRED.get() or params.mock_response is not None:
        return params
    if params.litellm_credential_name is not None:
        return params
    provider: Final = params.model.partition("/")[0]
    if provider == "bedrock":
        return route_bedrock(params, base_for, mode)
    if mode == "realtime" or params.api_base is not None:
        return params
    if provider not in {"openai", "anthropic"}:
        return params
    base: Final = base_for(provider)
    if base is None:
        return params
    return params.model_copy(update={"api_base": f"{base}/v1" if provider == "openai" else base})
