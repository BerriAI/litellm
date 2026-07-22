from typing import Literal, Optional, Tuple, Union

import litellm
from litellm._logging import verbose_logger
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams

CLAUDE_PLATFORM_SERVICE_NAME: Literal["aws-external-anthropic"] = "aws-external-anthropic"
CLAUDE_PLATFORM_BEDROCK_ROUTE = "claude_platform/"

# Auth/routing params consumed by validate_environment / sign_request that
# must not be forwarded in the Messages API request body (together with any
# key prefixed "aws_") — the API rejects unknown fields with
# "Extra inputs are not permitted".
CLAUDE_PLATFORM_ON_AWS_NON_REQUEST_PARAMS = {
    "workspace_id",
    "anthropic_workspace_id",
    "anthropic-workspace-id",
}

# Messages API fields that are valid on Anthropic's first-party API but are
# not yet supported by the Claude Platform on AWS (aws-external-anthropic)
# endpoint, which rejects them with "Extra inputs are not permitted". Unlike
# the auth params above these carry user intent, so dropping them is logged at
# WARNING — the request succeeds but the requested feature is not applied.
CLAUDE_PLATFORM_ON_AWS_UNSUPPORTED_REQUEST_PARAMS = {
    "context_management",
}


def filter_claude_platform_request_body(
    params: dict,
    unsupported_override: Optional[frozenset[str]] = None,
    log_dropped: bool = True,
) -> dict:
    """Return a copy of ``params`` with fields the Claude Platform on AWS
    endpoint rejects removed.

    Strips auth/routing config (workspace-id aliases and any ``aws_``-prefixed
    key) silently, since those are consumed by validate_environment /
    sign_request and never belong in the body. Strips Messages API fields the
    AWS endpoint does not support yet (e.g. ``context_management``) with a
    WARNING, since those reflect user intent that will not be applied on this
    route.

    ``unsupported_override``, when provided, replaces the default
    CLAUDE_PLATFORM_ON_AWS_UNSUPPORTED_REQUEST_PARAMS set. Pass an empty frozenset
    to disable unsupported-param filtering entirely (e.g. when the AWS
    endpoint adds support before a litellm release).

    Filters a copy so callers' ``sign_request`` still sees ``aws_region_name``.
    """
    unsupported = (
        unsupported_override
        if unsupported_override is not None
        else CLAUDE_PLATFORM_ON_AWS_UNSUPPORTED_REQUEST_PARAMS
    )
    dropped_unsupported = [k for k in params if k in unsupported]
    if dropped_unsupported and log_dropped:
        verbose_logger.warning(
            "bedrock/claude_platform: dropping unsupported Messages API "
            "param(s) %s from the request body - the Claude Platform on AWS "
            "(aws-external-anthropic) endpoint does not support them and "
            "rejects unknown fields. The request will proceed without them.",
            dropped_unsupported,
        )
    return {
        k: v
        for k, v in params.items()
        if k not in CLAUDE_PLATFORM_ON_AWS_NON_REQUEST_PARAMS
        and k not in unsupported
        and not k.startswith("aws_")
    }


def resolve_unsupported_override(
    litellm_params: Union[dict[str, object], GenericLiteLLMParams],
    log_invalid: bool = True,
) -> Optional[frozenset[str]]:
    """Read ``claude_platform_unsupported_params`` from litellm_params.

    Returns None (use the default set) when the key is absent or set to a
    non-collection type; returns a frozenset when a list/set/tuple is given,
    letting operators override or clear the unsupported-param list via proxy
    config without a code change. A present but non-collection value is a
    misconfiguration: it is ignored (defaults apply) and logged at WARNING when
    ``log_invalid`` is set, so operators see the override had no effect. The
    header-derivation pass disables it so the warning fires once per request.
    """
    raw = litellm_params.get("claude_platform_unsupported_params")
    if raw is None:
        return None
    if isinstance(raw, (list, set, frozenset, tuple)):
        return frozenset(str(item) for item in raw)
    if log_invalid:
        verbose_logger.warning(
            "bedrock/claude_platform: ignoring claude_platform_unsupported_params "
            "of unsupported type %s; expected a list/set/tuple of param names. "
            "Using the default unsupported-param set.",
            type(raw).__name__,
        )
    return None


def strip_claude_platform_route(model: str) -> str:
    if model.startswith(CLAUDE_PLATFORM_BEDROCK_ROUTE):
        return model.replace(CLAUDE_PLATFORM_BEDROCK_ROUTE, "", 1)
    return model


class BedrockClaudePlatformMixin(BaseAWSLLM):
    @staticmethod
    def _get_workspace_id(optional_params: dict, litellm_params: dict) -> Optional[str]:
        workspace_id = (
            optional_params.get("workspace_id")
            or litellm_params.get("workspace_id")
            or optional_params.get("aws_workspace_id")
            or litellm_params.get("aws_workspace_id")
            or optional_params.get("anthropic-workspace-id")
            or litellm_params.get("anthropic-workspace-id")
        )
        if workspace_id is None:
            workspace_id = optional_params.get("anthropic_workspace_id") or litellm_params.get("anthropic_workspace_id")
        if workspace_id is not None:
            return str(workspace_id)
        return get_secret_str("ANTHROPIC_AWS_WORKSPACE_ID") or get_secret_str("ANTHROPIC_WORKSPACE_ID")

    def _get_required_aws_region_name(self, optional_params: dict) -> str:
        aws_region_name = (
            optional_params.get("aws_region_name")
            or get_secret_str("AWS_REGION_NAME")
            or get_secret_str("AWS_REGION")
            or get_secret_str("AWS_DEFAULT_REGION")
        )
        if aws_region_name is None:
            raise litellm.AuthenticationError(
                message=(
                    "Missing AWS region for Claude Platform on AWS. Pass "
                    "`aws_region_name` or set a standard AWS region environment value."
                ),
                llm_provider="bedrock",
                model="",
            )
        self._validate_aws_region_name(str(aws_region_name))
        return str(aws_region_name)

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        api_base = (
            api_base
            or litellm.api_base
            or get_secret_str("ANTHROPIC_AWS_BASE_URL")
            or get_secret_str("ANTHROPIC_AWS_API_BASE")
        )
        if api_base is None:
            aws_region_name = self._get_required_aws_region_name(optional_params)
            api_base = f"https://{CLAUDE_PLATFORM_SERVICE_NAME}.{aws_region_name}.api.aws"
        if not api_base.endswith("/v1/messages"):
            api_base = f"{api_base.rstrip('/')}/v1/messages"
        return api_base

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
        if api_key or get_secret_str("ANTHROPIC_AWS_API_KEY"):
            return headers, None

        return self._sign_request(
            service_name=CLAUDE_PLATFORM_SERVICE_NAME,
            headers=headers,
            optional_params=optional_params,
            request_data=request_data,
            api_base=api_base,
            model=model,
            stream=stream,
            fake_stream=fake_stream,
        )
