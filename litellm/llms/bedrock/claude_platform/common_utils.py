from collections.abc import Mapping
from typing import Final

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams

CLAUDE_PLATFORM_SERVICE_NAME: Final = "aws-external-anthropic"
CLAUDE_PLATFORM_BEDROCK_ROUTE: Final = "claude_platform/"

CLAUDE_PLATFORM_UNSUPPORTED_PARAMS_OVERRIDE_KEY: Final = "claude_platform_unsupported_params"
CLAUDE_PLATFORM_ON_AWS_NON_REQUEST_PARAMS: Final = frozenset(
    {
        "workspace_id",
        "aws_workspace_id",
        "anthropic_workspace_id",
        "anthropic-workspace-id",
        CLAUDE_PLATFORM_UNSUPPORTED_PARAMS_OVERRIDE_KEY,
    }
)
CLAUDE_PLATFORM_ON_AWS_UNSUPPORTED_REQUEST_PARAMS: Final = frozenset({"context_management"})


def filter_claude_platform_request_body(
    params: Mapping[str, object],
    unsupported_override: frozenset[str] | None = None,
    log_dropped: bool = True,
) -> dict[str, object]:
    unsupported: Final = (
        unsupported_override if unsupported_override is not None else CLAUDE_PLATFORM_ON_AWS_UNSUPPORTED_REQUEST_PARAMS
    )
    dropped_unsupported: Final = tuple(k for k in params if k in unsupported)
    if dropped_unsupported and log_dropped:
        verbose_logger.warning(
            "bedrock/claude_platform: dropping unsupported Messages API param(s) %s from the request body; "
            "the Claude Platform on AWS endpoint rejects unknown fields. The request will proceed without them.",
            dropped_unsupported,
        )
    return {
        k: v
        for k, v in params.items()
        if k not in CLAUDE_PLATFORM_ON_AWS_NON_REQUEST_PARAMS and k not in unsupported and not k.startswith("aws_")
    }


def resolve_unsupported_override(
    litellm_params: Mapping[str, object] | GenericLiteLLMParams,
    optional_params: Mapping[str, object] | None = None,
    log_invalid: bool = True,
) -> frozenset[str] | None:
    from_optional: Final = (optional_params or {}).get(CLAUDE_PLATFORM_UNSUPPORTED_PARAMS_OVERRIDE_KEY)
    raw: Final = (
        from_optional
        if from_optional is not None
        else litellm_params.get(CLAUDE_PLATFORM_UNSUPPORTED_PARAMS_OVERRIDE_KEY)
    )
    if raw is None:
        return None
    if isinstance(raw, (list, set, frozenset, tuple)):
        return frozenset(str(item) for item in raw)
    if log_invalid:
        verbose_logger.warning(
            "bedrock/claude_platform: ignoring claude_platform_unsupported_params of type %s; "
            "expected a list of param names. Using the default unsupported-param set.",
            type(raw).__name__,
        )
    return None

# Aliases litellm accepts for the workspace identifier; all of them are
# consumed only as the `anthropic-workspace-id` HTTP header and must be
# stripped from request-body params before transformation (#29272).
_WORKSPACE_ID_PARAM_KEYS: Tuple[str, ...] = (
    "workspace_id",
    "aws_workspace_id",
    "anthropic-workspace-id",
    "anthropic_workspace_id",
)


def strip_claude_platform_route(model: str) -> str:
    if model.startswith(CLAUDE_PLATFORM_BEDROCK_ROUTE):
        return model.replace(CLAUDE_PLATFORM_BEDROCK_ROUTE, "", 1)
    return model


class BedrockClaudePlatformMixin(BaseAWSLLM):
    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, object] | httpx.Headers,  # mutable-ok: base passes response headers as a dict
    ) -> BedrockError:
        return BedrockError(status_code=status_code, message=error_message, headers=headers)

    @staticmethod
    def _get_workspace_id(optional_params: dict, litellm_params: dict) -> str | None:
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

    @staticmethod
    def _pop_workspace_id_params(optional_params: dict, litellm_params: dict) -> None:
        """Strip every workspace_id alias from both param dicts.

        ``workspace_id`` (and its aliases) is consumed only as the
        ``anthropic-workspace-id`` HTTP header. If we leave the keys in
        ``optional_params``, the inherited ``AnthropicConfig.transform_request``
        serializes them into the JSON body, and Anthropic's ``/v1/messages``
        rejects unknown top-level fields (#29272).
        """
        for key in _WORKSPACE_ID_PARAM_KEYS:
            optional_params.pop(key, None)
            litellm_params.pop(key, None)

    def _get_required_aws_region_name(self, optional_params: dict) -> str:
        aws_region_name: Final = (
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
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        api_base = (
            api_base
            or litellm.api_base
            or get_secret_str("ANTHROPIC_AWS_BASE_URL")
            or get_secret_str("ANTHROPIC_AWS_API_BASE")
        )
        if api_base is None:
            aws_region_name: Final = self._get_required_aws_region_name(optional_params)
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
        api_key: str | None = None,
        model: str | None = None,
        stream: bool | None = None,
        fake_stream: bool | None = None,
    ) -> tuple[dict, bytes | None]:
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
