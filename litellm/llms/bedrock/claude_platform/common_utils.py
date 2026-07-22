from typing import Literal, Optional, Tuple

import litellm
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.secret_managers.main import get_secret_str

CLAUDE_PLATFORM_SERVICE_NAME: Literal["aws-external-anthropic"] = "aws-external-anthropic"
CLAUDE_PLATFORM_BEDROCK_ROUTE = "claude_platform/"

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
