from typing import Any, Dict, List, Literal, Optional, Tuple

import litellm
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    DEFAULT_ANTHROPIC_API_VERSION,
    AnthropicMessagesConfig,
)
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.router import GenericLiteLLMParams


CLAUDE_PLATFORM_SERVICE_NAME: Literal["aws-external-anthropic"] = (
    "aws-external-anthropic"
)
CLAUDE_PLATFORM_BEDROCK_ROUTE = "claude_platform/"


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
            workspace_id = optional_params.get(
                "anthropic_workspace_id"
            ) or litellm_params.get("anthropic_workspace_id")
        if workspace_id is not None:
            return str(workspace_id)
        return get_secret_str("ANTHROPIC_AWS_WORKSPACE_ID") or get_secret_str(
            "ANTHROPIC_WORKSPACE_ID"
        )

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
            or get_secret_str("ANTHROPIC_BASE_URL")
            or get_secret_str("ANTHROPIC_API_BASE")
        )
        if api_base is None:
            aws_region_name = self._get_required_aws_region_name(optional_params)
            api_base = (
                f"https://{CLAUDE_PLATFORM_SERVICE_NAME}.{aws_region_name}.api.aws"
            )
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
        if (
            api_key
            or get_secret_str("ANTHROPIC_AWS_API_KEY")
            or get_secret_str("ANTHROPIC_API_KEY")
        ):
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


class BedrockClaudePlatformConfig(BedrockClaudePlatformMixin, AnthropicConfig):
    """
    Bedrock Claude Platform uses Anthropic's Messages API with AWS gateway auth.
    """

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "bedrock"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Dict:
        workspace_id = self._get_workspace_id(optional_params, litellm_params)
        if workspace_id is None:
            raise litellm.AuthenticationError(
                message=(
                    "Missing workspace ID for Claude Platform on AWS. Pass "
                    "`workspace_id` or configure the provider workspace setting."
                ),
                llm_provider="bedrock",
                model=model,
            )

        api_key = (
            api_key
            or get_secret_str("ANTHROPIC_AWS_API_KEY")
            or get_secret_str("ANTHROPIC_API_KEY")
        )
        anthropic_headers = self.get_anthropic_headers(
            api_key=api_key,
            auth_token=None,
            computer_tool_used=self.is_computer_tool_used(
                tools=optional_params.get("tools")
            ),
            prompt_caching_set=self.is_cache_control_set(messages=messages),
            pdf_used=self.is_pdf_used(messages=messages),
            file_id_used=self.is_file_id_used(messages=messages),
            mcp_server_used=self.is_mcp_server_used(
                mcp_servers=optional_params.get("mcp_servers")
            ),
            web_search_tool_used=self.is_web_search_tool_used(
                tools=optional_params.get("tools")
            ),
            tool_search_used=self.is_tool_search_used(
                tools=optional_params.get("tools")
            ),
            programmatic_tool_calling_used=self.is_programmatic_tool_calling_used(
                tools=optional_params.get("tools")
            ),
            input_examples_used=self.is_input_examples_used(
                tools=optional_params.get("tools")
            ),
            effort_used=self.is_effort_used(
                optional_params=optional_params, model=model
            ),
            user_anthropic_beta_headers=self._get_user_anthropic_beta_headers(
                anthropic_beta_header=headers.get("anthropic-beta")
            ),
            code_execution_tool_used=self.is_code_execution_tool_used(
                tools=optional_params.get("tools")
            ),
            container_with_skills_used=self.is_container_with_skills_used(
                optional_params=optional_params
            ),
        )
        anthropic_headers["anthropic-workspace-id"] = workspace_id
        return {**headers, **anthropic_headers}

    def get_model_response_iterator(
        self,
        streaming_response: Any,
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ) -> Any:
        from litellm.llms.anthropic.chat.handler import ModelResponseIterator

        return ModelResponseIterator(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=bool(json_mode),
        )


class BedrockClaudePlatformMessagesConfig(
    BedrockClaudePlatformMixin, AnthropicMessagesConfig
):
    def validate_anthropic_messages_environment(
        self,
        headers: dict,
        model: str,
        messages: List[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Tuple[dict, Optional[str]]:
        workspace_id = self._get_workspace_id(optional_params, litellm_params)
        if workspace_id is None:
            raise litellm.AuthenticationError(
                message=(
                    "Missing workspace ID for Claude Platform on AWS. Pass "
                    "`workspace_id` or configure the provider workspace setting."
                ),
                llm_provider="bedrock",
                model=model,
            )

        resolved_api_key = (
            api_key
            or get_secret_str("ANTHROPIC_AWS_API_KEY")
            or get_secret_str("ANTHROPIC_API_KEY")
        )
        headers = {
            **headers,
            "anthropic-version": headers.get(
                "anthropic-version", DEFAULT_ANTHROPIC_API_VERSION
            ),
            "content-type": headers.get("content-type", "application/json"),
            "anthropic-workspace-id": workspace_id,
        }
        if resolved_api_key and "x-api-key" not in headers:
            headers["x-api-key"] = resolved_api_key

        headers = self._update_headers_with_anthropic_beta(
            headers=headers,
            optional_params=optional_params,
        )

        return headers, api_base

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: List[Dict],
        anthropic_messages_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        return super().transform_anthropic_messages_request(
            model=strip_claude_platform_route(model),
            messages=messages,
            anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )
