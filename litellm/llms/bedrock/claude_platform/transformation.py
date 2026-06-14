from typing import Any, Dict, List, Optional

import litellm
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from .common_utils import BedrockClaudePlatformMixin


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

        api_key = api_key or get_secret_str("ANTHROPIC_AWS_API_KEY")
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
