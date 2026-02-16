"""
This file contains common utils for anthropic calls.
"""

from typing import Dict, List, Optional, Union

import httpx

import litellm
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_file_ids_from_messages,
)
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo, BaseTokenCounter
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.anthropic import (
    ANTHROPIC_HOSTED_TOOLS,
    ANTHROPIC_OAUTH_BETA_HEADER,
    ANTHROPIC_OAUTH_TOKEN_PREFIX,
    AllAnthropicToolsValues,
    AnthropicMcpServerTool,
)
from litellm.types.llms.openai import AllMessageValues


def optionally_handle_anthropic_oauth(
    headers: dict, api_key: Optional[str]
) -> tuple[dict, Optional[str]]:
    """
    Handle Anthropic OAuth token detection and header setup.

    If an OAuth token is detected in the Authorization header, extracts it
    and sets the required OAuth headers.

    Args:
        headers: Request headers dict
        api_key: Current API key (may be None)

    Returns:
        Tuple of (updated headers, api_key)
    """
    # Check Authorization header (passthrough / forwarded requests)
    auth_header = headers.get("authorization", "")
    if auth_header and auth_header.startswith(f"Bearer {ANTHROPIC_OAUTH_TOKEN_PREFIX}"):
        api_key = auth_header.replace("Bearer ", "")
        headers.pop("x-api-key", None)
        headers["anthropic-beta"] = ANTHROPIC_OAUTH_BETA_HEADER
        headers["anthropic-dangerous-direct-browser-access"] = "true"
        return headers, api_key
    # Check api_key directly (standard chat/completion flow)
    if api_key and api_key.startswith(ANTHROPIC_OAUTH_TOKEN_PREFIX):
        headers.pop("x-api-key", None)
        headers["authorization"] = f"Bearer {api_key}"
        headers["anthropic-beta"] = ANTHROPIC_OAUTH_BETA_HEADER
        headers["anthropic-dangerous-direct-browser-access"] = "true"
    return headers, api_key


class AnthropicError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message,
        headers: Optional[httpx.Headers] = None,
    ):
        super().__init__(status_code=status_code, message=message, headers=headers)


class AnthropicModelInfo(BaseLLMModelInfo):
    def is_cache_control_set(self, messages: List[AllMessageValues]) -> bool:
        """
        Return if {"cache_control": ..} in message content block

        Used to check if anthropic prompt caching headers need to be set.
        """
        for message in messages:
            if message.get("cache_control", None) is not None:
                return True
            _message_content = message.get("content")
            if _message_content is not None and isinstance(_message_content, list):
                for content in _message_content:
                    if "cache_control" in content:
                        return True

        return False

    def is_file_id_used(self, messages: List[AllMessageValues]) -> bool:
        """
        Return if {"source": {"type": "file", "file_id": ..}} in message content block
        """
        file_ids = get_file_ids_from_messages(messages)
        return len(file_ids) > 0

    def is_mcp_server_used(
        self, mcp_servers: Optional[List[AnthropicMcpServerTool]]
    ) -> bool:
        if mcp_servers is None:
            return False
        if mcp_servers:
            return True
        return False

    def is_computer_tool_used(
        self, tools: Optional[List[AllAnthropicToolsValues]]
    ) -> Optional[str]:
        """Returns the computer tool version if used, e.g. 'computer_20250124' or None"""
        if tools is None:
            return None
        for tool in tools:
            if "type" in tool and tool["type"].startswith("computer_"):
                return tool["type"]
        return None

    def is_web_search_tool_used(
        self, tools: Optional[List[AllAnthropicToolsValues]]
    ) -> bool:
        """Returns True if web_search tool is used"""
        if tools is None:
            return False
        for tool in tools:
            if "type" in tool and tool["type"].startswith(
                ANTHROPIC_HOSTED_TOOLS.WEB_SEARCH.value
            ):
                return True
        return False

    def is_pdf_used(self, messages: List[AllMessageValues]) -> bool:
        """
        Set to true if media passed into messages.

        """
        for message in messages:
            if (
                "content" in message
                and message["content"] is not None
                and isinstance(message["content"], list)
            ):
                for content in message["content"]:
                    if "type" in content and content["type"] != "text":
                        return True
        return False

    def is_tool_search_used(self, tools: Optional[List]) -> bool:
        """
        Check if tool search tools are present in the tools list.
        """
        if not tools:
            return False

        for tool in tools:
            tool_type = tool.get("type", "")
            if tool_type in [
                "tool_search_tool_regex_20251119",
                "tool_search_tool_bm25_20251119",
            ]:
                return True
        return False

    def is_programmatic_tool_calling_used(self, tools: Optional[List]) -> bool:
        """
        Check if programmatic tool calling is being used (tools with allowed_callers field).

        Returns True if any tool has allowed_callers containing 'code_execution_20250825'.
        """
        if not tools:
            return False

        for tool in tools:
            # Check top-level allowed_callers
            allowed_callers = tool.get("allowed_callers", None)
            if allowed_callers and isinstance(allowed_callers, list):
                if "code_execution_20250825" in allowed_callers:
                    return True

            # Check function.allowed_callers for OpenAI format tools
            function = tool.get("function", {})
            if isinstance(function, dict):
                function_allowed_callers = function.get("allowed_callers", None)
                if function_allowed_callers and isinstance(
                    function_allowed_callers, list
                ):
                    if "code_execution_20250825" in function_allowed_callers:
                        return True

        return False

    def is_input_examples_used(self, tools: Optional[List]) -> bool:
        """
        Check if input_examples is being used in any tools.

        Returns True if any tool has input_examples field.
        """
        if not tools:
            return False

        for tool in tools:
            # Check top-level input_examples
            input_examples = tool.get("input_examples", None)
            if (
                input_examples
                and isinstance(input_examples, list)
                and len(input_examples) > 0
            ):
                return True

            # Check function.input_examples for OpenAI format tools
            function = tool.get("function", {})
            if isinstance(function, dict):
                function_input_examples = function.get("input_examples", None)
                if (
                    function_input_examples
                    and isinstance(function_input_examples, list)
                    and len(function_input_examples) > 0
                ):
                    return True

        return False

    def is_effort_used(
        self, optional_params: Optional[dict], model: Optional[str] = None
    ) -> bool:
        """
        Check if effort parameter is being used.

        Returns True if effort-related parameters are present.
        """
        if not optional_params:
            return False

        # Check if reasoning_effort is provided for Claude Opus 4.5
        if model and ("opus-4-5" in model.lower() or "opus_4_5" in model.lower()):
            reasoning_effort = optional_params.get("reasoning_effort")
            if reasoning_effort and isinstance(reasoning_effort, str):
                return True

        # Check if output_config is directly provided
        output_config = optional_params.get("output_config")
        if output_config and isinstance(output_config, dict):
            effort = output_config.get("effort")
            if effort and isinstance(effort, str):
                return True

        return False

    def is_code_execution_tool_used(self, tools: Optional[List]) -> bool:
        """
        Check if code execution tool is being used.

        Returns True if any tool has type "code_execution_20250825".
        """
        if not tools:
            return False

        for tool in tools:
            tool_type = tool.get("type", "")
            if tool_type == "code_execution_20250825":
                return True
        return False

    def is_container_with_skills_used(self, optional_params: Optional[dict]) -> bool:
        """
        Check if container with skills is being used.

        Returns True if optional_params contains container with skills.
        """
        if not optional_params:
            return False

        container = optional_params.get("container")
        if container and isinstance(container, dict):
            skills = container.get("skills")
            if skills and isinstance(skills, list) and len(skills) > 0:
                return True
        return False

    def _get_user_anthropic_beta_headers(
        self, anthropic_beta_header: Optional[str]
    ) -> Optional[List[str]]:
        if anthropic_beta_header is None:
            return None
        return anthropic_beta_header.split(",")

    def get_computer_tool_beta_header(self, computer_tool_version: str) -> str:
        """
        Get the appropriate beta header for a given computer tool version.

        Args:
            computer_tool_version: The computer tool version (e.g., 'computer_20250124', 'computer_20241022')

        Returns:
            The corresponding beta header string
        """
        computer_tool_beta_mapping = {
            "computer_20250124": "computer-use-2025-01-24",
            "computer_20241022": "computer-use-2024-10-22",
        }
        return computer_tool_beta_mapping.get(
            computer_tool_version, "computer-use-2024-10-22"  # Default fallback
        )

    def get_anthropic_beta_list(
        self,
        model: str,
        optional_params: Optional[dict] = None,
        computer_tool_used: Optional[str] = None,
        prompt_caching_set: bool = False,
        file_id_used: bool = False,
        mcp_server_used: bool = False,
    ) -> List[str]:
        """
        Get list of common beta headers based on the features that are active.

        Returns:
            List of beta header strings
        """
        from litellm.types.llms.anthropic import (
            ANTHROPIC_EFFORT_BETA_HEADER,
        )

        betas = []

        # Detect features
        effort_used = self.is_effort_used(optional_params, model)

        if effort_used:
            betas.append(ANTHROPIC_EFFORT_BETA_HEADER)  # effort-2025-11-24

        if computer_tool_used:
            beta_header = self.get_computer_tool_beta_header(computer_tool_used)
            betas.append(beta_header)

        # Anthropic no longer requires the prompt-caching beta header
        # Prompt caching now works automatically when cache_control is used in messages
        # Reference: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching

        if file_id_used:
            betas.append("files-api-2025-04-14")
            betas.append("code-execution-2025-05-22")

        if mcp_server_used:
            betas.append("mcp-client-2025-04-04")

        return list(set(betas))

    def get_anthropic_headers(
        self,
        api_key: str,
        anthropic_version: Optional[str] = None,
        computer_tool_used: Optional[str] = None,
        prompt_caching_set: bool = False,
        pdf_used: bool = False,
        file_id_used: bool = False,
        mcp_server_used: bool = False,
        web_search_tool_used: bool = False,
        tool_search_used: bool = False,
        programmatic_tool_calling_used: bool = False,
        input_examples_used: bool = False,
        effort_used: bool = False,
        is_vertex_request: bool = False,
        user_anthropic_beta_headers: Optional[List[str]] = None,
        code_execution_tool_used: bool = False,
        container_with_skills_used: bool = False,
    ) -> dict:
        betas = set()
        # Anthropic no longer requires the prompt-caching beta header
        # Prompt caching now works automatically when cache_control is used in messages
        # Reference: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
        if computer_tool_used:
            beta_header = self.get_computer_tool_beta_header(computer_tool_used)
            betas.add(beta_header)
        # if pdf_used:
        #     betas.add("pdfs-2024-09-25")
        if file_id_used:
            betas.add("files-api-2025-04-14")
            betas.add("code-execution-2025-05-22")
        if mcp_server_used:
            betas.add("mcp-client-2025-04-04")
        # Tool search, programmatic tool calling, and input_examples all use the same beta header
        if tool_search_used or programmatic_tool_calling_used or input_examples_used:
            from litellm.types.llms.anthropic import ANTHROPIC_TOOL_SEARCH_BETA_HEADER

            betas.add(ANTHROPIC_TOOL_SEARCH_BETA_HEADER)

        # Effort parameter uses a separate beta header
        if effort_used:
            from litellm.types.llms.anthropic import ANTHROPIC_EFFORT_BETA_HEADER

            betas.add(ANTHROPIC_EFFORT_BETA_HEADER)

        # Code execution tool uses a separate beta header
        if code_execution_tool_used:
            betas.add("code-execution-2025-08-25")

        # Container with skills uses a separate beta header
        if container_with_skills_used:
            betas.add("skills-2025-10-02")

        _is_oauth = api_key and api_key.startswith(ANTHROPIC_OAUTH_TOKEN_PREFIX)
        headers = {
            "anthropic-version": anthropic_version or "2023-06-01",
            "accept": "application/json",
            "content-type": "application/json",
        }
        if _is_oauth:
            headers["authorization"] = f"Bearer {api_key}"
            headers["anthropic-dangerous-direct-browser-access"] = "true"
            betas.add(ANTHROPIC_OAUTH_BETA_HEADER)
        else:
            headers["x-api-key"] = api_key

        if user_anthropic_beta_headers is not None:
            betas.update(user_anthropic_beta_headers)

        # Don't send any beta headers to Vertex, except web search which is required
        if is_vertex_request is True:
            # Vertex AI requires web search beta header for web search to work
            if web_search_tool_used:
                from litellm.types.llms.anthropic import ANTHROPIC_BETA_HEADER_VALUES

                headers[
                    "anthropic-beta"
                ] = ANTHROPIC_BETA_HEADER_VALUES.WEB_SEARCH_2025_03_05.value
        elif len(betas) > 0:
            headers["anthropic-beta"] = ",".join(betas)

        return headers

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
        # Check for Anthropic OAuth token in headers
        headers, api_key = optionally_handle_anthropic_oauth(
            headers=headers, api_key=api_key
        )
        if api_key is None:
            raise litellm.AuthenticationError(
                message="Missing Anthropic API Key - A call is being made to anthropic but no key is set either in the environment variables or via params. Please set `ANTHROPIC_API_KEY` in your environment vars",
                llm_provider="anthropic",
                model=model,
            )

        tools = optional_params.get("tools")
        prompt_caching_set = self.is_cache_control_set(messages=messages)
        computer_tool_used = self.is_computer_tool_used(tools=tools)
        mcp_server_used = self.is_mcp_server_used(
            mcp_servers=optional_params.get("mcp_servers")
        )
        pdf_used = self.is_pdf_used(messages=messages)
        file_id_used = self.is_file_id_used(messages=messages)
        web_search_tool_used = self.is_web_search_tool_used(tools=tools)
        tool_search_used = self.is_tool_search_used(tools=tools)
        programmatic_tool_calling_used = self.is_programmatic_tool_calling_used(
            tools=tools
        )
        input_examples_used = self.is_input_examples_used(tools=tools)
        effort_used = self.is_effort_used(optional_params=optional_params, model=model)
        code_execution_tool_used = self.is_code_execution_tool_used(tools=tools)
        container_with_skills_used = self.is_container_with_skills_used(
            optional_params=optional_params
        )
        user_anthropic_beta_headers = self._get_user_anthropic_beta_headers(
            anthropic_beta_header=headers.get("anthropic-beta")
        )
        anthropic_headers = self.get_anthropic_headers(
            computer_tool_used=computer_tool_used,
            prompt_caching_set=prompt_caching_set,
            pdf_used=pdf_used,
            api_key=api_key,
            file_id_used=file_id_used,
            web_search_tool_used=web_search_tool_used,
            is_vertex_request=optional_params.get("is_vertex_request", False),
            user_anthropic_beta_headers=user_anthropic_beta_headers,
            mcp_server_used=mcp_server_used,
            tool_search_used=tool_search_used,
            programmatic_tool_calling_used=programmatic_tool_calling_used,
            input_examples_used=input_examples_used,
            effort_used=effort_used,
            code_execution_tool_used=code_execution_tool_used,
            container_with_skills_used=container_with_skills_used,
        )

        headers = {**headers, **anthropic_headers}

        return headers

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> Optional[str]:
        from litellm.secret_managers.main import get_secret_str

        return (
            api_base
            or get_secret_str("ANTHROPIC_API_BASE")
            or "https://api.anthropic.com"
        )

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        from litellm.secret_managers.main import get_secret_str

        return api_key or get_secret_str("ANTHROPIC_API_KEY")

    @staticmethod
    def get_base_model(model: Optional[str] = None) -> Optional[str]:
        return model.replace("anthropic/", "") if model else None

    def get_models(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None
    ) -> List[str]:
        api_base = AnthropicModelInfo.get_api_base(api_base)
        api_key = AnthropicModelInfo.get_api_key(api_key)
        if api_base is None or api_key is None:
            raise ValueError(
                "ANTHROPIC_API_BASE or ANTHROPIC_API_KEY is not set. Please set the environment variable, to query Anthropic's `/models` endpoint."
            )
        response = litellm.module_level_client.get(
            url=f"{api_base}/v1/models",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            raise Exception(
                f"Failed to fetch models from Anthropic. Status code: {response.status_code}, Response: {response.text}"
            )

        models = response.json()["data"]

        litellm_model_names = []
        for model in models:
            stripped_model_name = model["id"]
            litellm_model_name = "anthropic/" + stripped_model_name
            litellm_model_names.append(litellm_model_name)
        return litellm_model_names

    def get_token_counter(self) -> Optional[BaseTokenCounter]:
        """
        Factory method to create an Anthropic token counter.

        Returns:
            AnthropicTokenCounter instance for this provider.
        """
        from litellm.llms.anthropic.count_tokens.token_counter import (
            AnthropicTokenCounter,
        )

        return AnthropicTokenCounter()


def process_anthropic_headers(headers: Union[httpx.Headers, dict]) -> dict:
    openai_headers = {}
    if "anthropic-ratelimit-requests-limit" in headers:
        openai_headers["x-ratelimit-limit-requests"] = headers[
            "anthropic-ratelimit-requests-limit"
        ]
    if "anthropic-ratelimit-requests-remaining" in headers:
        openai_headers["x-ratelimit-remaining-requests"] = headers[
            "anthropic-ratelimit-requests-remaining"
        ]
    if "anthropic-ratelimit-tokens-limit" in headers:
        openai_headers["x-ratelimit-limit-tokens"] = headers[
            "anthropic-ratelimit-tokens-limit"
        ]
    if "anthropic-ratelimit-tokens-remaining" in headers:
        openai_headers["x-ratelimit-remaining-tokens"] = headers[
            "anthropic-ratelimit-tokens-remaining"
        ]

    llm_response_headers = {
        "{}-{}".format("llm_provider", k): v for k, v in headers.items()
    }

    additional_headers = {**llm_response_headers, **openai_headers}
    return additional_headers
