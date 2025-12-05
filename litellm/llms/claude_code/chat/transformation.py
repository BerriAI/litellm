"""
Claude Code Chat Configuration

Translates OpenAI-style chat completion requests to Claude Code CLI format.
"""

from typing import Any, List, Optional, Union, TYPE_CHECKING

import httpx

from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.utils import ModelResponse


class ClaudeCodeError(BaseLLMException):
    """Exception class for Claude Code errors."""
    pass


class ClaudeCodeChatConfig(BaseConfig):
    """
    Configuration for Claude Code CLI provider.

    Claude Code CLI Arguments:
    claude --system-prompt <prompt> --verbose --output-format stream-json
           --disallowedTools <tools> --max-turns 1 --model <model> -p
    """

    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    thinking_budget_tokens: Optional[int] = None

    # Claude Code specific settings
    claude_code_path: Optional[str] = None
    max_turns: int = 1

    # Tools to disable in Claude Code (forces custom tool format)
    DISABLED_TOOLS = [
        "Task",
        "Bash",
        "Glob",
        "Grep",
        "LS",
        "exit_plan_mode",
        "Read",
        "Edit",
        "MultiEdit",
        "Write",
        "NotebookRead",
        "NotebookEdit",
        "WebFetch",
        "TodoRead",
        "TodoWrite",
        "WebSearch",
    ]

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        thinking_budget_tokens: Optional[int] = None,
        claude_code_path: Optional[str] = None,
        max_turns: int = 1,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "claude_code"

    @classmethod
    def get_config(cls) -> dict:
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not k.startswith("_")
            and not callable(v)
            and k not in ["DISABLED_TOOLS"]
        }

    def get_supported_openai_params(self, model: str) -> List[str]:
        """
        Returns list of supported OpenAI parameters for Claude Code.
        """
        return [
            "max_tokens",
            "temperature",
            "messages",
            "stream",
            "tools",
            "tool_choice",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool = False,
    ) -> dict:
        """
        Map OpenAI parameters to Claude Code format.
        """
        for param, value in non_default_params.items():
            if param == "max_tokens":
                optional_params["max_tokens"] = value
            elif param == "temperature":
                optional_params["temperature"] = value
            elif param == "stream":
                optional_params["stream"] = value

        return optional_params

    def get_claude_code_path(self) -> str:
        """Get the path to the Claude Code CLI executable."""
        return (
            self.claude_code_path
            or get_secret_str("CLAUDE_CODE_PATH")
            or "claude"
        )

    def get_disabled_tools_string(self) -> str:
        """Get comma-separated string of disabled tools."""
        return ",".join(self.DISABLED_TOOLS)

    def transform_messages_to_claude_code_format(
        self,
        messages: List[AllMessageValues],
        system_prompt: Optional[str] = None,
    ) -> tuple:
        """
        Transform OpenAI-style messages to Claude Code CLI format.

        Returns:
            tuple: (system_prompt, transformed_messages)
        """
        transformed_messages = []
        extracted_system_prompt = system_prompt or ""

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            # Extract system prompt from messages
            if role == "system":
                if isinstance(content, str):
                    extracted_system_prompt = content
                elif isinstance(content, list):
                    # Handle content blocks
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                    extracted_system_prompt = "\n".join(text_parts)
                continue

            # Transform user/assistant messages
            transformed_message = {
                "role": role,
                "content": self._transform_content(content),
            }
            transformed_messages.append(transformed_message)

        return extracted_system_prompt, transformed_messages

    def _transform_content(self, content: Any) -> Any:
        """
        Transform message content to Anthropic format for Claude Code.
        Claude Code supports images via base64 encoding.
        """
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            transformed = []
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type", "")

                    if block_type == "text":
                        transformed.append(block)
                    elif block_type == "image":
                        # Anthropic native image format - pass through directly
                        transformed.append(block)
                    elif block_type == "image_url":
                        # Convert OpenAI image_url format to Anthropic image format
                        image_url = block.get("image_url", {})
                        url = image_url.get("url", "") if isinstance(image_url, dict) else image_url

                        if url.startswith("data:"):
                            # Parse data URL: data:image/jpeg;base64,<data>
                            # Format: data:<media_type>;base64,<base64_data>
                            try:
                                header, base64_data = url.split(",", 1)
                                media_type = header.split(":")[1].split(";")[0]
                                transformed.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": base64_data,
                                    }
                                })
                            except (ValueError, IndexError):
                                # If parsing fails, pass through as-is
                                transformed.append(block)
                        else:
                            # URL-based image - pass through for Claude Code to handle
                            transformed.append({
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": url,
                                }
                            })
                    else:
                        # Pass through other content types (tool_use, tool_result, etc.)
                        transformed.append(block)
                else:
                    transformed.append(block)
            return transformed

        return content

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """
        Validate that Claude Code CLI is available.
        """
        # No API key needed - Claude Code handles its own auth
        return {}

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform request for Claude Code.
        Note: Claude Code uses CLI, so this returns the data needed for subprocess.
        """
        system_prompt, transformed_messages = self.transform_messages_to_claude_code_format(
            messages=messages
        )
        return {
            "model": model,
            "system_prompt": system_prompt,
            "messages": transformed_messages,
            "optional_params": optional_params,
        }

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: "ModelResponse",
        logging_obj: "LiteLLMLoggingObj",
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> "ModelResponse":
        """
        Transform response from Claude Code.
        Note: Claude Code handler processes responses directly, so this is a passthrough.
        """
        return model_response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[dict, httpx.Headers],
    ) -> BaseLLMException:
        """
        Return the appropriate error class for Claude Code errors.
        """
        return ClaudeCodeError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
