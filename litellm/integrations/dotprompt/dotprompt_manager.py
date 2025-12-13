"""
Dotprompt manager that integrates with LiteLLM's prompt management system.
Builds on top of PromptManagementBase to provide .prompt file support.
"""

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from litellm.integrations.custom_prompt_management import CustomPromptManagement
from litellm.integrations.prompt_management_base import PromptManagementClient
from litellm.types.llms.openai import AllMessageValues
from litellm.types.prompts.init_prompts import PromptSpec
from litellm.types.utils import StandardCallbackDynamicParams

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

from .prompt_manager import PromptManager, PromptTemplate


class DotpromptManager(CustomPromptManagement):
    """
    Dotprompt manager that integrates with LiteLLM's prompt management system.

    This class enables using .prompt files with the litellm completion() function
    by implementing the PromptManagementBase interface.

    Usage:
        # Set global prompt directory
        litellm.prompt_directory = "path/to/prompts"

        # Use with completion
        response = litellm.completion(
            model="dotprompt/gpt-4",
            prompt_id="my_prompt",
            prompt_variables={"variable": "value"},
            messages=[{"role": "user", "content": "This will be combined with the prompt"}]
        )
    """

    def __init__(
        self,
        prompt_directory: Optional[str] = None,
        prompt_file: Optional[str] = None,
        prompt_data: Optional[Union[dict, str]] = None,
        prompt_id: Optional[str] = None,
    ):
        import litellm

        self.prompt_directory = prompt_directory or litellm.global_prompt_directory
        # Support for JSON-based prompts stored in memory/database
        if isinstance(prompt_data, str):
            self.prompt_data = json.loads(prompt_data)
        else:
            self.prompt_data = prompt_data or {}

        self._prompt_manager: Optional[PromptManager] = None
        self.prompt_file = prompt_file
        self.prompt_id = prompt_id

    @property
    def integration_name(self) -> str:
        """Integration name used in model names like 'dotprompt/gpt-4'."""
        return "dotprompt"

    @property
    def prompt_manager(self) -> PromptManager:
        """Lazy-load the prompt manager."""
        if self._prompt_manager is None:
            if (
                self.prompt_directory is None
                and not self.prompt_data
                and not self.prompt_file
            ):
                raise ValueError(
                    "Either prompt_directory or prompt_data must be set before using dotprompt manager. "
                    "Set litellm.global_prompt_directory, initialize with prompt_directory parameter, or provide prompt_data."
                )
            self._prompt_manager = PromptManager(
                prompt_directory=self.prompt_directory,
                prompt_data=self.prompt_data,
                prompt_file=self.prompt_file,
                prompt_id=self.prompt_id,
            )
        return self._prompt_manager

    def should_run_prompt_management(
        self,
        prompt_id: Optional[str],
        prompt_spec: Optional[PromptSpec],
        dynamic_callback_params: StandardCallbackDynamicParams,
    ) -> bool:
        """
        Determine if prompt management should run based on the prompt_id.

        Returns True if the prompt_id exists in our prompt manager.
        """
        if prompt_id is None:
            return False
        try:
            return prompt_id in self.prompt_manager.list_prompts()
        except Exception:
            # If there's any error accessing prompts, don't run prompt management
            return False

    def _compile_prompt_helper(
        self,
        prompt_id: Optional[str],
        prompt_spec: Optional[PromptSpec],
        prompt_variables: Optional[dict],
        dynamic_callback_params: StandardCallbackDynamicParams,
        prompt_label: Optional[str] = None,
        prompt_version: Optional[int] = None,
    ) -> PromptManagementClient:
        """
        Compile a .prompt file into a PromptManagementClient structure.

        This method:
        1. Loads the prompt template from the .prompt file (with optional version)
        2. Renders it with the provided variables
        3. Converts the rendered text into chat messages
        4. Extracts model and optional parameters from metadata
        """

        if prompt_id is None:
            raise ValueError("prompt_id is required for dotprompt manager")

        try:

            # Get the prompt template (versioned or base)
            template = self.prompt_manager.get_prompt(
                prompt_id=prompt_id, version=prompt_version
            )
            if template is None:
                version_str = f" (version {prompt_version})" if prompt_version else ""
                raise ValueError(
                    f"Prompt '{prompt_id}'{version_str} not found in prompt directory"
                )

            # Render the template with variables (pass version for proper lookup)
            rendered_content = self.prompt_manager.render(
                prompt_id=prompt_id,
                prompt_variables=prompt_variables,
                version=prompt_version,
            )

            # Convert rendered content to chat messages
            messages = self._convert_to_messages(rendered_content)

            # Extract model from metadata (if specified)
            template_model = template.model

            # Extract optional parameters from metadata
            optional_params = self._extract_optional_params(template)

            return PromptManagementClient(
                prompt_id=prompt_id,
                prompt_template=messages,
                prompt_template_model=template_model,
                prompt_template_optional_params=optional_params,
                completed_messages=None,
            )

        except Exception as e:
            raise ValueError(f"Error compiling prompt '{prompt_id}': {e}")

    async def async_compile_prompt_helper(
        self,
        prompt_id: Optional[str],
        prompt_variables: Optional[dict],
        dynamic_callback_params: StandardCallbackDynamicParams,
        prompt_spec: Optional[PromptSpec] = None,
        prompt_label: Optional[str] = None,
        prompt_version: Optional[int] = None,
    ) -> PromptManagementClient:
        """
        Async version of compile prompt helper. Since dotprompt operations are synchronous,
        this simply delegates to the sync version.
        """
        if prompt_id is None:
            raise ValueError("prompt_id is required for dotprompt manager")

        return self._compile_prompt_helper(
            prompt_id=prompt_id,
            prompt_spec=prompt_spec,
            prompt_variables=prompt_variables,
            dynamic_callback_params=dynamic_callback_params,
            prompt_label=prompt_label,
            prompt_version=prompt_version,
        )

    def get_chat_completion_prompt(
        self,
        model: str,
        messages: List[AllMessageValues],
        non_default_params: dict,
        prompt_id: Optional[str],
        prompt_variables: Optional[dict],
        dynamic_callback_params: StandardCallbackDynamicParams,
        prompt_spec: Optional[PromptSpec] = None,
        prompt_label: Optional[str] = None,
        prompt_version: Optional[int] = None,
        ignore_prompt_manager_model: Optional[bool] = False,
        ignore_prompt_manager_optional_params: Optional[bool] = False,
    ) -> Tuple[str, List[AllMessageValues], dict]:

        from litellm.integrations.prompt_management_base import PromptManagementBase

        return PromptManagementBase.get_chat_completion_prompt(
            self,
            model,
            messages,
            non_default_params,
            prompt_id,
            prompt_variables,
            dynamic_callback_params,
            prompt_spec=prompt_spec,
            prompt_label=prompt_label,
            prompt_version=prompt_version,
        )

    async def async_get_chat_completion_prompt(
        self,
        model: str,
        messages: List[AllMessageValues],
        non_default_params: dict,
        prompt_id: Optional[str],
        prompt_variables: Optional[dict],
        dynamic_callback_params: StandardCallbackDynamicParams,
        litellm_logging_obj: LiteLLMLoggingObj,
        prompt_spec: Optional[PromptSpec] = None,
        tools: Optional[List[Dict]] = None,
        prompt_label: Optional[str] = None,
        prompt_version: Optional[int] = None,
        ignore_prompt_manager_model: Optional[bool] = False,
        ignore_prompt_manager_optional_params: Optional[bool] = False,
    ) -> Tuple[str, List[AllMessageValues], dict]:
        """
        Async version - delegates to PromptManagementBase async implementation.
        """
        from litellm.integrations.prompt_management_base import PromptManagementBase

        return await PromptManagementBase.async_get_chat_completion_prompt(
            self,
            model,
            messages,
            non_default_params,
            prompt_id=prompt_id,
            prompt_variables=prompt_variables,
            litellm_logging_obj=litellm_logging_obj,
            dynamic_callback_params=dynamic_callback_params,
            prompt_spec=prompt_spec,
            tools=tools,
            prompt_label=prompt_label,
            prompt_version=prompt_version,
            ignore_prompt_manager_model=ignore_prompt_manager_model,
            ignore_prompt_manager_optional_params=ignore_prompt_manager_optional_params,
        )

    def _convert_to_messages(self, rendered_content: str) -> List[AllMessageValues]:
        """
        Convert rendered prompt content to chat messages.

        This method supports multiple formats:
        1. Simple text -> converted to user message
        2. Text with role prefixes (System:, User:, Assistant:) -> parsed into separate messages
        3. Already formatted as a single message
        """
        # Clean up the content
        content = rendered_content.strip()

        # Try to parse role-based format (System: ..., User: ..., etc.)
        messages = []
        current_role = None
        current_content = []

        lines = content.split("\n")

        for line in lines:
            line = line.strip()

            # Check for role prefixes
            if line.startswith("System:"):
                if current_role and current_content:
                    messages.append(
                        self._create_message(
                            current_role, "\n".join(current_content).strip()
                        )
                    )
                current_role = "system"
                current_content = [line[7:].strip()]  # Remove "System:" prefix
            elif line.startswith("User:"):
                if current_role and current_content:
                    messages.append(
                        self._create_message(
                            current_role, "\n".join(current_content).strip()
                        )
                    )
                current_role = "user"
                current_content = [line[5:].strip()]  # Remove "User:" prefix
            elif line.startswith("Assistant:"):
                if current_role and current_content:
                    messages.append(
                        self._create_message(
                            current_role, "\n".join(current_content).strip()
                        )
                    )
                current_role = "assistant"
                current_content = [line[10:].strip()]  # Remove "Assistant:" prefix
            else:
                # Continue current message content
                if current_role:
                    current_content.append(line)
                else:
                    # No role prefix found, treat as user message
                    current_role = "user"
                    current_content = [line]

        # Add the last message
        if current_role and current_content:
            content_text = "\n".join(current_content).strip()
            if content_text:  # Only add if there's actual content
                messages.append(self._create_message(current_role, content_text))

        # If no messages were created, treat the entire content as a user message
        if not messages and content:
            messages.append(self._create_message("user", content))

        return messages

    def _create_message(self, role: str, content: str) -> AllMessageValues:
        """Create a message with the specified role and content."""
        return {
            "role": role,  # type: ignore
            "content": content,
        }

    def _extract_optional_params(self, template: PromptTemplate) -> dict:
        """
        Extract optional parameters from the prompt template metadata.

        Includes parameters like temperature, max_tokens, etc.
        """
        optional_params = {}

        # Extract common parameters from metadata
        if template.optional_params is not None:
            optional_params.update(template.optional_params)

        return optional_params

    def set_prompt_directory(self, prompt_directory: str) -> None:
        """Set the prompt directory and reload prompts."""
        self.prompt_directory = prompt_directory
        self._prompt_manager = None  # Reset to force reload

    def reload_prompts(self) -> None:
        """Reload all prompts from the directory."""
        if self._prompt_manager:
            self._prompt_manager.reload_prompts()

    def add_prompt_from_json(self, prompt_id: str, json_data: Dict[str, Any]) -> None:
        """Add a prompt from JSON data."""
        content = json_data.get("content", "")
        metadata = json_data.get("metadata", {})
        self.prompt_manager.add_prompt(prompt_id, content, metadata)

    def load_prompts_from_json(self, prompts_data: Dict[str, Dict[str, Any]]) -> None:
        """Load multiple prompts from JSON data."""
        self.prompt_manager.load_prompts_from_json_data(prompts_data)

    def get_prompts_as_json(self) -> Dict[str, Dict[str, Any]]:
        """Get all prompts in JSON format."""
        return self.prompt_manager.get_all_prompts_as_json()

    def convert_prompt_file_to_json(self, file_path: str) -> Dict[str, Any]:
        """Convert a .prompt file to JSON format."""
        return self.prompt_manager.prompt_file_to_json(file_path)
