"""
BitBucket prompt manager that integrates with LiteLLM's prompt management system.
Fetches .prompt files from BitBucket repositories and provides team-based access control.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from jinja2 import DictLoader, Environment, select_autoescape

from litellm.integrations.custom_prompt_management import CustomPromptManagement
from litellm.integrations.prompt_management_base import (
    PromptManagementBase,
    PromptManagementClient,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import StandardCallbackDynamicParams

from .bitbucket_client import BitBucketClient


class BitBucketPromptTemplate:
    """
    Represents a prompt template loaded from BitBucket.
    """

    def __init__(
        self,
        template_id: str,
        content: str,
        metadata: Dict[str, Any],
        model: Optional[str] = None,
    ):
        self.template_id = template_id
        self.content = content
        self.metadata = metadata
        self.model = model or metadata.get("model")
        self.temperature = metadata.get("temperature")
        self.max_tokens = metadata.get("max_tokens")
        self.input_schema = metadata.get("input", {}).get("schema", {})
        self.optional_params = {
            k: v for k, v in metadata.items() if k not in ["model", "input", "content"]
        }

    def __repr__(self):
        return f"BitBucketPromptTemplate(id='{self.template_id}', model='{self.model}')"


class BitBucketTemplateManager:
    """
    Manager for loading and rendering .prompt files from BitBucket repositories.

    Supports:
    - Fetching .prompt files from BitBucket repositories
    - Team-based access control through BitBucket permissions
    - YAML frontmatter for metadata
    - Handlebars-style templating (using Jinja2)
    - Input/output schema validation
    - Model configuration
    """

    def __init__(
        self,
        bitbucket_config: Dict[str, Any],
        prompt_id: Optional[str] = None,
    ):
        self.bitbucket_config = bitbucket_config
        self.prompt_id = prompt_id
        self.prompts: Dict[str, BitBucketPromptTemplate] = {}
        self.bitbucket_client = BitBucketClient(bitbucket_config)

        self.jinja_env = Environment(
            loader=DictLoader({}),
            autoescape=select_autoescape(["html", "xml"]),
            # Use Handlebars-style delimiters to match Dotprompt spec
            variable_start_string="{{",
            variable_end_string="}}",
            block_start_string="{%",
            block_end_string="%}",
            comment_start_string="{#",
            comment_end_string="#}",
        )

        # Load prompts from BitBucket if prompt_id is provided
        if self.prompt_id:
            self._load_prompt_from_bitbucket(self.prompt_id)

    def _load_prompt_from_bitbucket(self, prompt_id: str) -> None:
        """Load a specific .prompt file from BitBucket."""
        try:
            # Fetch the .prompt file from BitBucket
            prompt_content = self.bitbucket_client.get_file_content(
                f"{prompt_id}.prompt"
            )

            if prompt_content:
                template = self._parse_prompt_file(prompt_content, prompt_id)
                self.prompts[prompt_id] = template
        except Exception as e:
            raise Exception(f"Failed to load prompt '{prompt_id}' from BitBucket: {e}")

    def _parse_prompt_file(
        self, content: str, prompt_id: str
    ) -> BitBucketPromptTemplate:
        """Parse a .prompt file content and extract metadata and template."""
        # Split frontmatter and content
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter_str = parts[1].strip()
                template_content = parts[2].strip()
            else:
                frontmatter_str = ""
                template_content = content
        else:
            frontmatter_str = ""
            template_content = content

        # Parse YAML frontmatter
        metadata: Dict[str, Any] = {}
        if frontmatter_str:
            try:
                import yaml

                metadata = yaml.safe_load(frontmatter_str) or {}
            except ImportError:
                # Fallback to basic parsing if PyYAML is not available
                metadata = self._parse_yaml_basic(frontmatter_str)
            except Exception:
                metadata = {}

        return BitBucketPromptTemplate(
            template_id=prompt_id,
            content=template_content,
            metadata=metadata,
        )

    def _parse_yaml_basic(self, yaml_str: str) -> Dict[str, Any]:
        """Basic YAML parser for simple cases when PyYAML is not available."""
        result: Dict[str, Any] = {}
        for line in yaml_str.split("\n"):
            line = line.strip()
            if ":" in line and not line.startswith("#"):
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()

                # Try to parse value as appropriate type
                if value.lower() in ["true", "false"]:
                    result[key] = value.lower() == "true"
                elif value.isdigit():
                    result[key] = int(value)
                elif value.replace(".", "").isdigit():
                    result[key] = float(value)
                else:
                    result[key] = value.strip("\"'")
        return result

    def render_template(
        self, template_id: str, variables: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render a template with the given variables."""
        if template_id not in self.prompts:
            raise ValueError(f"Template '{template_id}' not found")

        template = self.prompts[template_id]
        jinja_template = self.jinja_env.from_string(template.content)

        return jinja_template.render(**(variables or {}))

    def get_template(self, template_id: str) -> Optional[BitBucketPromptTemplate]:
        """Get a template by ID."""
        return self.prompts.get(template_id)

    def list_templates(self) -> List[str]:
        """List all available template IDs."""
        return list(self.prompts.keys())


class BitBucketPromptManager(CustomPromptManagement):
    """
    BitBucket prompt manager that integrates with LiteLLM's prompt management system.

    This class enables using .prompt files from BitBucket repositories with the
    litellm completion() function by implementing the PromptManagementBase interface.

    Usage:
        # Configure BitBucket access
        bitbucket_config = {
            "workspace": "your-workspace",
            "repository": "your-repo",
            "access_token": "your-token",
            "branch": "main"  # optional, defaults to main
        }

        # Use with completion
        response = litellm.completion(
            model="bitbucket/gpt-4",
            prompt_id="my_prompt",
            prompt_variables={"variable": "value"},
            bitbucket_config=bitbucket_config,
            messages=[{"role": "user", "content": "This will be combined with the prompt"}]
        )
    """

    def __init__(
        self,
        bitbucket_config: Dict[str, Any],
        prompt_id: Optional[str] = None,
    ):
        self.bitbucket_config = bitbucket_config
        self.prompt_id = prompt_id
        self._prompt_manager: Optional[BitBucketTemplateManager] = None

    @property
    def integration_name(self) -> str:
        """Integration name used in model names like 'bitbucket/gpt-4'."""
        return "bitbucket"

    @property
    def prompt_manager(self) -> BitBucketTemplateManager:
        """Get or create the prompt manager instance."""
        if self._prompt_manager is None:
            self._prompt_manager = BitBucketTemplateManager(
                bitbucket_config=self.bitbucket_config,
                prompt_id=self.prompt_id,
            )
        return self._prompt_manager

    def get_prompt_template(
        self,
        prompt_id: str,
        prompt_variables: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Get a prompt template and render it with variables.

        Args:
            prompt_id: The ID of the prompt template
            prompt_variables: Variables to substitute in the template

        Returns:
            Tuple of (rendered_prompt, metadata)
        """
        template = self.prompt_manager.get_template(prompt_id)
        if not template:
            raise ValueError(f"Prompt template '{prompt_id}' not found")

        # Render the template
        rendered_prompt = self.prompt_manager.render_template(
            prompt_id, prompt_variables or {}
        )

        # Extract metadata
        metadata = {
            "model": template.model,
            "temperature": template.temperature,
            "max_tokens": template.max_tokens,
            **template.optional_params,
        }

        return rendered_prompt, metadata

    def pre_call_hook(
        self,
        user_id: Optional[str],
        messages: List[AllMessageValues],
        function_call: Optional[Union[Dict[str, Any], str]] = None,
        litellm_params: Optional[Dict[str, Any]] = None,
        prompt_id: Optional[str] = None,
        prompt_variables: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[List[AllMessageValues], Optional[Dict[str, Any]]]:
        """
        Pre-call hook that processes the prompt template before making the LLM call.
        """
        if not prompt_id:
            return messages, litellm_params

        try:
            # Get the rendered prompt and metadata
            rendered_prompt, prompt_metadata = self.get_prompt_template(
                prompt_id, prompt_variables
            )

            # Parse the rendered prompt into messages
            parsed_messages = self._parse_prompt_to_messages(rendered_prompt)

            # Merge with existing messages
            if parsed_messages:
                # If we have parsed messages, use them instead of the original messages
                final_messages: List[AllMessageValues] = parsed_messages
            else:
                # If no messages were parsed, prepend the prompt to existing messages
                final_messages = [
                    {"role": "user", "content": rendered_prompt}  # type: ignore
                ] + messages

            # Update litellm_params with prompt metadata
            if litellm_params is None:
                litellm_params = {}

            # Apply model and parameters from prompt metadata
            if prompt_metadata.get("model"):
                litellm_params["model"] = prompt_metadata["model"]

            for param in [
                "temperature",
                "max_tokens",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
            ]:
                if param in prompt_metadata:
                    litellm_params[param] = prompt_metadata[param]

            return final_messages, litellm_params

        except Exception as e:
            # Log error but don't fail the call
            import litellm

            litellm._logging.verbose_proxy_logger.error(
                f"Error in BitBucket prompt pre_call_hook: {e}"
            )
            return messages, litellm_params

    def _parse_prompt_to_messages(self, prompt_content: str) -> List[AllMessageValues]:
        """
        Parse prompt content into a list of messages.
        Handles both simple prompts and multi-role conversations.
        """
        messages = []
        lines = prompt_content.strip().split("\n")
        current_role = None
        current_content = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for role indicators
            if line.lower().startswith("system:"):
                if current_role and current_content:
                    messages.append(
                        {
                            "role": current_role,
                            "content": "\n".join(current_content).strip(),
                        }  # type: ignore
                    )
                current_role = "system"
                current_content = [line[7:].strip()]  # Remove "System:" prefix
            elif line.lower().startswith("user:"):
                if current_role and current_content:
                    messages.append(
                        {
                            "role": current_role,
                            "content": "\n".join(current_content).strip(),
                        }  # type: ignore
                    )
                current_role = "user"
                current_content = [line[5:].strip()]  # Remove "User:" prefix
            elif line.lower().startswith("assistant:"):
                if current_role and current_content:
                    messages.append(
                        {
                            "role": current_role,
                            "content": "\n".join(current_content).strip(),
                        }  # type: ignore
                    )
                current_role = "assistant"
                current_content = [line[10:].strip()]  # Remove "Assistant:" prefix
            else:
                # Continue building current message
                current_content.append(line)

        # Add the last message
        if current_role and current_content:
            messages.append(
                {"role": current_role, "content": "\n".join(current_content).strip()}
            )

        # If no role indicators found, treat as a single user message
        if not messages and prompt_content.strip():
            messages = [{"role": "user", "content": prompt_content.strip()}]  # type: ignore

        return messages  # type: ignore

    def post_call_hook(
        self,
        user_id: Optional[str],
        response: Any,
        input_messages: List[AllMessageValues],
        function_call: Optional[Union[Dict[str, Any], str]] = None,
        litellm_params: Optional[Dict[str, Any]] = None,
        prompt_id: Optional[str] = None,
        prompt_variables: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        """
        Post-call hook for any post-processing after the LLM call.
        """
        return response

    def get_available_prompts(self) -> List[str]:
        """Get list of available prompt IDs."""
        return self.prompt_manager.list_templates()

    def reload_prompts(self) -> None:
        """Reload prompts from BitBucket."""
        if self.prompt_id:
            self._prompt_manager = None  # Reset to force reload
            self.prompt_manager  # This will trigger reload

    def should_run_prompt_management(
        self,
        prompt_id: str,
        dynamic_callback_params: StandardCallbackDynamicParams,
    ) -> bool:
        """
        Determine if prompt management should run based on the prompt_id.

        For BitBucket, we always return True and handle the prompt loading
        in the _compile_prompt_helper method.
        """
        return True

    def _compile_prompt_helper(
        self,
        prompt_id: str,
        prompt_variables: Optional[dict],
        dynamic_callback_params: StandardCallbackDynamicParams,
        prompt_label: Optional[str] = None,
        prompt_version: Optional[int] = None,
    ) -> PromptManagementClient:
        """
        Compile a BitBucket prompt template into a PromptManagementClient structure.

        This method:
        1. Loads the prompt template from BitBucket
        2. Renders it with the provided variables
        3. Converts the rendered text into chat messages
        4. Extracts model and optional parameters from metadata
        """
        try:
            # Load the prompt from BitBucket if not already loaded
            if prompt_id not in self.prompt_manager.prompts:
                self.prompt_manager._load_prompt_from_bitbucket(prompt_id)

            # Get the rendered prompt and metadata
            rendered_prompt, prompt_metadata = self.get_prompt_template(
                prompt_id, prompt_variables
            )

            # Convert rendered content to chat messages
            messages = self._parse_prompt_to_messages(rendered_prompt)

            # Extract model from metadata (if specified)
            template_model = prompt_metadata.get("model")

            # Extract optional parameters from metadata
            optional_params = {}
            for param in [
                "temperature",
                "max_tokens",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
            ]:
                if param in prompt_metadata:
                    optional_params[param] = prompt_metadata[param]

            return PromptManagementClient(
                prompt_id=prompt_id,
                prompt_template=messages,
                prompt_template_model=template_model,
                prompt_template_optional_params=optional_params,
                completed_messages=None,
            )

        except Exception as e:
            raise ValueError(f"Error compiling prompt '{prompt_id}': {e}")

    def get_chat_completion_prompt(
        self,
        model: str,
        messages: List[AllMessageValues],
        non_default_params: dict,
        prompt_id: Optional[str],
        prompt_variables: Optional[dict],
        dynamic_callback_params: StandardCallbackDynamicParams,
        prompt_label: Optional[str] = None,
        prompt_version: Optional[int] = None,
    ) -> Tuple[str, List[AllMessageValues], dict]:
        """
        Get chat completion prompt from BitBucket and return processed model, messages, and parameters.
        """
        return PromptManagementBase.get_chat_completion_prompt(
            self,
            model,
            messages,
            non_default_params,
            prompt_id,
            prompt_variables,
            dynamic_callback_params,
            prompt_label,
            prompt_version,
        )
