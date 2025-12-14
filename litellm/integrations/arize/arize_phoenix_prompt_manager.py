"""
Arize Phoenix prompt manager that integrates with LiteLLM's prompt management system.
Fetches prompt versions from Arize Phoenix and provides workspace-based access control.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from jinja2 import DictLoader, Environment, select_autoescape

from litellm.integrations.custom_prompt_management import CustomPromptManagement
from litellm.integrations.prompt_management_base import (
    PromptManagementBase,
    PromptManagementClient,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.prompts.init_prompts import PromptSpec
from litellm.types.utils import StandardCallbackDynamicParams

from .arize_phoenix_client import ArizePhoenixClient


class ArizePhoenixPromptTemplate:
    """
    Represents a prompt template loaded from Arize Phoenix.
    """

    def __init__(
        self,
        template_id: str,
        messages: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        model: Optional[str] = None,
    ):
        self.template_id = template_id
        self.messages = messages
        self.metadata = metadata
        self.model = model or metadata.get("model_name")
        self.model_provider = metadata.get("model_provider")
        self.temperature = metadata.get("temperature")
        self.max_tokens = metadata.get("max_tokens")
        self.invocation_parameters = metadata.get("invocation_parameters", {})
        self.description = metadata.get("description", "")
        self.template_format = metadata.get("template_format", "MUSTACHE")

    def __repr__(self):
        return (
            f"ArizePhoenixPromptTemplate(id='{self.template_id}', model='{self.model}')"
        )


class ArizePhoenixTemplateManager:
    """
    Manager for loading and rendering prompt templates from Arize Phoenix.

    Supports:
    - Fetching prompt versions from Arize Phoenix API
    - Workspace-based access control through Arize Phoenix permissions
    - Mustache/Handlebars-style templating (using Jinja2)
    - Model configuration and invocation parameters
    - Multi-message chat templates
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        prompt_id: Optional[str] = None,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.prompt_id = prompt_id
        self.prompts: Dict[str, ArizePhoenixPromptTemplate] = {}
        self.arize_client = ArizePhoenixClient(
            api_key=self.api_key, api_base=self.api_base
        )

        self.jinja_env = Environment(
            loader=DictLoader({}),
            autoescape=select_autoescape(["html", "xml"]),
            # Use Mustache/Handlebars-style delimiters
            variable_start_string="{{",
            variable_end_string="}}",
            block_start_string="{%",
            block_end_string="%}",
            comment_start_string="{#",
            comment_end_string="#}",
        )

        # Load prompt from Arize Phoenix if prompt_id is provided
        if self.prompt_id:
            self._load_prompt_from_arize(self.prompt_id)

    def _load_prompt_from_arize(self, prompt_version_id: str) -> None:
        """Load a specific prompt version from Arize Phoenix."""
        try:
            # Fetch the prompt version from Arize Phoenix
            prompt_data = self.arize_client.get_prompt_version(prompt_version_id)

            if prompt_data:
                template = self._parse_prompt_data(prompt_data, prompt_version_id)
                self.prompts[prompt_version_id] = template
            else:
                raise ValueError(f"Prompt version '{prompt_version_id}' not found")
        except Exception as e:
            raise Exception(
                f"Failed to load prompt version '{prompt_version_id}' from Arize Phoenix: {e}"
            )

    def _parse_prompt_data(
        self, data: Dict[str, Any], prompt_version_id: str
    ) -> ArizePhoenixPromptTemplate:
        """Parse Arize Phoenix prompt data and extract messages and metadata."""
        template_data = data.get("template", {})
        messages = template_data.get("messages", [])

        # Extract invocation parameters
        invocation_params = data.get("invocation_parameters", {})
        provider_params = {}

        # Extract provider-specific parameters
        if "openai" in invocation_params:
            provider_params = invocation_params["openai"]
        elif "anthropic" in invocation_params:
            provider_params = invocation_params["anthropic"]
        else:
            # Try to find any nested provider params
            for key, value in invocation_params.items():
                if isinstance(value, dict):
                    provider_params = value
                    break

        # Build metadata dictionary
        metadata = {
            "model_name": data.get("model_name"),
            "model_provider": data.get("model_provider"),
            "description": data.get("description", ""),
            "template_type": data.get("template_type"),
            "template_format": data.get("template_format", "MUSTACHE"),
            "invocation_parameters": invocation_params,
            "temperature": provider_params.get("temperature"),
            "max_tokens": provider_params.get("max_tokens"),
        }

        return ArizePhoenixPromptTemplate(
            template_id=prompt_version_id,
            messages=messages,
            metadata=metadata,
        )

    def render_template(
        self, template_id: str, variables: Optional[Dict[str, Any]] = None
    ) -> List[AllMessageValues]:
        """Render a template with the given variables and return formatted messages."""
        if template_id not in self.prompts:
            raise ValueError(f"Template '{template_id}' not found")

        template = self.prompts[template_id]
        rendered_messages: List[AllMessageValues] = []

        for message in template.messages:
            role = message.get("role", "user")
            content_parts = message.get("content", [])

            # Render each content part
            rendered_content_parts = []
            for part in content_parts:
                if part.get("type") == "text":
                    text = part.get("text", "")
                    # Render the text with Jinja2 (Mustache-style)
                    jinja_template = self.jinja_env.from_string(text)
                    rendered_text = jinja_template.render(**(variables or {}))
                    rendered_content_parts.append(rendered_text)
                else:
                    # Handle other content types if needed
                    rendered_content_parts.append(part)

            # Combine rendered content
            final_content = " ".join(rendered_content_parts)

            rendered_messages.append(
                {"role": role, "content": final_content}  # type: ignore
            )

        return rendered_messages

    def get_template(self, template_id: str) -> Optional[ArizePhoenixPromptTemplate]:
        """Get a template by ID."""
        return self.prompts.get(template_id)

    def list_templates(self) -> List[str]:
        """List all available template IDs."""
        return list(self.prompts.keys())


class ArizePhoenixPromptManager(CustomPromptManagement):
    """
    Arize Phoenix prompt manager that integrates with LiteLLM's prompt management system.

    This class enables using prompt versions from Arize Phoenix with the
    litellm completion() function by implementing the PromptManagementBase interface.

    Usage:
        # Configure Arize Phoenix access
        arize_config = {
            "workspace": "your-workspace",
            "access_token": "your-token",
        }

        # Use with completion
        response = litellm.completion(
            model="arize/gpt-4o",
            prompt_id="UHJvbXB0VmVyc2lvbjox",
            prompt_variables={"question": "What is AI?"},
            arize_config=arize_config,
            messages=[{"role": "user", "content": "This will be combined with the prompt"}]
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        prompt_id: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.api_base = api_base
        self.prompt_id = prompt_id
        self._prompt_manager: Optional[ArizePhoenixTemplateManager] = None

    @property
    def integration_name(self) -> str:
        """Integration name used in model names like 'arize/gpt-4o'."""
        return "arize"

    @property
    def prompt_manager(self) -> ArizePhoenixTemplateManager:
        """Get or create the prompt manager instance."""
        if self._prompt_manager is None:
            self._prompt_manager = ArizePhoenixTemplateManager(
                api_key=self.api_key,
                api_base=self.api_base,
                prompt_id=self.prompt_id,
            )
        return self._prompt_manager

    def get_prompt_template(
        self,
        prompt_id: str,
        prompt_variables: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[AllMessageValues], Dict[str, Any]]:
        """
        Get a prompt template and render it with variables.

        Args:
            prompt_id: The ID of the prompt version
            prompt_variables: Variables to substitute in the template

        Returns:
            Tuple of (rendered_messages, metadata)
        """
        template = self.prompt_manager.get_template(prompt_id)
        if not template:
            raise ValueError(f"Prompt template '{prompt_id}' not found")

        # Render the template
        rendered_messages = self.prompt_manager.render_template(
            prompt_id, prompt_variables or {}
        )

        # Extract metadata
        metadata = {
            "model": template.model,
            "temperature": template.temperature,
            "max_tokens": template.max_tokens,
        }

        # Add additional invocation parameters
        invocation_params = template.invocation_parameters
        provider_params = {}

        if "openai" in invocation_params:
            provider_params = invocation_params["openai"]
        elif "anthropic" in invocation_params:
            provider_params = invocation_params["anthropic"]

        # Add any additional parameters
        for key, value in provider_params.items():
            if key not in metadata:
                metadata[key] = value

        return rendered_messages, metadata

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
            # Get the rendered messages and metadata
            rendered_messages, prompt_metadata = self.get_prompt_template(
                prompt_id, prompt_variables
            )

            # Merge rendered messages with existing messages
            if rendered_messages:
                # Prepend rendered messages to existing messages
                final_messages = rendered_messages + messages
            else:
                final_messages = messages

            # Update litellm_params with prompt metadata
            if litellm_params is None:
                litellm_params = {}

            # Apply model and parameters from prompt metadata
            if prompt_metadata.get("model") and not self.ignore_prompt_manager_model:
                litellm_params["model"] = prompt_metadata["model"]

            if not self.ignore_prompt_manager_optional_params:
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
                f"Error in Arize Phoenix prompt pre_call_hook: {e}"
            )
            return messages, litellm_params

    def get_available_prompts(self) -> List[str]:
        """Get list of available prompt IDs."""
        return self.prompt_manager.list_templates()

    def reload_prompts(self) -> None:
        """Reload prompts from Arize Phoenix."""
        if self.prompt_id:
            self._prompt_manager = None  # Reset to force reload
            self.prompt_manager  # This will trigger reload

    def should_run_prompt_management(
        self,
        prompt_id: Optional[str],
        prompt_spec: Optional[PromptSpec],
        dynamic_callback_params: StandardCallbackDynamicParams,
    ) -> bool:
        """
        Determine if prompt management should run based on the prompt_id.

        For Arize Phoenix, we always return True and handle the prompt loading
        in the _compile_prompt_helper method.
        """
        return True

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
        Compile an Arize Phoenix prompt template into a PromptManagementClient structure.

        This method:
        1. Loads the prompt version from Arize Phoenix
        2. Renders it with the provided variables
        3. Returns formatted chat messages
        4. Extracts model and optional parameters from metadata
        """
        if prompt_id is None:
            raise ValueError("prompt_id is required for Arize Phoenix prompt manager")
        try:
            # Load the prompt from Arize Phoenix if not already loaded
            if prompt_id not in self.prompt_manager.prompts:
                self.prompt_manager._load_prompt_from_arize(prompt_id)

            # Get the rendered messages and metadata
            rendered_messages, prompt_metadata = self.get_prompt_template(
                prompt_id, prompt_variables
            )

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
                prompt_template=rendered_messages,
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
        Async version of compile prompt helper. Since Arize Phoenix operations are synchronous,
        this simply delegates to the sync version.
        """
        if prompt_id is None:
            raise ValueError("prompt_id is required for Arize Phoenix prompt manager")
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
        """
        Get chat completion prompt from Arize Phoenix and return processed model, messages, and parameters.
        """
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
            ignore_prompt_manager_model=ignore_prompt_manager_model,
            ignore_prompt_manager_optional_params=ignore_prompt_manager_optional_params,
        )
