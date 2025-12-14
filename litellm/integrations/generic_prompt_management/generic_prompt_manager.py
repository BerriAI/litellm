"""
Generic prompt manager that integrates with LiteLLM's prompt management system.
Fetches prompts from any API that implements the /beta/litellm_prompt_management endpoint.
"""

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import httpx

from litellm.integrations.custom_prompt_management import CustomPromptManagement
from litellm.integrations.prompt_management_base import (
    PromptManagementBase,
    PromptManagementClient,
)
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.types.llms.custom_http import httpxSpecialProvider
from litellm.types.llms.openai import AllMessageValues
from litellm.types.prompts.init_prompts import PromptSpec
from litellm.types.utils import StandardCallbackDynamicParams

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class GenericPromptManager(CustomPromptManagement):
    """
    Generic prompt manager that integrates with LiteLLM's prompt management system.

    This class enables using prompts from any API that implements the
    /beta/litellm_prompt_management endpoint.

    Usage:
        # Configure API access
        generic_config = {
            "api_base": "https://your-api.com",
            "api_key": "your-api-key",  # optional
            "timeout": 30,  # optional, defaults to 30
        }

        # Use with completion
        response = litellm.completion(
            model="generic_prompt/gpt-4",
            prompt_id="my_prompt_id",
            prompt_variables={"variable": "value"},
            generic_prompt_config=generic_config,
            messages=[{"role": "user", "content": "Additional message"}]
        )
    """

    def __init__(
        self,
        api_base: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        prompt_id: Optional[str] = None,
        additional_provider_specific_query_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize the Generic Prompt Manager.

        Args:
            api_base: Base URL for the API (e.g., "https://your-api.com")
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds (default: 30)
            prompt_id: Optional prompt ID to pre-load
        """
        super().__init__(**kwargs)
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.prompt_id = prompt_id
        self.additional_provider_specific_query_params = (
            additional_provider_specific_query_params
        )
        self._prompt_cache: Dict[str, PromptManagementClient] = {}

    @property
    def integration_name(self) -> str:
        """Integration name used in model names like 'generic_prompt/gpt-4'."""
        return "generic_prompt"

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _fetch_prompt_from_api(
        self, prompt_id: Optional[str], prompt_spec: Optional[PromptSpec]
    ) -> Dict[str, Any]:
        """
        Fetch a prompt from the API.

        Args:
            prompt_id: The ID of the prompt to fetch

        Returns:
            The prompt data from the API

        Raises:
            Exception: If the API request fails
        """
        if prompt_id is None and prompt_spec is None:
            raise ValueError("prompt_id or prompt_spec is required")

        url = f"{self.api_base}/beta/litellm_prompt_management"
        params = {
            "prompt_id": prompt_id,
            **(self.additional_provider_specific_query_params or {}),
        }
        http_client = _get_httpx_client()

        try:

            response = http_client.get(
                url,
                params=params,
                headers=self._get_headers(),
            )

            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise Exception(f"Failed to fetch prompt '{prompt_id}' from API: {e}")
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse prompt response for '{prompt_id}': {e}")

    async def async_fetch_prompt_from_api(
        self, prompt_id: Optional[str], prompt_spec: Optional[PromptSpec]
    ) -> Dict[str, Any]:
        """
        Fetch a prompt from the API asynchronously.
        """
        if prompt_id is None and prompt_spec is None:
            raise ValueError("prompt_id or prompt_spec is required")

        url = f"{self.api_base}/beta/litellm_prompt_management"
        params = {
            "prompt_id": prompt_id,
            **(
                prompt_spec.litellm_params.provider_specific_query_params
                if prompt_spec
                and prompt_spec.litellm_params.provider_specific_query_params
                else {}
            ),
        }

        http_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.PromptManagement,
        )

        try:
            response = await http_client.get(
                url,
                params=params,
                headers=self._get_headers(),
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise Exception(f"Failed to fetch prompt '{prompt_id}' from API: {e}")
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse prompt response for '{prompt_id}': {e}")

    def _parse_api_response(
        self,
        prompt_id: Optional[str],
        prompt_spec: Optional[PromptSpec],
        api_response: Dict[str, Any],
    ) -> PromptManagementClient:
        """
        Parse the API response into a PromptManagementClient structure.

        Expected API response format:
        {
            "prompt_id": "string",
            "prompt_template": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."}
            ],
            "prompt_template_model": "gpt-4",  # optional
            "prompt_template_optional_params": {  # optional
                "temperature": 0.7,
                "max_tokens": 100
            }
        }

        Args:
            prompt_id: The ID of the prompt
            api_response: The response from the API

        Returns:
            PromptManagementClient structure
        """
        return PromptManagementClient(
            prompt_id=prompt_id,
            prompt_template=api_response.get("prompt_template", []),
            prompt_template_model=api_response.get("prompt_template_model"),
            prompt_template_optional_params=api_response.get(
                "prompt_template_optional_params"
            ),
            completed_messages=None,
        )

    def should_run_prompt_management(
        self,
        prompt_id: Optional[str],
        prompt_spec: Optional[PromptSpec],
        dynamic_callback_params: StandardCallbackDynamicParams,
    ) -> bool:
        """
        Determine if prompt management should run based on the prompt_id.

        For Generic Prompt Manager, we always return True and handle the prompt loading
        in the _compile_prompt_helper method.
        """
        if prompt_id is not None or (
            prompt_spec is not None
            and prompt_spec.litellm_params.provider_specific_query_params is not None
        ):
            return True
        return False

    def _get_cache_key(
        self,
        prompt_id: Optional[str],
        prompt_label: Optional[str] = None,
        prompt_version: Optional[int] = None,
    ) -> str:
        return f"{prompt_id}:{prompt_label}:{prompt_version}"

    def _common_caching_logic(
        self,
        prompt_id: Optional[str],
        prompt_label: Optional[str] = None,
        prompt_version: Optional[int] = None,
        prompt_variables: Optional[dict] = None,
    ) -> Optional[PromptManagementClient]:
        """
        Common caching logic for the prompt manager.
        """
        # Check cache first
        cache_key = self._get_cache_key(prompt_id, prompt_label, prompt_version)
        if cache_key in self._prompt_cache:
            cached_prompt = self._prompt_cache[cache_key]
            # Return a copy with variables applied if needed
            if prompt_variables:
                return self._apply_variables(cached_prompt, prompt_variables)
            return cached_prompt
        return None

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
        Compile a prompt template into a PromptManagementClient structure.

        This method:
        1. Fetches the prompt from the API (with caching)
        2. Applies any prompt variables (if the API supports it)
        3. Returns the structured prompt data

        Args:
            prompt_id: The ID of the prompt
            prompt_variables: Variables to substitute in the template (optional)
            dynamic_callback_params: Dynamic callback parameters
            prompt_label: Optional label for the prompt version
            prompt_version: Optional specific version number

        Returns:
            PromptManagementClient structure
        """
        cached_prompt = self._common_caching_logic(
            prompt_id=prompt_id,
            prompt_label=prompt_label,
            prompt_version=prompt_version,
            prompt_variables=prompt_variables,
        )
        if cached_prompt:
            return cached_prompt

        cache_key = self._get_cache_key(prompt_id, prompt_label, prompt_version)
        try:
            # Fetch from API
            api_response = self._fetch_prompt_from_api(prompt_id, prompt_spec)

            # Parse the response
            prompt_client = self._parse_api_response(
                prompt_id, prompt_spec, api_response
            )

            # Cache the result
            self._prompt_cache[cache_key] = prompt_client

            # Apply variables if provided
            if prompt_variables:
                prompt_client = self._apply_variables(prompt_client, prompt_variables)

            return prompt_client

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

        # Check cache first
        cached_prompt = self._common_caching_logic(
            prompt_id=prompt_id,
            prompt_label=prompt_label,
            prompt_version=prompt_version,
            prompt_variables=prompt_variables,
        )
        if cached_prompt:
            return cached_prompt

        cache_key = self._get_cache_key(prompt_id, prompt_label, prompt_version)

        try:
            # Fetch from API

            api_response = await self.async_fetch_prompt_from_api(
                prompt_id=prompt_id, prompt_spec=prompt_spec
            )

            # Parse the response
            prompt_client = self._parse_api_response(
                prompt_id, prompt_spec, api_response
            )

            # Cache the result
            self._prompt_cache[cache_key] = prompt_client

            # Apply variables if provided
            if prompt_variables:
                prompt_client = self._apply_variables(prompt_client, prompt_variables)

            return prompt_client

        except Exception as e:
            raise ValueError(
                f"Error compiling prompt '{prompt_id}': {e}, prompt_spec: {prompt_spec}"
            )

    def _apply_variables(
        self,
        prompt_client: PromptManagementClient,
        variables: Dict[str, Any],
    ) -> PromptManagementClient:
        """
        Apply variables to the prompt template.

        This performs simple string substitution using {variable_name} syntax.

        Args:
            prompt_client: The prompt client structure
            variables: Variables to substitute

        Returns:
            Updated PromptManagementClient with variables applied
        """
        # Create a copy of the prompt template with variables applied
        updated_messages: List[AllMessageValues] = []
        for message in prompt_client["prompt_template"]:
            updated_message = dict(message)  # type: ignore
            if "content" in updated_message and isinstance(
                updated_message["content"], str
            ):
                content = updated_message["content"]
                for key, value in variables.items():
                    content = content.replace(f"{{{key}}}", str(value))
                    content = content.replace(
                        f"{{{{{key}}}}}", str(value)
                    )  # Also support {{key}}
                updated_message["content"] = content
            updated_messages.append(updated_message)  # type: ignore

        return PromptManagementClient(
            prompt_id=prompt_client["prompt_id"],
            prompt_template=updated_messages,
            prompt_template_model=prompt_client["prompt_template_model"],
            prompt_template_optional_params=prompt_client[
                "prompt_template_optional_params"
            ],
            completed_messages=None,
        )

    async def async_get_chat_completion_prompt(
        self,
        model: str,
        messages: List[AllMessageValues],
        non_default_params: dict,
        prompt_id: Optional[str],
        prompt_variables: Optional[dict],
        dynamic_callback_params: StandardCallbackDynamicParams,
        litellm_logging_obj: "LiteLLMLoggingObj",
        prompt_spec: Optional[PromptSpec] = None,
        tools: Optional[List[Dict]] = None,
        prompt_label: Optional[str] = None,
        prompt_version: Optional[int] = None,
        ignore_prompt_manager_model: Optional[bool] = False,
        ignore_prompt_manager_optional_params: Optional[bool] = False,
    ) -> Tuple[str, List[AllMessageValues], dict]:
        """
        Get chat completion prompt and return processed model, messages, and parameters.
        """

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
            ignore_prompt_manager_model=(
                ignore_prompt_manager_model
                or prompt_spec.litellm_params.ignore_prompt_manager_model
                if prompt_spec
                else False
            ),
            ignore_prompt_manager_optional_params=(
                ignore_prompt_manager_optional_params
                or prompt_spec.litellm_params.ignore_prompt_manager_optional_params
                if prompt_spec
                else False
            ),
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
        Get chat completion prompt and return processed model, messages, and parameters.
        """
        return PromptManagementBase.get_chat_completion_prompt(
            self,
            model,
            messages,
            non_default_params,
            prompt_id=prompt_id,
            prompt_variables=prompt_variables,
            dynamic_callback_params=dynamic_callback_params,
            prompt_spec=prompt_spec,
            prompt_label=prompt_label,
            prompt_version=prompt_version,
            ignore_prompt_manager_model=(
                ignore_prompt_manager_model
                or prompt_spec.litellm_params.ignore_prompt_manager_model
                if prompt_spec
                else False
            ),
            ignore_prompt_manager_optional_params=(
                ignore_prompt_manager_optional_params
                or prompt_spec.litellm_params.ignore_prompt_manager_optional_params
                if prompt_spec
                else False
            ),
        )

    def clear_cache(self) -> None:
        """Clear the prompt cache."""
        self._prompt_cache.clear()
