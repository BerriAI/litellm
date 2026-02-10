"""
Qualifire prompt manager that integrates with LiteLLM's prompt management system.
Fetches compiled prompts from Qualifire Studio's /compile endpoint.
"""

from typing import Any, Dict, List, Optional, Tuple

from litellm.integrations.custom_prompt_management import CustomPromptManagement
from litellm.integrations.prompt_management_base import (
    PromptManagementBase,
    PromptManagementClient,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.prompts.init_prompts import PromptSpec
from litellm.types.utils import StandardCallbackDynamicParams

from .qualifire_client import QualifireClient

# Parameters to extract from the Qualifire response
SUPPORTED_PARAMETERS = [
    "temperature",
    "top_p",
    "max_tokens",
    "frequency_penalty",
    "presence_penalty",
    "reasoning_effort",
]


class QualifirePromptManager(CustomPromptManagement):
    """
    Qualifire prompt manager that integrates with LiteLLM's prompt management system.

    Uses Qualifire's /compile endpoint which handles variable substitution server-side,
    so no client-side templating is needed. Variables are POSTed to the API and compiled
    messages, tools, and parameters are returned.
    """

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.qualifire.ai",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.api_base = api_base
        self.client = QualifireClient(api_key=api_key, api_base=api_base)

    @property
    def integration_name(self) -> str:
        return "qualifire"

    def should_run_prompt_management(
        self,
        prompt_id: Optional[str],
        prompt_spec: Optional[PromptSpec],
        dynamic_callback_params: StandardCallbackDynamicParams,
    ) -> bool:
        return True

    @staticmethod
    def _extract_revision(prompt_spec: Optional[PromptSpec]) -> Optional[str]:
        """Extract revision from prompt_spec's provider_specific_query_params."""
        if (
            prompt_spec
            and prompt_spec.litellm_params.provider_specific_query_params
        ):
            return prompt_spec.litellm_params.provider_specific_query_params.get(
                "revision"
            )
        return None

    @staticmethod
    def _parse_compile_response(
        prompt_id: Optional[str],
        response: Dict[str, Any],
    ) -> PromptManagementClient:
        """
        Parse the Qualifire compile response into a PromptManagementClient.

        Qualifire compile response format:
        {
            "messages": [...],
            "parameters": {
                "model": "gpt-4",
                "temperature": 0.7,
                ...
            },
            "tools": [...]  # optional
        }
        """
        messages = response.get("messages", [])
        parameters = response.get("parameters", {})
        tools = response.get("tools")

        # Extract model from parameters
        model = parameters.get("model")

        # Extract optional params, filtering out None values
        optional_params: Dict[str, Any] = {}
        for param in SUPPORTED_PARAMETERS:
            value = parameters.get(param)
            if value is not None:
                optional_params[param] = value

        # Include tools if present and non-empty
        if tools:
            optional_params["tools"] = tools

        return PromptManagementClient(
            prompt_id=prompt_id,
            prompt_template=messages,
            prompt_template_model=model,
            prompt_template_optional_params=optional_params if optional_params else None,
            completed_messages=None,
        )

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
        Sync wrapper that delegates to the async compile prompt helper.
        """
        import asyncio

        return asyncio.run(
            self.async_compile_prompt_helper(
                prompt_id=prompt_id,
                prompt_variables=prompt_variables,
                dynamic_callback_params=dynamic_callback_params,
                prompt_spec=prompt_spec,
                prompt_label=prompt_label,
                prompt_version=prompt_version,
            )
        )

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
        Compile a prompt using the Qualifire /compile endpoint (async).
        """
        if prompt_id is None:
            raise ValueError("prompt_id is required for Qualifire prompt manager")

        revision = self._extract_revision(prompt_spec)

        try:
            response = await self.client.compile_prompt(
                prompt_id=prompt_id,
                variables=prompt_variables,
                revision=revision,
            )
            return self._parse_compile_response(prompt_id, response)
        except Exception as e:
            raise ValueError(f"Error compiling prompt '{prompt_id}' from Qualifire: {e}")

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
                or (
                    prompt_spec.litellm_params.ignore_prompt_manager_model
                    if prompt_spec
                    else False
                )
            ),
            ignore_prompt_manager_optional_params=(
                ignore_prompt_manager_optional_params
                or (
                    prompt_spec.litellm_params.ignore_prompt_manager_optional_params
                    if prompt_spec
                    else False
                )
            ),
        )

    async def async_get_chat_completion_prompt(
        self,
        model: str,
        messages: List[AllMessageValues],
        non_default_params: dict,
        prompt_id: Optional[str],
        prompt_variables: Optional[dict],
        dynamic_callback_params: StandardCallbackDynamicParams,
        litellm_logging_obj: Any = None,
        prompt_spec: Optional[PromptSpec] = None,
        tools: Optional[List[Dict]] = None,
        prompt_label: Optional[str] = None,
        prompt_version: Optional[int] = None,
        ignore_prompt_manager_model: Optional[bool] = False,
        ignore_prompt_manager_optional_params: Optional[bool] = False,
    ) -> Tuple[str, List[AllMessageValues], dict]:
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
                or (
                    prompt_spec.litellm_params.ignore_prompt_manager_model
                    if prompt_spec
                    else False
                )
            ),
            ignore_prompt_manager_optional_params=(
                ignore_prompt_manager_optional_params
                or (
                    prompt_spec.litellm_params.ignore_prompt_manager_optional_params
                    if prompt_spec
                    else False
                )
            ),
        )
