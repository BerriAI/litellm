"""
Humanloop integration

https://humanloop.com/
"""

from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union, cast

import httpx

import litellm
from litellm.caching import DualCache
from litellm.llms.custom_httpx.http_handler import _get_httpx_client
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import StandardCallbackDynamicParams

from .custom_logger import CustomLogger


class PromptManagementClient(TypedDict):
    prompt_id: str
    prompt_template: List[AllMessageValues]
    model: Optional[str]
    optional_params: Optional[Dict[str, Any]]


class HumanLoopPromptManager(DualCache):
    @property
    def integration_name(self):
        return "humanloop"

    def _get_prompt_from_id_cache(
        self, humanloop_prompt_id: str
    ) -> Optional[PromptManagementClient]:
        return cast(
            Optional[PromptManagementClient], self.get_cache(key=humanloop_prompt_id)
        )

    def _compile_prompt_helper(
        self, prompt_template: List[AllMessageValues], prompt_variables: Dict[str, Any]
    ) -> List[AllMessageValues]:
        """
        Helper function to compile the prompt by substituting variables in the template.

        Args:
            prompt_template: List[AllMessageValues]
            prompt_variables (dict): A dictionary of variables to substitute into the prompt template.

        Returns:
            list: A list of dictionaries with variables substituted.
        """
        compiled_prompts: List[AllMessageValues] = []

        for template in prompt_template:
            tc = template.get("content")
            if tc and isinstance(tc, str):
                formatted_template = tc.replace("{{", "{").replace("}}", "}")
                compiled_content = formatted_template.format(**prompt_variables)
                template["content"] = compiled_content
            compiled_prompts.append(template)

        return compiled_prompts

    def _get_prompt_from_id_api(
        self, humanloop_prompt_id: str, humanloop_api_key: str
    ) -> PromptManagementClient:
        client = _get_httpx_client()

        base_url = "https://api.humanloop.com/v5/prompts/{}".format(humanloop_prompt_id)

        response = client.get(
            url=base_url,
            headers={
                "X-Api-Key": humanloop_api_key,
                "Content-Type": "application/json",
            },
        )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise Exception(f"Error getting prompt from Humanloop: {e.response.text}")

        json_response = response.json()
        template_message = json_response["template"]
        if isinstance(template_message, dict):
            template_messages = [template_message]
        elif isinstance(template_message, list):
            template_messages = template_message
        else:
            raise ValueError(f"Invalid template message type: {type(template_message)}")
        template_model = json_response["model"]
        optional_params = {}
        for k, v in json_response.items():
            if k in litellm.OPENAI_CHAT_COMPLETION_PARAMS:
                optional_params[k] = v
        return PromptManagementClient(
            prompt_id=humanloop_prompt_id,
            prompt_template=cast(List[AllMessageValues], template_messages),
            model=template_model,
            optional_params=optional_params,
        )

    def _get_prompt_from_id(
        self, humanloop_prompt_id: str, humanloop_api_key: str
    ) -> PromptManagementClient:
        prompt = self._get_prompt_from_id_cache(humanloop_prompt_id)
        if prompt is None:
            prompt = self._get_prompt_from_id_api(
                humanloop_prompt_id, humanloop_api_key
            )
            self.set_cache(
                key=humanloop_prompt_id,
                value=prompt,
                ttl=litellm.HUMANLOOP_PROMPT_CACHE_TTL_SECONDS,
            )
        return prompt

    def compile_prompt(
        self,
        prompt_template: List[AllMessageValues],
        prompt_variables: Optional[dict],
    ) -> List[AllMessageValues]:
        compiled_prompt: Optional[Union[str, list]] = None

        if prompt_variables is None:
            prompt_variables = {}

        compiled_prompt = self._compile_prompt_helper(
            prompt_template=prompt_template,
            prompt_variables=prompt_variables,
        )

        return compiled_prompt

    def _get_model_from_prompt(
        self, prompt_management_client: PromptManagementClient, model: str
    ) -> str:
        if prompt_management_client["model"] is not None:
            return prompt_management_client["model"]
        else:
            return model.replace("{}/".format(self.integration_name), "")


prompt_manager = HumanLoopPromptManager()


class HumanloopLogger(CustomLogger):
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
    ) -> Tuple[
        str,
        List[AllMessageValues],
        dict,
    ]:
        humanloop_api_key = dynamic_callback_params.get(
            "humanloop_api_key"
        ) or get_secret_str("HUMANLOOP_API_KEY")

        if prompt_id is None:
            raise ValueError("prompt_id is required for Humanloop integration")

        if humanloop_api_key is None:
            return super().get_chat_completion_prompt(
                model=model,
                messages=messages,
                non_default_params=non_default_params,
                prompt_id=prompt_id,
                prompt_variables=prompt_variables,
                dynamic_callback_params=dynamic_callback_params,
            )

        prompt_template = prompt_manager._get_prompt_from_id(
            humanloop_prompt_id=prompt_id, humanloop_api_key=humanloop_api_key
        )

        updated_messages = prompt_manager.compile_prompt(
            prompt_template=prompt_template["prompt_template"],
            prompt_variables=prompt_variables,
        )

        prompt_template_optional_params = prompt_template["optional_params"] or {}

        updated_non_default_params = {
            **non_default_params,
            **prompt_template_optional_params,
        }

        model = prompt_manager._get_model_from_prompt(
            prompt_management_client=prompt_template, model=model
        )

        return model, updated_messages, updated_non_default_params
