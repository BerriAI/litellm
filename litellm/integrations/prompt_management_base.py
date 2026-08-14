from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Final

from typing_extensions import TypedDict

from litellm.types.llms.openai import AllMessageValues
from litellm.types.prompts.init_prompts import PromptSpec
from litellm.types.utils import StandardCallbackDynamicParams

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class PromptManagementClient(TypedDict):
    prompt_id: str | None
    prompt_template: list[AllMessageValues]
    prompt_template_model: str | None
    prompt_template_optional_params: dict[str, Any] | None
    completed_messages: list[AllMessageValues] | None


class PromptManagementBase(ABC):
    @property
    @abstractmethod
    def integration_name(self) -> str:
        pass

    @abstractmethod
    def should_run_prompt_management(
        self,
        prompt_id: str | None,
        prompt_spec: PromptSpec | None,
        dynamic_callback_params: StandardCallbackDynamicParams,
    ) -> bool:
        pass

    @abstractmethod
    def _compile_prompt_helper(
        self,
        prompt_id: str | None,
        prompt_spec: PromptSpec | None,
        prompt_variables: dict | None,
        dynamic_callback_params: StandardCallbackDynamicParams,
        prompt_label: str | None = None,
        prompt_version: int | None = None,
    ) -> PromptManagementClient:
        pass

    @abstractmethod
    async def async_compile_prompt_helper(
        self,
        prompt_id: str | None,
        prompt_variables: dict | None,
        dynamic_callback_params: StandardCallbackDynamicParams,
        prompt_spec: PromptSpec | None = None,
        prompt_label: str | None = None,
        prompt_version: int | None = None,
    ) -> PromptManagementClient:
        pass

    def merge_messages(
        self,
        prompt_template: list[AllMessageValues],
        client_messages: list[AllMessageValues],
    ) -> list[AllMessageValues]:
        return prompt_template + client_messages

    def compile_prompt(
        self,
        prompt_id: str,
        prompt_variables: dict | None,
        client_messages: list[AllMessageValues],
        dynamic_callback_params: StandardCallbackDynamicParams,
        prompt_label: str | None = None,
        prompt_version: int | None = None,
        prompt_spec: PromptSpec | None = None,
    ) -> PromptManagementClient:
        compiled_prompt_client: Final = self._compile_prompt_helper(
            prompt_id=prompt_id,
            prompt_spec=prompt_spec,
            prompt_variables=prompt_variables,
            dynamic_callback_params=dynamic_callback_params,
            prompt_label=prompt_label,
            prompt_version=prompt_version,
        )

        try:
            messages: Final = compiled_prompt_client["prompt_template"] + client_messages
        except Exception as e:
            raise ValueError(f"Error compiling prompt: {e}. Prompt id={prompt_id}")

        compiled_prompt_client["completed_messages"] = messages
        return compiled_prompt_client

    async def async_compile_prompt(
        self,
        prompt_id: str | None,
        prompt_variables: dict | None,
        client_messages: list[AllMessageValues],
        dynamic_callback_params: StandardCallbackDynamicParams,
        prompt_spec: PromptSpec | None = None,
        prompt_label: str | None = None,
        prompt_version: int | None = None,
    ) -> PromptManagementClient:
        compiled_prompt_client: Final = await self.async_compile_prompt_helper(
            prompt_id=prompt_id,
            prompt_spec=prompt_spec,
            prompt_variables=prompt_variables,
            dynamic_callback_params=dynamic_callback_params,
            prompt_label=prompt_label,
            prompt_version=prompt_version,
        )

        try:
            messages: Final = compiled_prompt_client["prompt_template"] + client_messages
        except Exception as e:
            raise ValueError(f"Error compiling prompt: {e}. Prompt id={prompt_id}")

        compiled_prompt_client["completed_messages"] = messages
        return compiled_prompt_client

    def _get_model_from_prompt(self, prompt_management_client: PromptManagementClient, model: str) -> str:
        if prompt_management_client["prompt_template_model"] is not None:
            return prompt_management_client["prompt_template_model"]
        else:
            return model.replace(f"{self.integration_name}/", "")

    def post_compile_prompt_processing(
        self,
        prompt_template: PromptManagementClient,
        messages: list[AllMessageValues],
        non_default_params: dict,
        model: str,
        ignore_prompt_manager_model: bool | None = False,
        ignore_prompt_manager_optional_params: bool | None = False,
    ):
        completed_messages: Final = prompt_template["completed_messages"] or messages

        prompt_template_optional_params: Final = prompt_template["prompt_template_optional_params"] or {}

        updated_non_default_params: Final = {
            **non_default_params,
            **(prompt_template_optional_params if not ignore_prompt_manager_optional_params else {}),
        }

        if not ignore_prompt_manager_model:
            model = self._get_model_from_prompt(prompt_management_client=prompt_template, model=model)
        else:
            model = model

        return model, completed_messages, updated_non_default_params

    def get_chat_completion_prompt(
        self,
        model: str,
        messages: list[AllMessageValues],
        non_default_params: dict,
        prompt_id: str | None,
        prompt_variables: dict | None,
        dynamic_callback_params: StandardCallbackDynamicParams,
        prompt_spec: PromptSpec | None = None,
        prompt_label: str | None = None,
        prompt_version: int | None = None,
        ignore_prompt_manager_model: bool | None = False,
        ignore_prompt_manager_optional_params: bool | None = False,
    ) -> tuple[str, list[AllMessageValues], dict]:
        if prompt_id is None:
            raise ValueError("prompt_id is required for Prompt Management Base class")
        if not self.should_run_prompt_management(
            prompt_id=prompt_id,
            prompt_spec=prompt_spec,
            dynamic_callback_params=dynamic_callback_params,
        ):
            return model, messages, non_default_params

        prompt_template: Final = self.compile_prompt(
            prompt_id=prompt_id,
            prompt_variables=prompt_variables,
            client_messages=messages,
            dynamic_callback_params=dynamic_callback_params,
            prompt_label=prompt_label,
            prompt_version=prompt_version,
        )

        return self.post_compile_prompt_processing(
            prompt_template=prompt_template,
            messages=messages,
            non_default_params=non_default_params,
            model=model,
            ignore_prompt_manager_model=ignore_prompt_manager_model,
            ignore_prompt_manager_optional_params=ignore_prompt_manager_optional_params,
        )

    async def async_get_chat_completion_prompt(
        self,
        model: str,
        messages: list[AllMessageValues],
        non_default_params: dict,
        prompt_id: str | None,
        prompt_variables: dict | None,
        dynamic_callback_params: StandardCallbackDynamicParams,
        litellm_logging_obj: "LiteLLMLoggingObj",
        prompt_spec: PromptSpec | None = None,
        tools: list[dict] | None = None,
        prompt_label: str | None = None,
        prompt_version: int | None = None,
        ignore_prompt_manager_model: bool | None = False,
        ignore_prompt_manager_optional_params: bool | None = False,
    ) -> tuple[str, list[AllMessageValues], dict]:
        if not self.should_run_prompt_management(
            prompt_id=prompt_id,
            prompt_spec=prompt_spec,
            dynamic_callback_params=dynamic_callback_params,
        ):
            return model, messages, non_default_params

        prompt_template: Final = await self.async_compile_prompt(
            prompt_id=prompt_id,
            prompt_variables=prompt_variables,
            client_messages=messages,
            dynamic_callback_params=dynamic_callback_params,
            prompt_spec=prompt_spec,
            prompt_label=prompt_label,
            prompt_version=prompt_version,
        )

        return self.post_compile_prompt_processing(
            prompt_template=prompt_template,
            messages=messages,
            non_default_params=non_default_params,
            model=model,
            ignore_prompt_manager_model=ignore_prompt_manager_model,
            ignore_prompt_manager_optional_params=ignore_prompt_manager_optional_params,
        )
