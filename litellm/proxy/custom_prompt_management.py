from typing import Final

from litellm._logging import verbose_logger
from litellm.integrations.custom_prompt_management import CustomPromptManagement
from litellm.types.llms.openai import AllMessageValues
from litellm.types.prompts.init_prompts import PromptSpec
from litellm.types.utils import StandardCallbackDynamicParams


class X42PromptManagement(CustomPromptManagement):
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
        """
        Returns:
        - model: str - the model to use (can be pulled from prompt management tool)
        - messages: List[AllMessageValues] - the messages to use (can be pulled from prompt management tool)
        - non_default_params: dict - update with any optional params (e.g. temperature, max_tokens, etc.) to use (can be pulled from prompt management tool)
        """
        verbose_logger.debug(
            "in async get chat completion prompt. Prompt ID: %s, Prompt Variables: %s, Dynamic Callback Params: %s",
            prompt_id,
            prompt_variables,
            dynamic_callback_params,
        )

        return model, messages, non_default_params

    @property
    def integration_name(self) -> str:
        return "x42-prompt-management"


x42_prompt_management: Final = X42PromptManagement()
