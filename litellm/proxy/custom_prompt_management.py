from typing import List, Optional, Tuple

from litellm._logging import verbose_logger
from litellm.integrations.custom_prompt_management import CustomPromptManagement
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import StandardCallbackDynamicParams


class X42PromptManagement(CustomPromptManagement):
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
        Returns:
        - model: str - the model to use (can be pulled from prompt management tool)
        - messages: List[AllMessageValues] - the messages to use (can be pulled from prompt management tool)
        - non_default_params: dict - update with any optional params (e.g. temperature, max_tokens, etc.) to use (can be pulled from prompt management tool)
        """
        verbose_logger.debug(
            f"in async get chat completion prompt. Prompt ID: {prompt_id}, Prompt Variables: {prompt_variables}, Dynamic Callback Params: {dynamic_callback_params}"
        )

        return model, messages, non_default_params

    @property
    def integration_name(self) -> str:
        return "x42-prompt-management"


x42_prompt_management = X42PromptManagement()
