from typing import List, Optional, Tuple

from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.prompt_management_base import (
    PromptManagementBase,
    PromptManagementClient,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import StandardCallbackDynamicParams


class CustomPromptManagement(CustomLogger, PromptManagementBase):
    def get_chat_completion_prompt(
        self,
        model: str,
        messages: List[AllMessageValues],
        non_default_params: dict,
        prompt_id: str,
        prompt_variables: Optional[dict],
        dynamic_callback_params: StandardCallbackDynamicParams,
    ) -> Tuple[str, List[AllMessageValues], dict]:
        """
        Returns:
        - model: str - the model to use (can be pulled from prompt management tool)
        - messages: List[AllMessageValues] - the messages to use (can be pulled from prompt management tool)
        - non_default_params: dict - update with any optional params (e.g. temperature, max_tokens, etc.) to use (can be pulled from prompt management tool)
        """
        return model, messages, non_default_params

    @property
    def integration_name(self) -> str:
        return "custom-prompt-management"

    def should_run_prompt_management(
        self,
        prompt_id: str,
        dynamic_callback_params: StandardCallbackDynamicParams,
    ) -> bool:
        return True

    def _compile_prompt_helper(
        self,
        prompt_id: str,
        prompt_variables: Optional[dict],
        dynamic_callback_params: StandardCallbackDynamicParams,
    ) -> PromptManagementClient:
        raise NotImplementedError(
            "Custom prompt management does not support compile prompt helper"
        )
