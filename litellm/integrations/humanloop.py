"""
Humanloop integration

https://humanloop.com/
"""

from typing import List, Optional, Tuple

import httpx

import litellm
from litellm.caching import DualCache
from litellm.llms.custom_httpx.http_handler import _get_httpx_client
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import StandardCallbackDynamicParams

from .custom_logger import CustomLogger


class HumanLoopPromptManager(DualCache):
    def _get_prompt_from_id_cache(self, humanloop_prompt_id: str):
        return self.get_cache(key=humanloop_prompt_id)

    def _get_prompt_from_id_api(self, humanloop_prompt_id: str, humanloop_api_key: str):
        client = _get_httpx_client()

        base_url = "https://api.humanloop.com/v5/prompts/{}".format(humanloop_prompt_id)
        response = client.get(
            url=base_url,
            headers={
                "X-Api-Key": humanloop_api_key,
                "Content-Type": "application/json",
            },
        )

        return response.json()

    def _get_prompt_from_id(self, humanloop_prompt_id: str, humanloop_api_key: str):
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


prompt_manager = HumanLoopPromptManager()


class HumanloopLogger(CustomLogger):
    def get_chat_completion_prompt(
        self,
        model: str,
        messages: List[AllMessageValues],
        non_default_params: dict,
        headers: dict,
        prompt_id: str,
        prompt_variables: Optional[dict],
        dynamic_callback_params: StandardCallbackDynamicParams,
    ) -> Tuple[
        str,
        List[AllMessageValues],
        dict,
    ]:
        humanloop_api_key = dynamic_callback_params.get(
            "humanloop_api_key"
        ) or get_secret_str("HUMANLOOP_API_KEY")

        if humanloop_api_key is None:
            return super().get_chat_completion_prompt(
                model=model,
                messages=messages,
                non_default_params=non_default_params,
                headers=headers,
                prompt_id=prompt_id,
                prompt_variables=prompt_variables,
                dynamic_callback_params=dynamic_callback_params,
            )

        hl_client = prompt_manager._get_prompt_from_id(
            humanloop_prompt_id=prompt_id, humanloop_api_key=humanloop_api_key
        )

        return super().get_chat_completion_prompt(
            model,
            messages,
            non_default_params,
            headers,
            prompt_id,
            prompt_variables,
            dynamic_callback_params,
        )
