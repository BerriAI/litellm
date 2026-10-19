# +-------------------------------------------------------------+
#
#           Use OpenAI /moderations for your LLM calls
#
# +-------------------------------------------------------------+
#  Thank you users! We ❤️ you! - Krrish & Ishaan

import os
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import sys

from fastapi import HTTPException

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import DEFAULT_OPENAI_MODERATIONS_MODEL
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails._content_utils import split_message_text
from litellm.types.utils import CallTypesLiteral


class ENTERPRISE_OpenAI_Moderation(CustomLogger):
    @property
    def model_name(self) -> str:
        return litellm.openai_moderations_model_name or DEFAULT_OPENAI_MODERATIONS_MODEL

    #### CALL HOOKS - proxy only ####

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: CallTypesLiteral,
    ):
        # Covers multimodal list content + Responses-API input. System, instructions
        # and tool text go in a second call after the message text.
        message_text, added_text = split_message_text(data)
        texts = ("".join(message_text), "".join(added_text)) if added_text else ("".join(message_text),)

        from litellm.proxy.proxy_server import llm_router

        if llm_router is None:
            return

        for text in texts:
            moderation_response = await llm_router.amoderation(
                model=self.model_name, input=text
            )

            verbose_proxy_logger.debug("Moderation response: %s", moderation_response)
            if moderation_response and moderation_response.results[0].flagged is True:
                raise HTTPException(
                    status_code=403, detail={"error": "Violated content safety policy"}
                )
_ENTERPRISE_OpenAI_Moderation = ENTERPRISE_OpenAI_Moderation
