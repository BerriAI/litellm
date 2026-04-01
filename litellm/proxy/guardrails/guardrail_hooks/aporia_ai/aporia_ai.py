# +-------------------------------------------------------------+
#
#           Use AporiaAI for your LLM calls
#
# +-------------------------------------------------------------+
#  Thank you users! We ❤️ you! - Krrish & Ishaan

import os
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import json
import sys
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Type

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.litellm_core_utils.logging_utils import (
    convert_litellm_response_object_to_str,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import GuardrailEventHooks

GUARDRAIL_NAME = "aporia"

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


class AporiaGuardrail(CustomGuardrail):
    def __init__(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None, **kwargs
    ):
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )
        self.aporia_api_key = api_key or os.environ["APORIO_API_KEY"]
        self.aporia_api_base = api_base or os.environ["APORIO_API_BASE"]
        super().__init__(**kwargs)

    #### CALL HOOKS - proxy only ####
    def transform_messages(self, messages: List[dict]) -> List[dict]:
        supported_openai_roles = ["system", "user", "assistant"]
        default_role = "other"  # for unsupported roles - e.g. tool
        new_messages = []
        for m in messages:
            if m.get("role", "") in supported_openai_roles:
                new_messages.append(m)
            else:
                new_messages.append(
                    {
                        "role": default_role,
                        **{key: value for key, value in m.items() if key != "role"},
                    }
                )

        return new_messages

    async def prepare_aporia_request(
        self, new_messages: List[dict], response_string: Optional[str] = None
    ) -> dict:
        data: dict[str, Any] = {}
        if new_messages is not None:
            data["messages"] = new_messages
        if response_string is not None:
            data["response"] = response_string

        # Set validation target
        if new_messages and response_string:
            data["validation_target"] = "both"
        elif new_messages:
            data["validation_target"] = "prompt"
        elif response_string:
            data["validation_target"] = "response"

        verbose_proxy_logger.debug("Aporia AI request: %s", data)
        return data

    async def make_aporia_api_request(
        self,
        request_data: dict,
        new_messages: List[dict],
        response_string: Optional[str] = None,
    ):
        data = await self.prepare_aporia_request(
            new_messages=new_messages, response_string=response_string
        )

        data.update(
            self.get_guardrail_dynamic_request_body_params(request_data=request_data)
        )

        _json_data = json.dumps(data)

        """
        export APORIO_API_KEY=<your key>
        curl https://gr-prd-trial.aporia.com/some-id \
            -X POST \
            -H "X-APORIA-API-KEY: $APORIO_API_KEY" \
            -H "Content-Type: application/json" \
            -d '{
                "messages": [
                    {
                    "role": "user",
                    "content": "This is a test prompt"
                    }
                ],
                }
'
        """

        response = await self.async_handler.post(
            url=self.aporia_api_base + "/validate",
            data=_json_data,
            headers={
                "X-APORIA-API-KEY": self.aporia_api_key,
                "Content-Type": "application/json",
            },
        )
        verbose_proxy_logger.debug("Aporia AI response: %s", response.text)
        if response.status_code == 200:
            # check if the response was flagged
            _json_response = response.json()
            action: str = _json_response.get(
                "action"
            )  # possible values are modify, passthrough, block, rephrase
            if action == "block":
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Violated guardrail policy",
                        "aporia_ai_response": _json_response,
                    },
                )

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        """
        Use this for the post call moderation with Guardrails
        """
        event_type: GuardrailEventHooks = GuardrailEventHooks.post_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return

        response_str: Optional[str] = convert_litellm_response_object_to_str(response)
        if response_str is not None:
            await self.make_aporia_api_request(
                request_data=data,
                response_string=response_str,
                new_messages=data.get("messages", []),
            )

            add_guardrail_to_applied_guardrails_header(
                request_data=data, guardrail_name=self.guardrail_name
            )

        pass

    @log_guardrail_information
    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal[
            "completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "responses",
            "mcp_call",
            "anthropic_messages",
        ],
    ):
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )
        from litellm.proxy.guardrails.guardrail_helpers import (
            should_proceed_based_on_metadata,
        )

        event_type: GuardrailEventHooks = GuardrailEventHooks.during_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return

        # old implementation - backwards compatibility

        if (
            await should_proceed_based_on_metadata(
                data=data,
                guardrail_name=GUARDRAIL_NAME,
            )
            is False
        ):
            return

        new_messages: Optional[List[dict]] = None
        if "messages" in data and isinstance(data["messages"], list):
            new_messages = self.transform_messages(messages=data["messages"])

        if new_messages is not None:
            await self.make_aporia_api_request(
                request_data=data,
                new_messages=new_messages,
            )
            add_guardrail_to_applied_guardrails_header(
                request_data=data, guardrail_name=self.guardrail_name
            )
        else:
            verbose_proxy_logger.warning(
                "Aporia AI: not running guardrail. No messages in data"
            )
            pass

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.aporia_ai import (
            AporiaGuardrailConfigModel,
        )

        return AporiaGuardrailConfigModel
