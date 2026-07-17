# +-------------------------------------------------------------+
#
#            FangcunGuard Guardrail for LiteLLM
#              https://www.fangcunleap.com
#
# +-------------------------------------------------------------+

import os
from typing import TYPE_CHECKING, List, Literal, Optional, Type, Union

from fastapi import HTTPException

from litellm import DualCache
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

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

GUARDRAIL_NAME = "fangcunguard"
DEFAULT_API_BASE = "https://api.fangcunleap.com"


class FangcunGuardrail(CustomGuardrail):
    """FangcunGuard content-safety guardrail.

    FangcunGuard is a compact encoder-only safety classifier by FangcunLeap
    (https://www.fangcunleap.com). It scores text against 10 categories (9
    unsafe + 1 safe) and returns a verdict plus the specific risk label.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs,
    ):
        kwargs.setdefault("supported_event_hooks", list(self.get_supported_event_hooks()))
        self.async_handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.GuardrailCallback)
        self.fangcun_api_key = api_key or os.environ.get("FANGCUN_API_KEY")
        self.fangcun_api_base = api_base or os.environ.get("FANGCUN_API_BASE") or DEFAULT_API_BASE
        super().__init__(**kwargs)

    @classmethod
    def get_supported_event_hooks(cls) -> List[GuardrailEventHooks]:
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.during_call,
            GuardrailEventHooks.post_call,
        ]

    @staticmethod
    def _extract_texts(data: dict) -> List[str]:
        """Pull user/assistant text out of an OpenAI-style request."""
        texts: List[str] = []
        for message in data.get("messages", []) or []:
            content = message.get("content")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        texts.append(part.get("text", ""))
        return [t for t in texts if t and t.strip()]

    async def _check_text(self, text: str, request_data: dict) -> None:
        """Call FangcunGuard; raise HTTPException(400) if the text is unsafe."""
        headers = {
            "Content-Type": "application/json",
        }
        if self.fangcun_api_key:
            headers["Authorization"] = f"Bearer {self.fangcun_api_key}"

        body = {"text": text}
        body.update(self.get_guardrail_dynamic_request_body_params(request_data=request_data))

        response = await self.async_handler.post(
            url=self.fangcun_api_base.rstrip("/") + "/guard/context",
            json=body,
            headers=headers,
        )
        verbose_proxy_logger.debug("FangcunGuard response: %s", response.text)

        if response.status_code != 200:
            verbose_proxy_logger.warning("FangcunGuard: non-200 response (%s), skipping", response.status_code)
            return

        result = response.json()

        is_unsafe = not bool(result.get("is_safe", True))

        if is_unsafe:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Violated FangcunGuard content policy",
                    "fangcunguard_response": result,
                },
            )

    async def _scan_texts(self, texts: List[str], request_data: dict) -> None:
        for text in texts:
            await self._check_text(text, request_data)

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
            "mcp_call",
            "anthropic_messages",
        ],
    ) -> Optional[Union[Exception, str, dict]]:
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        event_type = GuardrailEventHooks.pre_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return data

        await self._scan_texts(self._extract_texts(data), request_data=data)
        add_guardrail_to_applied_guardrails_header(request_data=data, guardrail_name=self.guardrail_name)
        return data

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
    ) -> Optional[Union[Exception, str, dict]]:
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        event_type = GuardrailEventHooks.during_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return data

        await self._scan_texts(self._extract_texts(data), request_data=data)
        add_guardrail_to_applied_guardrails_header(request_data=data, guardrail_name=self.guardrail_name)
        return data

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

        event_type = GuardrailEventHooks.post_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return

        response_str: Optional[str] = convert_litellm_response_object_to_str(response)
        if response_str is not None:
            await self._check_text(response_str, request_data=data)
            add_guardrail_to_applied_guardrails_header(request_data=data, guardrail_name=self.guardrail_name)

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.fangcunguard import (
            FangcunGuardrailConfigModel,
        )

        return FangcunGuardrailConfigModel
