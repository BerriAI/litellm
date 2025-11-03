import os
import re
from typing import TYPE_CHECKING, Any, AsyncGenerator, Literal, Optional, Type, Union
from fastapi import HTTPException
from litellm import DualCache
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import (
    Choices,
    Delta,
    EmbeddingResponse,
    ImageResponse,
    ModelResponse,
    ModelResponseStream
)

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


class PromptSecurityGuardrailMissingSecrets(Exception):
    pass


class PromptSecurityGuardrail(CustomGuardrail):
    def __init__(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None, **kwargs
    ):
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )
        self.api_key = api_key or os.environ.get("PROMPT_SECURITY_API_KEY")
        if not self.api_key:
            msg = (
                "Couldn't get Prompt Security api key, either set the `PROMPT_SECURITY_API_KEY` in the environment or "
                "pass it as a parameter to the guardrail in the config file"
            )
            raise PromptSecurityGuardrailMissingSecrets(msg)
        self.api_base = (
            api_base or os.environ.get("PROMPT_SECURITY_API_BASE") or ""
        )
        super().__init__(**kwargs)

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
    ) -> Union[Exception, str, dict, None]:
        verbose_proxy_logger.debug("Inside PROMPT_SECURITY Pre-Call Hook")
        return await self.call_prompt_security_guardrail(
            data, hook="pre_call", key_alias=user_api_key_dict.key_alias
        )

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
    ) -> Union[Exception, str, dict, None]:
        verbose_proxy_logger.debug("Inside PROMPT_SECURITY Moderation Hook")

        await self.call_prompt_security_guardrail(
            data, hook="moderation", key_alias=user_api_key_dict.key_alias
        )
        return data

    async def call_prompt_security_guardrail(
        self, data: dict, hook: str, key_alias: Optional[str]
    ) -> dict:
        headers = {
            'APP-ID': '73a4a369-ca52-4802-8e17-5c9f62e113ca',
            'Content-Type': 'application/json'
        }
        response = await self.async_handler.post(
            f"https://staging-eu.prompt.security/api/protect",
            headers=headers,
            json={"messages": data.get("messages", [])},
        )
        response.raise_for_status()
        res = response.json()
        result = res.get("result", {}).get("prompt", {})
        action = result.get("action")
        if action == "block":
            self._handle_block_action({}, "")
        elif action == "modify":
            data["messages"] = result.get("modified_messages", [])
        else:
            verbose_proxy_logger.error(f"Prompt Security: {action} action")
        return data

    def _handle_block_action(self, analysis_result: Any, required_action: Any) -> None:
        detection_message = required_action.get("detection_message", None)
        verbose_proxy_logger.info(
            "Prompt Security: Violation detected enabled policies: {policies}".format(
                policies=list(analysis_result["policy_drill_down"].keys()),
            ),
        )
        raise HTTPException(status_code=400, detail=detection_message)

    async def call_prompt_security_guardrail_on_output(
        self, request_data: dict, output: str, hook: str, key_alias: Optional[str]
    ) -> dict:
        response = await self.async_handler.post(
            f"https://staging-eu.prompt.security/api/protect",
            headers = {
                'APP-ID': '73a4a369-ca52-4802-8e17-5c9f62e113ca',
                'Content-Type': 'application/json'
            },
            json = { "response": output }
        )
        response.raise_for_status()
        res = response.json()
        result = res.get("result", {}).get("response", {})
        return { "action": result.get("action"), "modified_text": result.get("modified_text") }

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Union[Any, ModelResponse, EmbeddingResponse, ImageResponse],
    ) -> Any:
        if (isinstance(response, ModelResponse) and response.choices and isinstance(response.choices[0], Choices)):
            content = response.choices[0].message.content or ""
            ret = await self.call_prompt_security_guardrail_on_output(data, content, "output", None)
            if ret.get("action") == "block":
                raise HTTPException(status_code=400, detail="Blocked by Prompt Security")
            elif ret.get("action") == "modify":
                response.choices[0].message.content = ret.get("modified_text")
        return response

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        buffer: str = ""
        WINDOW_SIZE = 50

        async for item in response:
            if not isinstance(item, ModelResponseStream) or not item.choices or len(item.choices) == 0:
                yield item
                continue

            choice = item.choices[0]
            if choice.delta and choice.delta.content:
                buffer += choice.delta.content

            if choice.finish_reason or len(buffer) >= WINDOW_SIZE:
                if buffer:
                    if not choice.finish_reason and re.search(r'\s', buffer):
                        chunk, buffer = re.split(r'(?=\s\S*$)', buffer, 1)
                    else:
                        chunk, buffer = buffer,''

                    ret = await self.call_prompt_security_guardrail_on_output(request_data, chunk, "streaming", None)
                    if ret.get("action") == "block":
                        from litellm.proxy.proxy_server import StreamingCallbackError
                        raise StreamingCallbackError("Blocked by Prompt Security")
                    elif ret.get("action") == "modify":
                        chunk = ret.get("modified_text")
                    
                    if choice.delta:
                        choice.delta.content = chunk
                    else:
                        choice.delta = Delta(content=chunk)
                yield item

    
    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.prompt_security import (
            PromptSecurityGuardrailConfigModel,
        )

        return PromptSecurityGuardrailConfigModel
