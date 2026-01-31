# +-------------------------------------------------------------+
#
#           Use Aim Security Guardrails for your LLM calls
#                   https://www.aim.security/
#
# +-------------------------------------------------------------+
import asyncio
import json
import os
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional, Type, Union

from fastapi import HTTPException
from pydantic import BaseModel
from websockets.asyncio.client import ClientConnection, connect

from litellm import DualCache
from litellm._logging import verbose_proxy_logger
from litellm._version import version as litellm_version
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import (
    CallTypesLiteral,
    Choices,
    EmbeddingResponse,
    ImageResponse,
    ModelResponse,
    ModelResponseStream,
)

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


class AimGuardrailMissingSecrets(Exception):
    pass


class AimGuardrail(CustomGuardrail):
    def __init__(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None, **kwargs
    ):
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )
        self.api_key = api_key or os.environ.get("AIM_API_KEY")
        if not self.api_key:
            msg = (
                "Couldn't get Aim api key, either set the `AIM_API_KEY` in the environment or "
                "pass it as a parameter to the guardrail in the config file"
            )
            raise AimGuardrailMissingSecrets(msg)
        self.api_base = (
            api_base or os.environ.get("AIM_API_BASE") or "https://api.aim.security"
        )
        self.ws_api_base = self.api_base.replace("http://", "ws://").replace(
            "https://", "wss://"
        )
        self.dlp_entities: list[dict] = []
        self._max_dlp_entities = 100
        super().__init__(**kwargs)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral,
    ) -> Union[Exception, str, dict, None]:
        verbose_proxy_logger.debug("Inside AIM Pre-Call Hook")
        return await self.call_aim_guardrail(
            data, hook="pre_call", key_alias=user_api_key_dict.key_alias
        )

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: CallTypesLiteral,
    ) -> Union[Exception, str, dict, None]:
        verbose_proxy_logger.debug("Inside AIM Moderation Hook")

        await self.call_aim_guardrail(
            data, hook="moderation", key_alias=user_api_key_dict.key_alias
        )
        return data

    async def call_aim_guardrail(
        self, data: dict, hook: str, key_alias: Optional[str]
    ) -> dict:
        user_email = data.get("metadata", {}).get("headers", {}).get("x-aim-user-email")
        call_id = data.get("litellm_call_id")
        headers = self._build_aim_headers(
            hook=hook,
            key_alias=key_alias,
            user_email=user_email,
            litellm_call_id=call_id,
        )
        response = await self.async_handler.post(
            f"{self.api_base}/fw/v1/analyze",
            headers=headers,
            json={"messages": data.get("messages", [])},
        )
        response.raise_for_status()
        res = response.json()
        required_action = res.get("required_action")
        action_type = required_action and required_action.get("action_type", None)
        if action_type is None:
            verbose_proxy_logger.debug("Aim: No required action specified")
            return data
        if action_type == "monitor_action":
            verbose_proxy_logger.info("Aim: monitor action")
        elif action_type == "block_action":
            self._handle_block_action(res["analysis_result"], required_action)
        elif action_type == "anonymize_action":
            return self._anonymize_request(
                res, data
            )
        else:
            verbose_proxy_logger.error(f"Aim: {action_type} action")
        return data

    def _handle_block_action(self, analysis_result: Any, required_action: Any) -> None:
        detection_message = required_action.get("detection_message", None)
        verbose_proxy_logger.info(
            "Aim: Violation detected enabled policies: {policies}".format(
                policies=list(analysis_result["policy_drill_down"].keys()),
            ),
        )
        raise HTTPException(status_code=400, detail=detection_message)

    def _anonymize_request(
        self, res: Any, data: dict
    ) -> dict:
        verbose_proxy_logger.info("Aim: anonymize action")
        redacted_chat = res.get("redacted_chat")
        if not redacted_chat:
            return data
        data["messages"] = [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in redacted_chat["all_redacted_messages"]
        ]
        return data

    async def call_aim_guardrail_on_output(
        self, request_data: dict, output: str, hook: str, key_alias: Optional[str]
    ) -> Optional[dict]:
        user_email = (
            request_data.get("metadata", {}).get("headers", {}).get("x-aim-user-email")
        )
        call_id = request_data.get("litellm_call_id")
        response = await self.async_handler.post(
            f"{self.api_base}/fw/v1/analyze",
            headers=self._build_aim_headers(
                hook=hook,
                key_alias=key_alias,
                user_email=user_email,
                litellm_call_id=call_id,
            ),
            json={
                "messages": request_data.get("messages", [])
                + [{"role": "assistant", "content": output}]
            },
        )
        response.raise_for_status()
        res = response.json()
        required_action = res.get("required_action")
        action_type = required_action and required_action.get("action_type", None)
        if action_type and action_type == "block_action":
            return self._handle_block_action_on_output(
                res["analysis_result"], required_action
            )
        redacted_chat = res.get("redacted_chat", None)

        if action_type and action_type == "anonymize_action" and redacted_chat:
            return {"redacted_output": redacted_chat["all_redacted_messages"][-1]["content"]}
        return {"redacted_output": output}

    def _handle_block_action_on_output(
        self, analysis_result: Any, required_action: Any
    ) -> dict | None:
        detection_message = required_action.get("detection_message", None)
        verbose_proxy_logger.info(
            "Aim: detected: {detected}, enabled policies: {policies}".format(
                detected=True,
                policies=list(analysis_result["policy_drill_down"].keys()),
            ),
        )
        return {"detection_message": detection_message}

    def _build_aim_headers(
        self,
        *,
        hook: str,
        key_alias: Optional[str],
        user_email: Optional[str],
        litellm_call_id: Optional[str],
    ):
        """
        A helper function to build the http headers that are required by AIM guardrails.
        """
        return (
            {
                "Authorization": f"Bearer {self.api_key}",
                # Used by Aim to apply only the guardrails that should be applied in a specific request phase.
                "x-aim-litellm-hook": hook,
                # Used by Aim to track LiteLLM version and provide backward compatibility.
                "x-aim-litellm-version": litellm_version,
            }
            # Used by Aim to track together single call input and output
            | ({"x-aim-call-id": litellm_call_id} if litellm_call_id else {})
            # Used by Aim to track guardrails violations by user.
            | ({"x-aim-user-email": user_email} if user_email else {})
            | (
                {
                    # Used by Aim apply only the guardrails that are associated with the key alias.
                    "x-aim-gateway-key-alias": key_alias,
                }
                if key_alias
                else {}
            )
        )

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Union[Any, ModelResponse, EmbeddingResponse, ImageResponse],
    ) -> Any:
        if (
            isinstance(response, ModelResponse)
            and response.choices
            and isinstance(response.choices[0], Choices)
        ):
            content = response.choices[0].message.content or ""
            aim_output_guardrail_result = await self.call_aim_guardrail_on_output(
                data, content, hook="output", key_alias=user_api_key_dict.key_alias
            )
            if aim_output_guardrail_result and aim_output_guardrail_result.get(
                "detection_message"
            ):
                raise HTTPException(
                    status_code=400,
                    detail=aim_output_guardrail_result.get("detection_message"),
                )
            if aim_output_guardrail_result and aim_output_guardrail_result.get(
                "redacted_output"
            ):
                response.choices[0].message.content = aim_output_guardrail_result.get(
                    "redacted_output"
                )
        return response

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        user_email = (
            request_data.get("metadata", {}).get("headers", {}).get("x-aim-user-email")
        )
        call_id = request_data.get("litellm_call_id")
        async with connect(
            f"{self.ws_api_base}/fw/v1/analyze/stream",
            additional_headers=self._build_aim_headers(
                hook="output",
                key_alias=user_api_key_dict.key_alias,
                user_email=user_email,
                litellm_call_id=call_id,
            ),
        ) as websocket:
            sender = asyncio.create_task(
                self.forward_the_stream_to_aim(websocket, response)
            )
            while True:
                result = json.loads(await websocket.recv())
                if verified_chunk := result.get("verified_chunk"):
                    yield ModelResponseStream.model_validate(verified_chunk)
                else:
                    sender.cancel()
                    if result.get("done"):
                        return
                    if blocking_message := result.get("blocking_message"):
                        from litellm.proxy.proxy_server import StreamingCallbackError

                        raise StreamingCallbackError(blocking_message)
                    verbose_proxy_logger.error(
                        f"Unknown message received from AIM: {result}"
                    )
                    return

    async def forward_the_stream_to_aim(
        self,
        websocket: ClientConnection,
        response_iter,
    ) -> None:
        async for chunk in response_iter:
            if isinstance(chunk, BaseModel):
                chunk = chunk.model_dump_json()
            if isinstance(chunk, dict):
                chunk = json.dumps(chunk)
            await websocket.send(chunk)
        await websocket.send(json.dumps({"done": True}))

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.aim import (
            AimGuardrailConfigModel,
        )

        return AimGuardrailConfigModel
