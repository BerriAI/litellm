# +-------------------------------------------------------------+
#
#           Use Aim Security Guardrails for your LLM calls
#                   https://www.aim.security/
#
# +-------------------------------------------------------------+
import asyncio
import json
import os
from typing import Any, AsyncGenerator, Literal, Optional, Union

from fastapi import HTTPException
from pydantic import BaseModel
from websockets.asyncio.client import ClientConnection, connect

from litellm import DualCache
from litellm._version import version as litellm_version
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.proxy_server import StreamingCallbackError
from litellm.types.utils import (
    Choices,
    EmbeddingResponse,
    ImageResponse,
    ModelResponse,
    ModelResponseStream,
)


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
        ],
    ) -> Union[Exception, str, dict, None]:
        verbose_proxy_logger.debug("Inside AIM Pre-Call Hook")

        return await self.call_aim_guardrail(
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
        ],
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
            f"{self.api_base}/detect/openai/v2",
            headers=headers,
            json={"messages": data.get("messages", [])},
        )
        response.raise_for_status()
        res = response.json()
        required_action = res.get("required_action") and res["required_action"].get("prompt_policy_action", None)
        if required_action is None:
            verbose_proxy_logger.warning("Aim: No required action specified")
            return data
        match required_action:
            case "monitor_action":
                verbose_proxy_logger.info("Aim: monitor action")
            case "block_action":
                self._handle_block_action(res["analysis_result"])
            case "anonymize_action":
                verbose_proxy_logger.info("Aim: anonymize action")
                return self._handle_anonymize_action(res, data)
            case "engage_action":
                verbose_proxy_logger.info("Aim: engage action")
            case _:
                verbose_proxy_logger.error("Aim: unknown action")
        return data

    def _handle_block_action(self, analysis_result: Any) -> None:
        detected = analysis_result["detected"]
        verbose_proxy_logger.info(
            "Aim: detected: {detected}, enabled policies: {policies}".format(
                detected=detected,
                policies=list(analysis_result["details"].keys()),
            ),
        )
        if detected:
            raise HTTPException(status_code=400, detail=analysis_result["detection_message"])

    def _handle_anonymize_action(self, res: Any, data: dict) -> dict:
        try:
            redaction_result = res["redaction_result"]
            if not redaction_result:
                return data
            analysis_result = res["analysis_result"]
            if analysis_result and analysis_result.get("session_entities"):
                self.dlp_entities = analysis_result.get("session_entities")
            data["messages"] = [
                    {
                        "role": redaction_result["redacted_new_message"]["role"],
                        "content": redaction_result["redacted_new_message"]["content"],
                    }
                ] + [
                    {
                        "role": message["role"],
                        "content": message["content"],
                    }
                    for message in redaction_result["all_redacted_messages"]
                ]
            return data
        except Exception as e:
            verbose_proxy_logger.error(f"Aim: Error while redacting: {e}")
            return data # todo do we want to fallback to this or raise?


    async def call_aim_guardrail_on_output(
        self, request_data: dict, output: str, hook: str, key_alias: Optional[str]
    ) -> Optional[dict]:
        user_email = (
            request_data.get("metadata", {}).get("headers", {}).get("x-aim-user-email")
        )
        call_id = request_data.get("litellm_call_id")
        response = await self.async_handler.post(
            f"{self.api_base}/detect/output/v2",
            headers=self._build_aim_headers(
                hook=hook,
                key_alias=key_alias,
                user_email=user_email,
                litellm_call_id=call_id,
            ),
            json={"output": output, "messages": request_data.get("messages", [])},
        )
        response.raise_for_status()
        res = response.json()
        required_action = res.get("required_action") and res["required_action"].get("prompt_policy_action", None)
        if required_action is None:
            verbose_proxy_logger.warning("Aim: No required action specified")
            return self._deanonymize_output(res, output)
        match required_action:
            case "monitor_action":
                verbose_proxy_logger.info("Aim: monitor action")
            case "block_action":
                return self._handle_block_action_on_output(res["analysis_result"])
            case "anonymize_action":
                verbose_proxy_logger.info("Aim: anonymize action")
            case "engage_action":
                verbose_proxy_logger.info("Aim: engage action")
            case _:
                verbose_proxy_logger.error("Aim: unknown action")
        return self._deanonymize_output(res, res)

    def _handle_block_action_on_output(self, analysis_result: Any):
        analysis_result = analysis_result
        detected = analysis_result["detected"]
        verbose_proxy_logger.info(
            "Aim: detected: {detected}, enabled policies: {policies}".format(
                detected=detected,
                policies=list(analysis_result["details"].keys()),
            ),
        )
        if detected:
            return {"detection_message": analysis_result["detection_message"]}
        return None

    def _deanonymize_output(self, res: Any, output: str) -> dict | None:
        try:
            for entity in self.dlp_entities:
                output = output.replace(f"[{entity["name"]}]", entity["content"])
            return {"redacted_output": output}
        except Exception as e:
            verbose_proxy_logger.error(f"Aim: Error while redacting output: {e}")
            return None

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
            | ({"x-aim-litellm-call-id": litellm_call_id} if litellm_call_id else {})
            # Used by Aim to track guardrails violations by user.
            | ({"x-aim-user-email": user_email} if user_email else {})
            | (
                {
                    # Used by Aim apply only the guardrails that are associated with the key alias.
                    "x-aim-litellm-key-alias": key_alias,
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
            if aim_output_guardrail_result and aim_output_guardrail_result.get("detection_message"):
                raise HTTPException(status_code=400, detail=aim_output_guardrail_result.get("detection_message"))
            if aim_output_guardrail_result and aim_output_guardrail_result.get("redacted_output"):
                response.choices[0].message.content = aim_output_guardrail_result.get("redacted_output")

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
            f"{self.ws_api_base}/detect/output/ws",
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
