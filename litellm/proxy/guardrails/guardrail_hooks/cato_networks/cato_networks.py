# +-------------------------------------------------------------+
#
#           Use Cato Networks Guardrails for your LLM calls
#                   https://www.catonetworks.com/
#
# +-------------------------------------------------------------+
import asyncio
import contextlib
import json
import os
import ssl
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional, Type, Union

from fastapi import HTTPException
from pydantic import BaseModel
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from litellm import DualCache
from litellm._logging import verbose_proxy_logger
from litellm._version import version as litellm_version
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    get_ssl_configuration,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails._content_utils import (
    apply_redacted_messages_back,
    build_inspection_messages,
)
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


class CatoNetworksGuardrailMissingSecrets(Exception):
    pass


class CatoNetworksGuardrail(CustomGuardrail):
    def __init__(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None, **kwargs
    ):
        ssl_verify = kwargs.pop("ssl_verify", None)
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
            params={"ssl_verify": ssl_verify} if ssl_verify is not None else None,
        )
        self.api_key = api_key or os.environ.get("CATO_API_KEY")
        if not self.api_key:
            msg = (
                "Couldn't get Cato Networks api key, either set the `CATO_API_KEY` in the environment or "
                "pass it as a parameter to the guardrail in the config file"
            )
            raise CatoNetworksGuardrailMissingSecrets(msg)
        self.api_base = (
            api_base
            or os.environ.get("CATO_API_BASE")
            or "https://api.aisec.catonetworks.com"
        )
        self.api_base = self.api_base.rstrip("/")
        self.ws_api_base = self.api_base.replace("http://", "ws://").replace(
            "https://", "wss://"
        )
        self._ws_connect_ssl_kwargs = self._build_ws_ssl_kwargs(
            ssl_verify, self.ws_api_base
        )
        super().__init__(**kwargs)

    @staticmethod
    def _build_ws_ssl_kwargs(
        ssl_verify: Optional[Union[bool, str]], ws_api_base: str
    ) -> dict:
        """Resolve the ``ssl`` argument for ``websockets.connect``. Mirrors the
        ``ssl_verify`` handling applied to the HTTP handler so a custom Cato instance
        behind TLS honours the same verification settings for streaming."""
        if ssl_verify is None or not ws_api_base.startswith("wss://"):
            return {}
        ssl_config = get_ssl_configuration(ssl_verify)
        if ssl_config is False:
            ssl_config = ssl.create_default_context()
            ssl_config.check_hostname = False
            ssl_config.verify_mode = ssl.CERT_NONE
        return {"ssl": ssl_config}

    @staticmethod
    def _resolve_cato_user_email(user_api_key_dict: UserAPIKeyAuth) -> Optional[str]:
        """Only the key/JWT-bound user email is trusted. ``end_user_id`` is derived from
        caller-supplied request fields (OpenAI ``user``, headers, metadata) and is spoofable,
        so it must never be forwarded as the Cato user identity."""
        return user_api_key_dict.user_email

    @staticmethod
    async def _cancel_background_task(task: asyncio.Task) -> None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral,
    ) -> Union[Exception, str, dict, None]:
        verbose_proxy_logger.debug("Inside Cato Pre-Call Hook")
        return await self.call_cato_guardrail(
            data,
            hook="pre_call",
            key_alias=user_api_key_dict.key_alias,
            user_email=self._resolve_cato_user_email(user_api_key_dict),
        )

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: CallTypesLiteral,
    ) -> Union[Exception, str, dict, None]:
        verbose_proxy_logger.debug("Inside Cato Moderation Hook")
        return await self.call_cato_guardrail(
            data,
            hook="moderation",
            key_alias=user_api_key_dict.key_alias,
            user_email=self._resolve_cato_user_email(user_api_key_dict),
        )

    @classmethod
    def _inspection_messages(cls, data: dict) -> list:
        """Flatten multimodal list ``content`` into plain text so Cato inspects
        every text fragment. Chat ``messages`` stay 1:1 with the request so
        redacted results map back by index, and every other field the proxy
        forwards to the model (Responses-API ``input``/``instructions`` and
        legacy completion ``prompt``) is appended as synthetic messages so blocked
        text cannot bypass inspection by hiding in one of them."""
        flattened = []
        for message in data.get("messages") or []:
            if isinstance(message, dict) and isinstance(message.get("content"), list):
                parts = build_inspection_messages({"messages": [message]})
                flattened.append(
                    {**message, "content": parts[0]["content"] if parts else ""}
                )
            else:
                flattened.append(message)
        for _field, messages in cls._extra_inspection_sources(data):
            flattened.extend(messages)
        return flattened

    @staticmethod
    def _prompt_inspection_messages(prompt: Any) -> list:
        """Synthetic user messages for a legacy completion ``prompt`` (a string
        or a list of string prompts)."""
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}] if prompt else []
        if isinstance(prompt, list):
            return [
                {"role": "user", "content": part}
                for part in prompt
                if isinstance(part, str) and part
            ]
        return []

    @classmethod
    def _extra_inspection_sources(cls, data: dict) -> list:
        """Text the proxy forwards to the model outside chat ``messages``:
        Responses-API ``input`` and ``instructions``, and legacy completion
        ``prompt``. Returned as ``(field, messages)`` in a fixed order so the
        anonymize path can slice redactions back to the field they came from."""
        sources: list = []
        input_messages = build_inspection_messages({"input": data.get("input")})
        if input_messages:
            sources.append(("input", input_messages))
        instructions = data.get("instructions")
        if isinstance(instructions, str) and instructions:
            sources.append(
                ("instructions", [{"role": "system", "content": instructions}])
            )
        prompt_messages = cls._prompt_inspection_messages(data.get("prompt"))
        if prompt_messages:
            sources.append(("prompt", prompt_messages))
        return sources

    async def call_cato_guardrail(
        self,
        data: dict,
        hook: str,
        key_alias: Optional[str],
        user_email: Optional[str] = None,
    ) -> dict:
        call_id = data.get("litellm_call_id")
        headers = self._build_cato_headers(
            hook=hook,
            key_alias=key_alias,
            user_email=user_email,
            litellm_call_id=call_id,
        )
        response = await self.async_handler.post(
            f"{self.api_base}/fw/v1/analyze",
            headers=headers,
            json={"messages": self._inspection_messages(data)},
        )
        response.raise_for_status()
        res = response.json()
        required_action = res.get("required_action")
        action_type = required_action and required_action.get("action_type", None)
        if action_type is None:
            verbose_proxy_logger.debug("Cato: No required action specified")
            return data
        if action_type == "monitor_action":
            verbose_proxy_logger.info("Cato: monitor action")
        elif action_type == "block_action":
            self._handle_block_action(res.get("analysis_result", {}), required_action)
        elif action_type == "anonymize_action":
            return self._anonymize_request(res, data)
        else:
            verbose_proxy_logger.error(f"Cato: {action_type} action")
        return data

    def _handle_block_action(self, analysis_result: Any, required_action: Any) -> None:
        detection_message = required_action.get("detection_message", None)
        verbose_proxy_logger.info(
            "Cato: Violation detected enabled policies: {policies}".format(
                policies=list(analysis_result.get("policy_drill_down", {}).keys()),
            ),
        )
        raise HTTPException(status_code=400, detail=detection_message)

    def _anonymize_request(self, res: Any, data: dict) -> dict:
        verbose_proxy_logger.info("Cato: anonymize action")
        redacted_chat = res.get("redacted_chat")
        if not redacted_chat:
            return data
        redacted_messages = redacted_chat.get("all_redacted_messages") or []
        original_messages = data.get("messages")
        offset = 0
        if original_messages:
            data["messages"] = [
                (
                    {**original, "content": redacted_messages[idx]["content"]}
                    if idx < len(redacted_messages)
                    and redacted_messages[idx].get("content") is not None
                    else original
                )
                for idx, original in enumerate(original_messages)
            ]
            offset = len(original_messages)
        for field, messages in self._extra_inspection_sources(data):
            redacted_slice = redacted_messages[offset : offset + len(messages)]
            offset += len(messages)
            if redacted_slice:
                self._apply_extra_redaction(data, field, redacted_slice)
        return data

    @classmethod
    def _apply_extra_redaction(cls, data: dict, field: str, redacted: list) -> None:
        if field == "input":
            input_only = {"input": data["input"]}
            apply_redacted_messages_back(input_only, redacted)
            data["input"] = input_only["input"]
        elif field == "instructions":
            if redacted[0].get("content") is not None:
                data["instructions"] = redacted[0]["content"]
        elif field == "prompt":
            cls._apply_prompt_redaction(data, redacted)

    @staticmethod
    def _apply_prompt_redaction(data: dict, redacted: list) -> None:
        contents = [m.get("content") for m in redacted if isinstance(m, dict)]
        prompt = data.get("prompt")
        if isinstance(prompt, str):
            if contents and contents[0] is not None:
                data["prompt"] = contents[0]
            return
        if isinstance(prompt, list):
            new_prompt = list(prompt)
            redactions = iter(contents)
            for idx, part in enumerate(new_prompt):
                if isinstance(part, str) and part:
                    replacement = next(redactions, None)
                    if replacement is not None:
                        new_prompt[idx] = replacement
            data["prompt"] = new_prompt

    async def call_cato_guardrail_on_output(
        self,
        request_data: dict,
        output: str,
        hook: str,
        key_alias: Optional[str],
        user_email: Optional[str] = None,
    ) -> Optional[dict]:
        call_id = request_data.get("litellm_call_id")
        response = await self.async_handler.post(
            f"{self.api_base}/fw/v1/analyze",
            headers=self._build_cato_headers(
                hook=hook,
                key_alias=key_alias,
                user_email=user_email,
                litellm_call_id=call_id,
            ),
            json={
                "messages": self._inspection_messages(request_data)
                + [{"role": "assistant", "content": output}]
            },
        )
        response.raise_for_status()
        res = response.json()
        required_action = res.get("required_action")
        action_type = required_action and required_action.get("action_type", None)
        if action_type and action_type == "block_action":
            return self._handle_block_action_on_output(
                res.get("analysis_result", {}), required_action
            )
        redacted_chat = res.get("redacted_chat", None)

        if action_type and action_type == "anonymize_action" and redacted_chat:
            all_redacted = redacted_chat.get("all_redacted_messages") or []
            redacted_output = all_redacted[-1].get("content") if all_redacted else None
            if redacted_output is not None:
                return {"redacted_output": redacted_output}
        return {"redacted_output": output}

    def _handle_block_action_on_output(
        self, analysis_result: Any, required_action: Any
    ) -> dict | None:
        detection_message = required_action.get("detection_message", None)
        verbose_proxy_logger.info(
            "Cato: detected: {detected}, enabled policies: {policies}".format(
                detected=True,
                policies=list(analysis_result.get("policy_drill_down", {}).keys()),
            ),
        )
        return {"detection_message": detection_message}

    def _build_cato_headers(
        self,
        *,
        hook: str,
        key_alias: Optional[str],
        user_email: Optional[str],
        litellm_call_id: Optional[str],
    ):
        """
        A helper function to build the http headers that are required by Cato guardrails.
        """
        return (
            {
                "Authorization": f"Bearer {self.api_key}",
                # Used by Cato Networks to apply only the guardrails that should be applied in a specific request phase.
                "x-cato-litellm-hook": hook,
                # Used by Cato Networks to track LiteLLM version and provide backward compatibility.
                "x-cato-litellm-version": litellm_version,
            }
            # Used by Cato Networks to track together single call input and output
            | ({"x-cato-call-id": litellm_call_id} if litellm_call_id else {})
            # Used by Cato Networks to track guardrails violations by user.
            | ({"x-cato-user-email": user_email} if user_email else {})
            | (
                {
                    # Used by Cato Networks apply only the guardrails that are associated with the key alias.
                    "x-cato-gateway-key-alias": key_alias,
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
        if isinstance(response, ModelResponse) and response.choices:
            user_email = self._resolve_cato_user_email(user_api_key_dict)
            for choice in response.choices:
                if not isinstance(choice, Choices):
                    continue
                if choice.message.content is None:
                    continue
                content = choice.message.content
                cato_output_guardrail_result = await self.call_cato_guardrail_on_output(
                    data,
                    content,
                    hook="output",
                    key_alias=user_api_key_dict.key_alias,
                    user_email=user_email,
                )
                if cato_output_guardrail_result and cato_output_guardrail_result.get(
                    "detection_message"
                ):
                    raise HTTPException(
                        status_code=400,
                        detail=cato_output_guardrail_result.get("detection_message"),
                    )
                redacted_output = (
                    cato_output_guardrail_result.get("redacted_output")
                    if cato_output_guardrail_result
                    else None
                )
                if redacted_output is not None:
                    choice.message.content = redacted_output
        return response

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        from litellm.proxy.proxy_server import StreamingCallbackError

        user_email = self._resolve_cato_user_email(user_api_key_dict)
        call_id = request_data.get("litellm_call_id")
        async with connect(
            f"{self.ws_api_base}/fw/v1/analyze/stream",
            additional_headers=self._build_cato_headers(
                hook="output",
                key_alias=user_api_key_dict.key_alias,
                user_email=user_email,
                litellm_call_id=call_id,
            ),
            **self._ws_connect_ssl_kwargs,
        ) as websocket:
            sender = asyncio.create_task(
                self.forward_the_stream_to_cato(websocket, response)
            )
            try:
                while True:
                    raw_message = await self._await_cato_message(websocket, sender)
                    result = json.loads(raw_message)
                    if verified_chunk := result.get("verified_chunk"):
                        yield ModelResponseStream.model_validate(verified_chunk)
                        continue
                    if result.get("done"):
                        return
                    if blocking_message := result.get("blocking_message"):
                        raise StreamingCallbackError(blocking_message)
                    verbose_proxy_logger.error(
                        f"Unknown message received from Cato: {result}"
                    )
                    return
            finally:
                await self._cancel_background_task(sender)

    async def _await_cato_message(
        self, websocket: ClientConnection, sender: asyncio.Task
    ) -> Any:
        """Wait for the next Cato message, surfacing a dead forwarding task instead of blocking."""
        from litellm.proxy.proxy_server import StreamingCallbackError

        recv_task = asyncio.ensure_future(websocket.recv())
        pending = {recv_task, sender} if not sender.done() else {recv_task}
        await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        if sender.done() and (sender_exc := sender.exception()) is not None:
            await self._cancel_background_task(recv_task)
            raise StreamingCallbackError(
                "Cato guardrail upstream stream failed"
            ) from sender_exc
        try:
            return await recv_task
        except ConnectionClosed as exc:
            raise StreamingCallbackError(
                "Cato guardrail connection closed unexpectedly"
            ) from exc

    async def forward_the_stream_to_cato(
        self,
        websocket: ClientConnection,
        response_iter: AsyncGenerator[Any, None],
    ) -> None:
        async for chunk in response_iter:
            if isinstance(chunk, BaseModel):
                chunk = chunk.model_dump_json()
            elif not isinstance(chunk, (str, bytes)):
                chunk = json.dumps(chunk)
            await websocket.send(chunk)
        await websocket.send(json.dumps({"done": True}))

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.cato_networks import (
            CatoNetworksGuardrailConfigModel,
        )

        return CatoNetworksGuardrailConfigModel
