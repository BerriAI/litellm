# +-------------------------------------------------------------+
#
#            FangcunGuard Guardrail for LiteLLM
#              https://www.fangcunleap.com
#
# +-------------------------------------------------------------+

import asyncio
import json
import os
from typing import TYPE_CHECKING, AsyncGenerator, Literal, Optional, Union

from fastapi import HTTPException

from litellm import DualCache
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
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

# Bounds to keep one request from exhausting the HTTP pool / guardrail quota
# (fan-out) or process memory (streaming buffer).
MAX_TEXTS_PER_REQUEST = 100
MAX_CONCURRENT_CHECKS = 10
MAX_STREAM_CHUNKS = 5000


class FangcunGuardMissingSecrets(Exception):
    """Raised when no FangcunGuard API key is configured."""


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
        fail_open: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("supported_event_hooks", list(self.get_supported_event_hooks()))
        self.async_handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.GuardrailCallback)
        self.fangcun_api_key = api_key or os.environ.get("FANGCUN_API_KEY")
        if not self.fangcun_api_key:
            raise FangcunGuardMissingSecrets(
                "FangcunGuard API key not set. Pass `api_key` in litellm_params or set the FANGCUN_API_KEY environment variable."
            )
        self.fangcun_api_base = api_base or os.environ.get("FANGCUN_API_BASE") or DEFAULT_API_BASE
        # When the API cannot return an authoritative verdict (non-200, network
        # error), fail closed by default. Set fail_open=True to allow instead.
        self.fail_open = fail_open
        super().__init__(**kwargs)

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.during_call,
            GuardrailEventHooks.post_call,
        ]

    def _extract_texts(self, data: dict, call_type: Optional[str] = None) -> list[str]:
        """Extract all user/assistant text from a request across supported shapes.

        Uses the shared ``get_guardrails_messages_for_call_type`` helper (handles
        /chat/completions, /messages, /responses) and falls back to the
        ``prompt`` (text completions) and ``input`` fields so alternate request
        shapes cannot bypass scanning.
        """
        texts: list[str] = []

        messages: Optional[list] = None
        if call_type is not None:
            try:
                messages = self.get_guardrails_messages_for_call_type(call_type=call_type, data=data)
            except Exception:  # noqa: BLE001 - fall back to raw messages on any extraction error
                messages = None
        if messages is None:
            messages = data.get("messages")

        for message in messages or []:
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        texts.append(part.get("text", ""))

        # Text completions store input under "prompt"; some endpoints use "input".
        for key in ("prompt", "input"):
            value = data.get(key)
            if isinstance(value, str):
                texts.append(value)
            elif isinstance(value, list):
                texts.extend([v for v in value if isinstance(v, str)])

        return [t for t in texts if t and t.strip()]

    async def _check_text(self, text: str, request_data: dict) -> None:
        """Call FangcunGuard; raise HTTPException(400) if the text is unsafe.

        Fails closed on any error that prevents an authoritative verdict, unless
        ``fail_open`` is set.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.fangcun_api_key}",
        }

        # Merge dynamic params first, then set the trusted text last so a
        # user-supplied `text` in extra_body can never replace the real input.
        body = self.get_guardrail_dynamic_request_body_params(request_data=request_data)
        body["text"] = text

        try:
            response = await self.async_handler.post(
                url=self.fangcun_api_base.rstrip("/") + "/guard/context",
                json=body,
                headers=headers,
            )
        except Exception as e:  # noqa: BLE001 - any error means no verdict; fail closed
            self._handle_unavailable(f"request error: {e}")
            return

        verbose_proxy_logger.debug("FangcunGuard response: %s", response.text)

        if response.status_code != 200:
            self._handle_unavailable(f"non-200 response ({response.status_code})")
            return

        result = response.json()

        if not bool(result.get("is_safe", True)):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Violated FangcunGuard content policy",
                    "fangcunguard_response": result,
                },
            )

    def _handle_unavailable(self, reason: str) -> None:
        """Fail closed (default) or open when no verdict can be obtained."""
        if self.fail_open:
            verbose_proxy_logger.warning("FangcunGuard: %s; fail_open=True, allowing request", reason)
            return
        raise HTTPException(
            status_code=500,
            detail={
                "error": "FangcunGuard is unavailable and fail_open is disabled",
                "reason": reason,
            },
        )

    async def _scan_texts(self, texts: list[str], request_data: dict) -> None:
        if len(texts) > MAX_TEXTS_PER_REQUEST:
            # Too many texts to scan authoritatively; treat like an outage so
            # the fail_open policy decides (fail closed by default).
            self._handle_unavailable(f"too many texts to scan ({len(texts)} > {MAX_TEXTS_PER_REQUEST})")
            return

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_CHECKS)

        async def _bounded(text: str) -> None:
            async with semaphore:
                await self._check_text(text, request_data)

        await asyncio.gather(*[_bounded(text) for text in texts])

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

        await self._scan_texts(self._extract_texts(data, call_type=call_type), request_data=data)
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

        await self._scan_texts(self._extract_texts(data, call_type=call_type), request_data=data)
        add_guardrail_to_applied_guardrails_header(request_data=data, guardrail_name=self.guardrail_name)
        return data

    @staticmethod
    def _texts_from_content(content) -> list[str]:
        """Pull text out of a message ``content`` (str or list-of-parts)."""
        if isinstance(content, str):
            return [content]
        if isinstance(content, list):
            return [part["text"] for part in content if isinstance(part, dict) and isinstance(part.get("text"), str)]
        return []

    @staticmethod
    def _texts_from_tool_calls(tool_calls) -> list[str]:
        """Pull ``function.arguments`` strings out of tool/function calls.

        Prohibited text can be smuggled into tool-call arguments, so these are
        scanned alongside plain content.
        """
        texts: list[str] = []
        for tool_call in tool_calls or []:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if isinstance(function, dict) and isinstance(function.get("arguments"), str):
                texts.append(function["arguments"])
        return texts

    @classmethod
    def _extract_response_texts(cls, response) -> list[str]:
        """Extract assistant text from any supported response type.

        Handles chat completions (``choices[].message.content`` and
        ``tool_calls[].function.arguments``), text completions
        (``choices[].text``), and the Responses API (``output[].content[].text``
        and function-call ``arguments``), so no output path is skipped.
        """
        response_dict: dict = {}
        if hasattr(response, "model_dump"):
            try:
                response_dict = response.model_dump()
            except Exception:  # noqa: BLE001 - fall back to empty on any dump error
                response_dict = {}
        elif isinstance(response, dict):
            response_dict = response

        texts: list[str] = []

        # Chat + text completions both live under "choices".
        for choice in response_dict.get("choices", []) or []:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict):
                texts.extend(cls._texts_from_content(message.get("content")))
                texts.extend(cls._texts_from_tool_calls(message.get("tool_calls")))
            if isinstance(choice.get("text"), str):
                texts.append(choice["text"])

        # Responses API output items (message content + function-call arguments).
        for item in response_dict.get("output", []) or []:
            if not isinstance(item, dict):
                continue
            texts.extend(cls._texts_from_content(item.get("content")))
            if isinstance(item.get("arguments"), str):
                texts.append(item["arguments"])

        return [t for t in texts if t and t.strip()]

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

        response_texts = self._extract_response_texts(response)
        if response_texts:
            await self._scan_texts(response_texts, request_data=data)
            add_guardrail_to_applied_guardrails_header(request_data=data, guardrail_name=self.guardrail_name)

    @log_guardrail_information
    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response,
        request_data: dict,
    ) -> AsyncGenerator:
        """Buffer the streamed output, scan it, then release it.

        Streaming responses would otherwise bypass ``async_post_call_success_hook``
        enforcement, so we assemble all chunks, scan the full text, and only then
        yield. On a violation we emit an SSE error event instead of raising (a
        raise inside a generator surfaces as a generic 500).
        """
        from litellm.main import stream_chunk_builder
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        event_type = GuardrailEventHooks.post_call
        if self.should_run_guardrail(data=request_data, event_type=event_type) is not True:
            async for chunk in response:
                yield chunk
            return

        all_chunks = []
        exceeded = False
        async for chunk in response:
            all_chunks.append(chunk)
            if len(all_chunks) > MAX_STREAM_CHUNKS:
                exceeded = True
                break

        if exceeded:
            # Too large to buffer/scan authoritatively. Fail closed by default;
            # if fail_open, release what we have and pass the rest through
            # un-scanned rather than holding it all in memory.
            if not self.fail_open:
                yield f"data: {json.dumps({'error': {'message': 'FangcunGuard: streaming response too large to scan', 'code': '500'}})}\n\n"
                return
            verbose_proxy_logger.warning(
                "FangcunGuard: stream exceeded %s chunks; fail_open=True, passing through",
                MAX_STREAM_CHUNKS,
            )
            for chunk in all_chunks:
                yield chunk
            async for chunk in response:
                yield chunk
            return

        assembled = stream_chunk_builder(chunks=all_chunks)
        response_texts = self._extract_response_texts(assembled) if assembled else []

        if response_texts:
            try:
                await self._scan_texts(response_texts, request_data=request_data)
            except HTTPException as e:
                detail = e.detail if isinstance(e.detail, dict) else {"message": str(e.detail)}
                error_obj = dict(detail) if isinstance(detail, dict) else {"message": str(detail)}
                error_obj["code"] = str(e.status_code)
                yield f"data: {json.dumps({'error': error_obj})}\n\n"
                return
            add_guardrail_to_applied_guardrails_header(request_data=request_data, guardrail_name=self.guardrail_name)

        for chunk in all_chunks:
            yield chunk

    @staticmethod
    def get_config_model() -> Optional[type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.fangcunguard import (
            FangcunGuardrailConfigModel,
        )

        return FangcunGuardrailConfigModel
