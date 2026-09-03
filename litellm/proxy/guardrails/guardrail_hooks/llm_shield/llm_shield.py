# +-------------------------------------------------------------+
#
#         Use LLM Shield for reversible PII redaction
#            https://github.com/ninadphalak/LLM-Shield-Proxy
#
# +-------------------------------------------------------------+

import os
import uuid
from collections.abc import AsyncGenerator
from typing import (
    TYPE_CHECKING,
    Any,  # noqa: TID251  # **kwargs forwards verbatim to CustomGuardrail.__init__
    ClassVar,
    Final,
    Literal,
    Optional,
)

import httpx

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    get_session_id_from_request_data,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.caching.caching import DualCache
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

GUARDRAIL_NAME: Final = "llm_shield"

_DEFAULT_API_BASE: Final = "http://localhost:8000"
_REDACT_PATH: Final = "/v1/guard/redact"
_REHYDRATE_PATH: Final = "/v1/guard/rehydrate"
_REHYDRATE_STREAM_PATH: Final = "/v1/guard/rehydrate/stream"

# The session id ties a redact call to the rehydrate calls that undo it. It is
# stored on the request dict rather than on the guardrail instance: the proxy
# registers one instance process-wide, so instance attributes would be shared
# across concurrent requests.
_SESSION_METADATA_KEY: Final = "llm_shield_session_id"

_DEFAULT_TIMEOUT_SECONDS: Final = 10.0


class LLMShieldGuardrail(CustomGuardrail):
    """Redacts PII before it leaves the proxy and restores it in the response.

    Unlike a masking guardrail, the substitution is reversible. Outbound text is
    replaced with placeholders held in a session vault inside the user's own LLM
    Shield deployment; the model's reply is then restored so the end user sees the
    original values while the provider never received them.

    Streaming is restored incrementally rather than by buffering the response. LLM
    Shield holds back only the trailing characters that could still turn out to be
    part of a placeholder, so tokens are forwarded as they arrive and a placeholder
    split across two chunks is never emitted in fragments.
    """

    # Our redaction and restoration run in the native lifecycle hooks below. Without
    # this the proxy would route every event through the unified apply_guardrail path
    # and the streaming hook would never fire.
    use_native_lifecycle_hooks: ClassVar[bool] = True

    def __init__(
        self,
        guardrail_name: str = GUARDRAIL_NAME,
        api_base: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.async_handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.GuardrailCallback)
        self.api_base: Final = (api_base or os.environ.get("LLM_SHIELD_API_BASE") or _DEFAULT_API_BASE).rstrip("/")
        self.api_key: Final = api_key or os.environ.get("LLM_SHIELD_API_KEY")
        super().__init__(guardrail_name=guardrail_name, **kwargs)

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        return [GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call]

    # --- transport ---------------------------------------------------------------

    def _headers(self, session_id: str) -> dict:
        headers = {"Content-Type": "application/json", "X-Session-ID": session_id}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _call_shield(self, path: str, session_id: str, payload: dict) -> dict:
        """Posts to LLM Shield, failing closed on any transport or status error.

        A redaction guardrail that fails open sends the very data it exists to
        protect to a third-party provider, so an unreachable or erroring shield
        blocks the request instead of passing it through.
        """
        try:
            response = await self.async_handler.post(
                f"{self.api_base}{path}",
                headers=self._headers(session_id),
                json=payload,
                timeout=_DEFAULT_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            verbose_proxy_logger.exception("LLM Shield returned %s for %s", exc.response.status_code, path)
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=f"LLM Shield returned {exc.response.status_code}; blocking the request.",
            ) from exc
        except Exception as exc:
            verbose_proxy_logger.exception("LLM Shield call to %s failed", path)
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message="LLM Shield is unreachable; blocking the request.",
            ) from exc

    async def _redact(self, texts: list, session_id: str) -> list:
        body = await self._call_shield(_REDACT_PATH, session_id, {"texts": texts})
        return self._same_length_or_raise(body.get("texts"), texts, "redact")

    async def _rehydrate(self, texts: list, session_id: str) -> list:
        body = await self._call_shield(_REHYDRATE_PATH, session_id, {"texts": texts})
        return self._same_length_or_raise(body.get("texts"), texts, "rehydrate")

    def _same_length_or_raise(self, returned: Any, sent: list, operation: str) -> list:
        """Guards the positional mapping the callers rely on to write results back."""
        if not isinstance(returned, list) or len(returned) != len(sent):
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=f"LLM Shield {operation} returned an unexpected payload; blocking the request.",
            )
        return returned

    # --- session ------------------------------------------------------------------

    def _session_id(self, data: dict) -> str:
        """Returns a session id stable across this request's hooks."""
        metadata = data.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            return f"litellm-{uuid.uuid4().hex}"
        existing = metadata.get(_SESSION_METADATA_KEY)
        if isinstance(existing, str) and existing:
            return existing
        session_id = get_session_id_from_request_data(data) or f"litellm-{uuid.uuid4().hex}"
        metadata[_SESSION_METADATA_KEY] = session_id
        return session_id

    # --- message traversal --------------------------------------------------------

    @staticmethod
    def _locate_texts(messages: list) -> list:
        """Finds every text span in a message list.

        Returns ``(message_index, part_index_or_None, text)``. The list form is the
        multimodal shape, where only ``text`` parts carry redactable content.
        """
        located = []
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str) and content:
                located.append((message_index, None, content))
            elif isinstance(content, list):
                for part_index, part in enumerate(content):
                    if not isinstance(part, dict) or part.get("type") != "text":
                        continue
                    text = part.get("text")
                    if isinstance(text, str) and text:
                        located.append((message_index, part_index, text))
        return located

    @staticmethod
    def _write_back(messages: list, located: list, replacements: list) -> None:
        for (message_index, part_index, _), replacement in zip(located, replacements):
            if part_index is None:
                messages[message_index]["content"] = replacement
            else:
                messages[message_index]["content"][part_index]["text"] = replacement

    # --- hooks --------------------------------------------------------------------

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: "DualCache",
        data: dict,
        call_type: str,
    ) -> dict | None:
        """Replaces PII in the outbound messages with vault placeholders."""
        if self.should_run_guardrail(data=data, event_type=GuardrailEventHooks.pre_call) is not True:
            return data

        messages = data.get("messages")
        if not isinstance(messages, list):
            return data

        located = self._locate_texts(messages)
        if not located:
            return data

        redacted = await self._redact([text for _, _, text in located], self._session_id(data))
        self._write_back(messages, located, redacted)
        return data

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
    ) -> Any:
        """Restores the original values in a non-streaming response."""
        if self.should_run_guardrail(data=data, event_type=GuardrailEventHooks.post_call) is not True:
            return response

        choices = getattr(response, "choices", None)
        if not choices:
            return response

        pending = []
        for choice in choices:
            message = getattr(choice, "message", None)
            content = getattr(message, "content", None)
            if isinstance(content, str) and content:
                pending.append((message, content))

        if not pending:
            return response

        restored = await self._rehydrate([text for _, text in pending], self._session_id(data))
        for (message, _), replacement in zip(pending, restored):
            message.content = replacement
        return response

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[Any, None]:
        """Restores original values incrementally, without buffering the stream.

        The carry-over window is a local of this generator, so it is scoped to one
        stream and cannot leak between concurrent requests. LLM Shield returns the
        text that is safe to emit now plus the trailing characters it is still
        holding, which are sent back with the next delta.
        """
        if self.should_run_guardrail(data=request_data, event_type=GuardrailEventHooks.post_call) is not True:
            async for chunk in response:
                yield chunk
            return

        session_id = self._session_id(request_data)
        carry = ""
        last_chunk = None

        async for chunk in response:
            last_chunk = chunk
            delta = self._stream_delta(chunk)
            text = getattr(delta, "content", None) if delta is not None else None
            is_final = self._is_final_chunk(chunk)

            if not isinstance(text, str) or not text:
                # Nothing to restore in this chunk, but a final chunk still has to
                # flush whatever the window is holding.
                if is_final and carry:
                    body = await self._stream_step("", carry, True, session_id)
                    carry = body["carry"]
                    if body["text"] and delta is not None:
                        delta.content = body["text"]
                yield chunk
                continue

            body = await self._stream_step(text, carry, is_final, session_id)
            carry = body["carry"]
            delta.content = body["text"]
            yield chunk

        # A stream that ended without a finish_reason can still leave text held back.
        if carry and last_chunk is not None:
            body = await self._stream_step("", carry, True, session_id)
            if body["text"]:
                trailing = last_chunk.model_copy(deep=True)
                trailing_delta = self._stream_delta(trailing)
                if trailing_delta is not None:
                    trailing_delta.content = body["text"]
                    yield trailing

    async def _stream_step(self, text: str, carry: str, final: bool, session_id: str) -> dict:
        body = await self._call_shield(
            _REHYDRATE_STREAM_PATH,
            session_id,
            {"text": text, "carry": carry, "final": final},
        )
        emitted = body.get("text")
        remaining = body.get("carry")
        if not isinstance(emitted, str) or not isinstance(remaining, str):
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message="LLM Shield stream rehydration returned an unexpected payload.",
            )
        return {"text": emitted, "carry": remaining}

    @staticmethod
    def _stream_delta(chunk: Any) -> Any:
        choices = getattr(chunk, "choices", None)
        if not choices:
            return None
        return getattr(choices[0], "delta", None)

    @staticmethod
    def _is_final_chunk(chunk: Any) -> bool:
        choices = getattr(chunk, "choices", None)
        if not choices:
            return False
        return bool(getattr(choices[0], "finish_reason", None))

    # --- unified API (powers the UI "Test guardrail" button) -----------------------

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        texts = inputs.get("texts")
        if not texts:
            return inputs

        session_id = self._session_id(request_data)
        if input_type == "request":
            inputs["texts"] = await self._redact(list(texts), session_id)
        else:
            inputs["texts"] = await self._rehydrate(list(texts), session_id)
        return inputs
