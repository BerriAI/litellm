# +-------------------------------------------------------------+
#
#         Use LLM Shield for reversible PII redaction
#            https://github.com/ninadphalak/LLM-Shield-Proxy
#
# +-------------------------------------------------------------+

import os
import uuid
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,  # noqa: TID251  # **kwargs forwards verbatim to CustomGuardrail.__init__
    ClassVar,
    Final,
    Literal,
    Optional,
    TypeAlias,
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

# The proxy's own request dict. Mutable by design: a pre-call guardrail rewrites
# the caller's payload in place, which is the entire point of the hook.
# mutable-ok: the shape is fixed by CustomLogger's hook signatures.
MutableRequest: TypeAlias = dict

# A JSON body on its way to httpx, which requires a real dict rather than a view.
# mutable-ok: handed straight to the HTTP client.
JsonBody: TypeAlias = dict

# One redactable span: the text as it stands, and the write that puts the
# replacement back where it came from.
_Slot: TypeAlias = tuple[str, Callable[[str], None]]  # mutable-ok: Callable's param list.


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
        **kwargs: Any,  # noqa: LIT008  # kwargs-ok: forwarded verbatim to CustomGuardrail.__init__
    ) -> None:
        self.async_handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.GuardrailCallback)
        self.api_base: Final = (api_base or os.environ.get("LLM_SHIELD_API_BASE") or _DEFAULT_API_BASE).rstrip("/")
        self.api_key: Final = api_key or os.environ.get("LLM_SHIELD_API_KEY")
        super().__init__(guardrail_name=guardrail_name, **kwargs)

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:  # mutable-ok: parent's signature.
        return [GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call]  # mutable-ok: parent's signature.

    # --- transport ---------------------------------------------------------------

    def _headers(self, session_id: str) -> JsonBody:
        headers: Final[JsonBody] = {  # mutable-ok: httpx requires a real dict.
            "Content-Type": "application/json",
            "X-Session-ID": session_id,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _call_shield(self, path: str, session_id: str, payload: JsonBody) -> Mapping[str, object]:
        """Posts to LLM Shield, failing closed on any transport or status error.

        A redaction guardrail that fails open sends the very data it exists to
        protect to a third-party provider, so an unreachable or erroring shield
        blocks the request instead of passing it through.
        """
        try:
            response: Final = await self.async_handler.post(
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

    async def _redact(self, texts: Sequence[str], session_id: str) -> Sequence[str]:
        payload: Final[JsonBody] = {"texts": list(texts)}  # mutable-ok: JSON body for httpx.
        body: Final = await self._call_shield(_REDACT_PATH, session_id, payload)
        return self._same_length_or_raise(body.get("texts"), texts, "redact")

    async def _rehydrate(self, texts: Sequence[str], session_id: str) -> Sequence[str]:
        payload: Final[JsonBody] = {"texts": list(texts)}  # mutable-ok: JSON body for httpx.
        body: Final = await self._call_shield(_REHYDRATE_PATH, session_id, payload)
        return self._same_length_or_raise(body.get("texts"), texts, "rehydrate")

    def _same_length_or_raise(self, returned: object, sent: Sequence[str], operation: str) -> Sequence[str]:
        """Guards the positional mapping the callers rely on to write results back."""
        if not isinstance(returned, list) or len(returned) != len(sent):
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=f"LLM Shield {operation} returned an unexpected payload; blocking the request.",
            )
        return tuple(returned)

    # --- session ------------------------------------------------------------------

    def _session_id(self, data: MutableRequest) -> str:
        """Returns a session id stable across this request's hooks."""
        metadata: Final = data.setdefault("metadata", {})  # mutable-ok: per-request store.
        if not isinstance(metadata, dict):
            return f"litellm-{uuid.uuid4().hex}"
        existing: Final = metadata.get(_SESSION_METADATA_KEY)
        if isinstance(existing, str) and existing:
            return existing
        session_id: Final = get_session_id_from_request_data(data) or f"litellm-{uuid.uuid4().hex}"
        metadata[_SESSION_METADATA_KEY] = session_id
        return session_id

    # --- request traversal --------------------------------------------------------

    @staticmethod
    def _locate_request_texts(data: MutableRequest) -> Sequence[_Slot]:
        """Finds every redactable span in an outbound request.

        Returns ``(text, write)`` pairs. Any shape missed here reaches the provider
        in the clear, so this walks all of the request shapes that carry caller text:

        - chat ``messages``, both string and multimodal list ``content``
        - tool call ``arguments``, which routinely carry the values a user asked
          the model to look up
        - the Responses API ``input``, as a bare string or a list of items
        """
        slots: Final[list[_Slot]] = []  # mutable-ok: accumulator, frozen on return.

        def add(container: MutableRequest, key: str, value: object) -> None:
            if isinstance(value, str) and value:
                slots.append((value, lambda new, c=container, k=key: c.__setitem__(k, new)))

        def add_content(container: MutableRequest) -> None:
            """Adds `content`, which is either a string or a list of typed parts."""
            content: Final = container.get("content")
            if isinstance(content, str):
                add(container, "content", content)
                return
            for part in content if isinstance(content, list) else ():
                if isinstance(part, dict):
                    add(part, "text", part.get("text"))

        def add_tool_calls(message: MutableRequest) -> None:
            for tool_call in message.get("tool_calls") or ():
                function = tool_call.get("function") if isinstance(tool_call, dict) else None
                if isinstance(function, dict):
                    add(function, "arguments", function.get("arguments"))

        for message in data.get("messages") or ():
            if isinstance(message, dict):
                add_content(message)
                add_tool_calls(message)

        request_input: Final = data.get("input")
        if isinstance(request_input, str):
            add(data, "input", request_input)
        else:
            for item in request_input if isinstance(request_input, list) else ():
                if isinstance(item, dict):
                    add_content(item)

        return tuple(slots)

    # --- hooks --------------------------------------------------------------------

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: "DualCache",
        data: MutableRequest,
        call_type: str,
    ) -> MutableRequest | None:
        """Replaces PII anywhere in the outbound request with vault placeholders."""
        if self.should_run_guardrail(data=data, event_type=GuardrailEventHooks.pre_call) is not True:
            return data

        slots: Final = self._locate_request_texts(data)
        if not slots:
            return data

        redacted: Final = await self._redact(tuple(text for text, _ in slots), self._session_id(data))
        for (_, write), replacement in zip(slots, redacted):
            write(replacement)
        return data

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: MutableRequest,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
    ) -> Any:
        """Restores the original values in a non-streaming response."""
        if self.should_run_guardrail(data=data, event_type=GuardrailEventHooks.post_call) is not True:
            return response

        if self._is_anthropic_message_response(response):
            return await self._restore_anthropic_response(response, data)

        text_blocks: Final = self._responses_api_text_blocks(response)
        if text_blocks:
            return await self._restore_responses_api_response(response, text_blocks, data)

        choices: Final = getattr(response, "choices", None)
        if not choices:
            return response

        pending: Final = tuple(
            (choice.message, choice.message.content)
            for choice in choices
            if getattr(choice, "message", None) is not None
            and isinstance(getattr(choice.message, "content", None), str)
            and choice.message.content
        )
        if not pending:
            return response

        restored: Final = await self._rehydrate(tuple(text for _, text in pending), self._session_id(data))
        for (message, _), replacement in zip(pending, restored):
            message.content = replacement
        return response

    @staticmethod
    def _is_anthropic_message_response(response: object) -> bool:
        """Anthropic's native /v1/messages reply arrives as a plain dict."""
        return (
            isinstance(response, dict)
            and response.get("type") == "message"
            and isinstance(response.get("content"), list)
        )

    async def _restore_anthropic_response(self, response: MutableRequest, data: MutableRequest) -> MutableRequest:
        """Restores text blocks in an Anthropic native message reply.

        This shape has no `choices`, so without its own branch the reply would go
        back to the caller still carrying placeholders.
        """
        blocks: Final = tuple(
            block
            for block in response["content"]
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
        )
        if not blocks:
            return response

        restored: Final = await self._rehydrate(tuple(block["text"] for block in blocks), self._session_id(data))
        for block, replacement in zip(blocks, restored):
            block["text"] = replacement
        return response

    @staticmethod
    def _responses_api_text_blocks(response: object) -> Sequence[object]:
        """Text blocks in a Responses API reply.

        That shape carries `output` items rather than `choices`, so it needs its own
        walk; without one the reply goes back to the caller still holding
        placeholders even though the request was redacted correctly. Blocks come
        through as dicts or as objects depending on how far the reply has been
        deserialised, so both are handled.
        """
        blocks: Final[list[object]] = []  # mutable-ok: accumulator, frozen on return.
        for item in getattr(response, "output", None) or ():
            for block in getattr(item, "content", None) or ():
                if isinstance(block, dict):
                    if isinstance(block.get("text"), str) and block["text"]:
                        blocks.append(block)
                elif isinstance(getattr(block, "text", None), str) and block.text:
                    blocks.append(block)
        return tuple(blocks)

    @staticmethod
    def _block_text(block: object) -> str:
        return block["text"] if isinstance(block, dict) else block.text

    async def _restore_responses_api_response(
        self, response: Any, blocks: Sequence[object], data: MutableRequest
    ) -> Any:
        """Puts the original values back into a Responses API reply."""
        restored: Final = await self._rehydrate(
            tuple(self._block_text(block) for block in blocks), self._session_id(data)
        )
        for block, replacement in zip(blocks, restored):
            if isinstance(block, dict):
                block["text"] = replacement
            else:
                block.text = replacement
        return response

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: MutableRequest,
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

        session_id: Final = self._session_id(request_data)
        carry = ""  # rebind-ok: the sliding window advances with every delta.
        last_chunk = None  # rebind-ok: tracks the most recent chunk for the final flush.

        async for chunk in response:
            last_chunk = chunk
            delta = self._stream_delta(chunk)
            text = getattr(delta, "content", None) if delta is not None else None
            is_final = self._is_final_chunk(chunk)

            if not isinstance(text, str) or not text:
                # Nothing to restore in this chunk, but a final chunk still has to
                # flush whatever the window is holding.
                if is_final and carry:
                    emitted, carry = await self._stream_step("", carry, True, session_id)
                    if emitted and delta is not None:
                        delta.content = emitted
                yield chunk
                continue

            emitted, carry = await self._stream_step(text, carry, is_final, session_id)
            delta.content = emitted
            yield chunk

        # A stream that ended without a finish_reason can still leave text held back.
        if carry and last_chunk is not None:
            flushed: Final = await self._stream_step("", carry, True, session_id)
            trailing_text, carry = flushed  # rebind-ok: window advances.
            if trailing_text:
                trailing: Final = last_chunk.model_copy(deep=True)
                trailing_delta: Final = self._stream_delta(trailing)
                if trailing_delta is not None:
                    trailing_delta.content = trailing_text
                    yield trailing

    async def _stream_step(self, text: str, carry: str, final: bool, session_id: str) -> tuple[str, str]:
        """Returns ``(text safe to emit now, window still being held)``."""
        body: Final = await self._call_shield(
            _REHYDRATE_STREAM_PATH,
            session_id,
            # mutable-ok: JSON request body for httpx.
            {"text": text, "carry": carry, "final": final},  # mutable-ok: JSON request body for httpx.
        )
        emitted: Final = body.get("text")
        remaining: Final = body.get("carry")
        if not isinstance(emitted, str) or not isinstance(remaining, str):
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message="LLM Shield stream rehydration returned an unexpected payload.",
            )
        return emitted, remaining

    @staticmethod
    def _stream_delta(chunk: object) -> Any:
        choices: Final = getattr(chunk, "choices", None)
        if not choices:
            return None
        return getattr(choices[0], "delta", None)

    @staticmethod
    def _is_final_chunk(chunk: object) -> bool:
        choices: Final = getattr(chunk, "choices", None)
        if not choices:
            return False
        return bool(getattr(choices[0], "finish_reason", None))

    # --- unified API (powers the UI "Test guardrail" button) -----------------------

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: MutableRequest,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        texts: Final = inputs.get("texts")
        if not texts:
            return inputs

        session_id: Final = self._session_id(request_data)
        replaced: Final = (
            await self._redact(tuple(texts), session_id)
            if input_type == "request"
            else await self._rehydrate(tuple(texts), session_id)
        )
        # Return a new mapping rather than rewriting the caller's, so this stays a
        # pure transform of the inputs it was handed.
        merged: Final[JsonBody] = {**inputs, "texts": list(replaced)}  # mutable-ok: TypedDict.
        return merged
