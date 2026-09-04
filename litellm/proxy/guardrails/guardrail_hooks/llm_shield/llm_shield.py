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

# Vault ids are minted here and never derived from anything the caller sends. The
# vault holds the plaintext behind every placeholder, so an id a caller could
# supply or guess would let one user rehydrate another user's values by getting a
# placeholder echoed back. The per-process prefix means a caller cannot even name
# a vault this process uses.
_VAULT_PREFIX: Final = f"litellm-{uuid.uuid4().hex}"

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

# The accumulator the collectors below append into. It never escapes
# _locate_request_texts, which freezes it into a tuple before returning.
_SlotSink: TypeAlias = list[_Slot]  # mutable-ok: accumulator passed between collectors.

# Sliding windows keyed by streaming choice index, threaded through one stream.
_CarryWindows: TypeAlias = dict  # mutable-ok: per-choice windows advanced in place.

# A caller-owned list whose entries are rewritten in place, such as a Completions
# `prompt` sent as an array of strings.
MutableSeq: TypeAlias = list  # mutable-ok: the request payload's own list.


def _collect(container: MutableRequest, key: str, slots: _SlotSink) -> None:
    """Records the string at `key`, along with the write that replaces it."""
    value: Final = container.get(key)
    if isinstance(value, str) and value:
        slots.append((value, lambda new, c=container, k=key: c.__setitem__(k, new)))


def _collect_entry(entries: MutableSeq, index: int, slots: _SlotSink) -> None:
    """Records a string held directly in a list, rather than under a key."""
    value: Final = entries[index]
    if isinstance(value, str) and value:
        slots.append((value, lambda new, e=entries, i=index: e.__setitem__(i, new)))


def _collect_prompt(data: MutableRequest, slots: _SlotSink) -> None:
    """The Completions API sends its text in a top-level `prompt`."""
    prompt: Final = data.get("prompt")
    if isinstance(prompt, str):
        _collect(data, "prompt", slots)
        return
    if not isinstance(prompt, list):
        return
    for index in range(len(prompt)):
        _collect_entry(prompt, index, slots)


def _collect_content(container: MutableRequest, slots: _SlotSink) -> None:
    """`content` is either a string or the multimodal list of typed parts."""
    content: Final = container.get("content")
    if isinstance(content, str):
        _collect(container, "content", slots)
        return
    for part in content if isinstance(content, list) else ():
        if isinstance(part, dict):
            _collect(part, "text", slots)


def _collect_tool_arguments(message: MutableRequest, slots: _SlotSink) -> None:
    """Tool arguments carry the values a user asked the model to act on."""
    for tool_call in message.get("tool_calls") or ():
        function = tool_call.get("function") if isinstance(tool_call, dict) else None  # rebind-ok: loop variable.
        if isinstance(function, dict):
            _collect(function, "arguments", slots)
    legacy: Final = message.get("function_call")
    if isinstance(legacy, dict):
        _collect(legacy, "arguments", slots)


def _collect_system(data: MutableRequest, slots: _SlotSink) -> None:
    """Anthropic's /v1/messages carries its system prompt at the top level."""
    system: Final = data.get("system")
    if isinstance(system, str):
        _collect(data, "system", slots)
        return
    for part in system if isinstance(system, list) else ():
        if isinstance(part, dict):
            _collect(part, "text", slots)


def _collect_responses_fields(data: MutableRequest, slots: _SlotSink) -> None:
    """The Responses API sends text outside `messages`, in `instructions` and `input`."""
    _collect(data, "instructions", slots)
    request_input: Final = data.get("input")
    if isinstance(request_input, str):
        _collect(data, "input", slots)
        return
    if not isinstance(request_input, list):
        return
    for index, item in enumerate(request_input):
        if isinstance(item, str):
            # The embeddings and moderations shape: `input` as an array of strings.
            _collect_entry(request_input, index, slots)
            continue
        if not isinstance(item, dict):
            continue
        _collect_content(item, slots)
        # A function_call item holds `arguments`; a function_call_output holds `output`.
        _collect(item, "arguments", slots)
        _collect(item, "output", slots)


def _choice_index(choice: object) -> int:
    """Streaming choices are matched across chunks by their index."""
    index: Final = getattr(choice, "index", 0)
    return index if isinstance(index, int) else 0


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

    @staticmethod
    def _mint_session_id(data: MutableRequest) -> str:
        """Mints a vault id for this request, overwriting anything already there.

        Redaction and restoration both happen inside one request/response pair, so
        a fresh id per request is all that is needed, and it is what keeps one
        caller from reaching another caller's vault.
        """
        session_id: Final = f"{_VAULT_PREFIX}-{uuid.uuid4().hex}"
        metadata: Final = data.setdefault("metadata", {})  # mutable-ok: per-request store.
        if isinstance(metadata, dict):
            metadata[_SESSION_METADATA_KEY] = session_id
        return session_id

    @staticmethod
    def _session_id(data: MutableRequest) -> str:
        """Reads back the vault id minted while redacting this request.

        Falls back to an unused id rather than to anything the caller supplied: a
        reply that cannot be restored is a visible placeholder, while trusting a
        caller-supplied id would hand them someone else's plaintext.
        """
        metadata: Final = data.get("metadata")
        existing: Final = metadata.get(_SESSION_METADATA_KEY) if isinstance(metadata, dict) else None
        if isinstance(existing, str) and existing.startswith(_VAULT_PREFIX):
            return existing
        return f"{_VAULT_PREFIX}-{uuid.uuid4().hex}"

    # --- request traversal --------------------------------------------------------

    @staticmethod
    def _locate_request_texts(data: MutableRequest) -> Sequence[_Slot]:
        """Finds every redactable span in an outbound request.

        Anything missed here reaches the provider in the clear while the guardrail
        still reports as enabled, so the walk covers every request shape that
        carries caller text.
        """
        slots: Final[_SlotSink] = []  # mutable-ok: accumulator, frozen on return.
        for message in data.get("messages") or ():
            if isinstance(message, dict):
                _collect_content(message, slots)
                _collect_tool_arguments(message, slots)
        _collect_responses_fields(data, slots)
        _collect_prompt(data, slots)
        _collect_system(data, slots)
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

        redacted: Final = await self._redact(tuple(text for text, _ in slots), self._mint_session_id(data))
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

        Each choice is its own token stream, so the sliding window is tracked per
        choice index. One shared window would splice the characters held back for
        one choice onto the next. The windows are locals of this generator, so they
        are scoped to a single stream and cannot leak between concurrent requests.
        """
        if self.should_run_guardrail(data=request_data, event_type=GuardrailEventHooks.post_call) is not True:
            async for chunk in response:
                yield chunk
            return

        session_id: Final = self._session_id(request_data)
        carries: Final[dict] = {}  # mutable-ok: per-choice windows, local to this stream.
        last_chunk = None  # rebind-ok: tracks the most recent chunk for the final flush.

        async for chunk in response:
            last_chunk = chunk
            for choice in getattr(chunk, "choices", None) or ():
                await self._restore_choice(choice, carries, session_id)
            yield chunk

        # A stream that ended without a finish_reason can still leave text held back.
        if last_chunk is not None and any(carries.values()):
            trailing: Final = last_chunk.model_copy(deep=True)
            if await self._flush_trailing(trailing, carries, session_id):
                yield trailing

    async def _restore_choice(self, choice: Any, carries: _CarryWindows, session_id: str) -> None:
        """Restores one choice's delta, advancing that choice's own window."""
        delta: Final = getattr(choice, "delta", None)
        if delta is None:
            return
        index: Final = _choice_index(choice)
        carry: Final = carries.get(index, "")
        text: Final = getattr(delta, "content", None)
        is_final: Final = bool(getattr(choice, "finish_reason", None))

        if not isinstance(text, str) or not text:
            # Nothing to restore here, but a final chunk still has to flush the window.
            if is_final and carry:
                flushed, flushed_carry = await self._stream_step("", carry, True, session_id)
                carries[index] = flushed_carry  # rebind-ok: this choice's window advances.
                if flushed:
                    delta.content = flushed
            return

        emitted, remaining = await self._stream_step(text, carry, is_final, session_id)
        carries[index] = remaining  # rebind-ok: this choice's window advances.
        delta.content = emitted

    async def _flush_trailing(self, trailing: Any, carries: _CarryWindows, session_id: str) -> bool:
        """Empties every still-held window into a copy of the last chunk."""
        emitted_any = False  # rebind-ok: set once any choice contributes text.
        for choice in getattr(trailing, "choices", None) or ():
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            index = _choice_index(choice)
            carry = carries.get(index, "")
            if not carry:
                delta.content = None
                continue
            text, remaining = await self._stream_step("", carry, True, session_id)
            carries[index] = remaining  # rebind-ok: this choice's window advances.
            delta.content = text or None
            emitted_any = emitted_any or bool(text)
        return emitted_any

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

        replaced: Final = (
            await self._redact(tuple(texts), self._mint_session_id(request_data))
            if input_type == "request"
            else await self._rehydrate(tuple(texts), self._session_id(request_data))
        )
        # Return a new mapping rather than rewriting the caller's, so this stays a
        # pure transform of the inputs it was handed.
        merged: Final[JsonBody] = {**inputs, "texts": list(replaced)}  # mutable-ok: TypedDict.
        return merged
