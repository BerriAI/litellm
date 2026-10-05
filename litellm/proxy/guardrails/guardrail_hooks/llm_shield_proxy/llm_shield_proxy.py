import copy
import functools
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
from litellm.types.utils import GenericGuardrailAPIInputs, TextChoices

if TYPE_CHECKING:
    from litellm.caching.caching import DualCache
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.llm_shield_proxy import (
        LLMShieldProxyGuardrailConfigModel,
    )
    from litellm.types.utils import CallTypes, LLMResponseTypes
from .payload import (
    JsonBody,
    MutableRequest,
    RequestTooDeep,
    Slot,
    SlotSink,
    as_array,
    as_object,
    choice_index,
    collect_json_leaves,
    collect_response_item,
    detached,
    read_field,
    read_list,
    rehydrate_slots,
    write_field,
)
from .request_walk import (
    locate_request_texts,
)
from .stream_restorers import (
    AnthropicSSERestorer,
    CarryKey,
    CarryWindows,
    ResponsesStreamRestorer,
    carry_sort_key,
    continuation_delta,
    responses_event_type,
)

GUARDRAIL_NAME: Final = "llm_shield_proxy"

_DEFAULT_API_BASE: Final = "http://localhost:8000"
_REDACT_PATH: Final = "/v1/guard/redact"
_REHYDRATE_PATH: Final = "/v1/guard/rehydrate"
_REHYDRATE_STREAM_PATH: Final = "/v1/guard/rehydrate/stream"

_SESSION_METADATA_KEY: Final = "llm_shield_session_id"

_DEPLOYMENT_RESTORE_KEY: Final = "llm_shield_restore_at_deployment"

_VAULT_PREFIX: Final = f"litellm-{uuid.uuid4().hex}"

_DEFAULT_TIMEOUT_SECONDS: Final = 10.0


class LLMShieldProxyGuardrail(CustomGuardrail):
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

    use_native_lifecycle_hooks: ClassVar[bool] = True

    def __init__(
        self,
        guardrail_name: str = GUARDRAIL_NAME,
        api_base: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,  # noqa: LIT008  # kwargs-ok: forwarded verbatim to CustomGuardrail.__init__
    ) -> None:
        self.async_handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.GuardrailCallback)
        env_base: Final = os.environ.get("LLM_SHIELD_PROXY_API_BASE")
        self.api_base: Final = (api_base or env_base or _DEFAULT_API_BASE).rstrip("/")
        self.api_key: Final = api_key or os.environ.get("LLM_SHIELD_PROXY_API_KEY")
        super().__init__(guardrail_name=guardrail_name, **kwargs)

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:  # mutable-ok: parent's signature.
        return [GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call]

    @staticmethod
    def get_config_model() -> type["LLMShieldProxyGuardrailConfigModel"]:
        from litellm.types.proxy.guardrails.guardrail_hooks.llm_shield_proxy import (
            LLMShieldProxyGuardrailConfigModel,
        )

        return LLMShieldProxyGuardrailConfigModel

    async def async_pre_call_deployment_hook(
        self,
        kwargs: MutableRequest,
        call_type: "CallTypes | None",
    ) -> MutableRequest | None:
        """Redacts a model-level guardrail's request, and keeps it out of the response cache.

        Outside the proxy this hook is the only redaction step, and the deployment post-call
        hook the only restoration step. LiteLLM builds the cache key after this hook, from
        the redacted request, and a cache hit returns before the post-call hook runs. So a
        cached reply would either reach the caller unrestored or, stored after restoration,
        hand this caller's values to the next caller whose redacted request matches. The
        request is therefore neither read from nor written to the cache. Inside the proxy
        this hook does not redact -- the proxy's pre-call hook already ran -- and caching
        is left alone, because the proxy restores after the cache write.

        A streamed request is refused once redacted. No hook restores an SDK stream, and the
        stream's cache writer reads the request from before this hook, so it would also be
        cached despite the bypass.
        """
        before: Final = self._minted_session_id(kwargs)
        _ = await super().async_pre_call_deployment_hook(kwargs, call_type)
        session_id: Final = self._minted_session_id(kwargs)
        if session_id is None or session_id == before:
            return kwargs
        if kwargs.get("stream") is True:
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=(
                    "LLM Shield Proxy cannot restore a streamed reply for a model-level guardrail "
                    "outside the LiteLLM proxy; send the request through the proxy or without stream=True."
                ),
            )
        metadata: Final = as_object(kwargs.get("litellm_metadata"))
        if metadata is not None:
            metadata[_DEPLOYMENT_RESTORE_KEY] = session_id
        cache_controls: Final = as_object(kwargs.get("cache"))
        kwargs["cache"] = {**(cache_controls or {}), "no-cache": True, "no-store": True}
        return kwargs

    async def async_post_call_success_deployment_hook(
        self,
        request_data: MutableRequest,
        response: "LLMResponseTypes",
        call_type: "CallTypes | None",
    ) -> "LLMResponseTypes | None":
        """Restores the reply here only when the deployment pre-call hook redacted it.

        LiteLLM caches what this hook returns. Inside the proxy the request was redacted by
        the proxy's pre-call hook and the proxy's post-call hook restores the reply after
        the cache write, so restoring here as well would cache this caller's plaintext under
        a key built from the redacted request. Outside the proxy nothing restores later, and
        the pre-call deployment hook has already kept that request out of the cache.
        """
        metadata: Final = as_object(request_data.get("litellm_metadata"))
        marker: Final = metadata.get(_DEPLOYMENT_RESTORE_KEY) if metadata is not None else None
        session_id: Final = self._minted_session_id(request_data)
        if session_id is None or marker != session_id:
            return None
        return await super().async_post_call_success_deployment_hook(request_data, response, call_type)

    def _headers(self, session_id: str) -> JsonBody:
        headers: Final[JsonBody] = {
            "Content-Type": "application/json",
            "X-Session-ID": session_id,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _call_shield(self, path: str, session_id: str, payload: JsonBody) -> Mapping[str, object]:
        """Posts to LLM Shield Proxy, failing closed on any transport or status error.

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
            verbose_proxy_logger.exception("LLM Shield Proxy returned %s for %s", exc.response.status_code, path)
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=f"LLM Shield Proxy returned {exc.response.status_code}; blocking the request.",
            ) from exc
        except Exception as exc:
            verbose_proxy_logger.exception("LLM Shield Proxy call to %s failed", path)
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message="LLM Shield Proxy is unreachable; blocking the request.",
            ) from exc

    async def _redact(self, texts: Sequence[str], session_id: str) -> Sequence[str]:
        payload: Final[JsonBody] = {"texts": list(texts)}
        body: Final = await self._call_shield(_REDACT_PATH, session_id, payload)
        return self._same_length_or_raise(body.get("texts"), texts, "redact")

    async def _rehydrate(self, texts: Sequence[str], session_id: str) -> Sequence[str]:
        payload: Final[JsonBody] = {"texts": list(texts)}
        body: Final = await self._call_shield(_REHYDRATE_PATH, session_id, payload)
        return self._same_length_or_raise(body.get("texts"), texts, "rehydrate")

    def _same_length_or_raise(self, returned: object, sent: Sequence[str], operation: str) -> Sequence[str]:
        """Guards the positional mapping the callers rely on to write results back."""
        entries: Final = as_array(returned)
        texts: Final = tuple(entry for entry in entries or () if isinstance(entry, str))
        if entries is None or len(entries) != len(sent) or len(texts) != len(entries):
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=f"LLM Shield Proxy {operation} returned an unexpected payload; blocking the request.",
            )
        return texts

    @staticmethod
    def _mint_session_id(data: MutableRequest) -> str:
        """Mints a vault id for this request, overwriting anything already there.

        Redaction and restoration both happen inside one request/response pair, so
        a fresh id per request is all that is needed, and it is what keeps one
        caller from reaching another caller's vault.
        """
        session_id: Final = f"{_VAULT_PREFIX}-{uuid.uuid4().hex}"
        metadata: Final = data.setdefault("litellm_metadata", {})
        if isinstance(metadata, dict):
            metadata[_SESSION_METADATA_KEY] = session_id
        return session_id

    @staticmethod
    def _minted_session_id(data: MutableRequest) -> str | None:
        """The vault id this process minted for `data`, or None if it has none.

        Read only from `litellm_metadata`, the proxy-private store `_mint_session_id` writes
        to. A caller can populate `metadata`; they cannot populate this.
        """
        metadata: Final = as_object(data.get("litellm_metadata"))
        existing: Final = metadata.get(_SESSION_METADATA_KEY) if metadata is not None else None
        return existing if isinstance(existing, str) and existing.startswith(_VAULT_PREFIX) else None

    @staticmethod
    def _session_id(data: MutableRequest) -> str:
        """Reads back the vault id minted while redacting this request.

        Falls back to an unused id rather than to anything the caller supplied: a
        reply that cannot be restored is a visible placeholder, while trusting a
        caller-supplied id would hand them someone else's plaintext.
        """
        existing: Final = LLMShieldProxyGuardrail._minted_session_id(data)
        return existing if existing is not None else f"{_VAULT_PREFIX}-{uuid.uuid4().hex}"

    @staticmethod
    def _locate_request_texts(data: MutableRequest) -> tuple[Sequence[Slot], Sequence[Slot]]:
        return locate_request_texts(data)

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

        try:
            slots, privileged = self._locate_request_texts(data)
        except RequestTooDeep as exc:
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=f"Request {exc} nests deeper than LLM Shield Proxy inspects; blocking the request.",
            ) from exc
        if not slots and not privileged:
            return data

        session_id: Final = self._mint_session_id(data)
        if privileged:
            await self._redact_into(privileged, f"{_VAULT_PREFIX}-{uuid.uuid4().hex}")
        if slots:
            await self._redact_into(slots, session_id)
        return data

    async def _redact_into(self, slots: Sequence[Slot], session_id: str) -> None:
        """Redacts every span in `slots` under one vault and writes the result back."""
        redacted: Final = await self._redact(tuple(text for text, _ in slots), session_id)
        for (_, write), replacement in zip(slots, redacted):
            write(replacement)

    async def async_post_call_success_hook(
        self,
        data: MutableRequest,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
    ) -> Any:
        """Restores the original values in a copy of a non-streaming response.

        The copy is what keeps plaintext out of the response cache. LiteLLM caches the
        reply it received from the provider, and on some paths -- a native Anthropic dict,
        an in-memory cache -- it stores the object itself rather than a serialised
        snapshot. Restoring that object in place would cache this caller's values under a
        key built from the redacted request, which another caller's identical-looking
        request then hits. Left untouched, the cached reply holds placeholders, and a hit
        is restored against the new caller's own vault.
        """
        if self.should_run_guardrail(data=data, event_type=GuardrailEventHooks.post_call) is not True:
            return response
        response = detached(response)  # rebind-ok: everything below restores the copy.

        if self._is_anthropic_message_response(response):
            return await self._restore_anthropic_response(response, data)

        response_slots: Final = self._responses_api_slots(response)
        if response_slots:
            return await self._restore_responses_api_response(response, response_slots, data)

        choices: Final = getattr(response, "choices", None)
        if not choices:
            return response

        pending: Final[SlotSink] = []
        for choice in choices:
            message = getattr(choice, "message", None)
            if message is None:
                text = read_field(choice, "text")
                if isinstance(text, str) and text:
                    pending.append((text, functools.partial(write_field, choice, "text")))
                continue
            content = getattr(message, "content", None)
            if isinstance(content, str) and content:
                pending.append((content, functools.partial(setattr, message, "content")))
            for tool_call in getattr(message, "tool_calls", None) or ():
                function = getattr(tool_call, "function", None)
                arguments = getattr(function, "arguments", None) if function is not None else None
                if isinstance(arguments, str) and arguments:
                    pending.append((arguments, functools.partial(setattr, function, "arguments")))
            legacy = getattr(message, "function_call", None)
            legacy_arguments = getattr(legacy, "arguments", None) if legacy is not None else None
            if isinstance(legacy_arguments, str) and legacy_arguments:
                pending.append((legacy_arguments, functools.partial(setattr, legacy, "arguments")))
        if not pending:
            return response

        restored: Final = await self._rehydrate(tuple(text for text, _ in pending), self._session_id(data))
        for (_, write), replacement in zip(pending, restored):
            write(replacement)
        return response

    @staticmethod
    def _is_anthropic_message_response(response: object) -> bool:
        """Anthropic's native /v1/messages reply arrives as a plain dict."""
        body: Final = as_object(response)
        return body is not None and body.get("type") == "message" and isinstance(body.get("content"), list)

    async def _restore_anthropic_response(self, response: MutableRequest, data: MutableRequest) -> MutableRequest:
        """Restores text blocks and tool inputs in an Anthropic native message reply.

        This shape has no `choices`, so without its own branch the reply would go
        back to the caller still carrying placeholders.

        A `tool_use` block's payload is `input`, an arbitrary JSON object rather than a
        string, and the request path redacts its string leaves -- so the reply's leaves
        have to come back or the application invokes the tool with placeholders.
        """
        slots: Final[SlotSink] = []
        for entry in read_list(response, "content"):
            block = as_object(entry)
            if block is None:
                continue
            kind = block.get("type")
            text = block.get("text")
            if kind == "text" and isinstance(text, str) and text:
                slots.append((text, functools.partial(block.__setitem__, "text")))
            elif kind == "tool_use" and as_object(block.get("input")) is not None:
                collect_json_leaves(block.get("input"), slots)
        if not slots:
            return response

        restored: Final = await self._rehydrate(tuple(text for text, _ in slots), self._session_id(data))
        for (_, write), replacement in zip(slots, restored):
            write(replacement)
        return response

    @staticmethod
    def _responses_api_slots(response: object) -> Sequence[Slot]:
        """Restorable spans in a Responses API reply.

        That shape carries `output` items rather than `choices`, so it needs its own
        walk; without one the reply goes back to the caller still holding
        placeholders even though the request was redacted correctly. Items and blocks
        come through as dicts or as objects depending on how far the reply has been
        deserialised, so both are handled.

        The item-level fields mirror `collect_responses_fields`, which walks the same
        fields on the request side -- a function_call item holds `arguments`, a
        function_call_output holds `output` -- so the two directions stay symmetric.
        """
        slots: Final[SlotSink] = []
        for item in getattr(response, "output", None) or ():
            collect_response_item(item, slots)
        return tuple(slots)

    async def _restore_responses_api_response(self, response: Any, slots: Sequence[Slot], data: MutableRequest) -> Any:
        """Puts the original values back into a Responses API reply."""
        await rehydrate_slots(slots, functools.partial(self._rehydrate, session_id=self._session_id(data)))
        return response

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: MutableRequest,
    ) -> AsyncGenerator[Any, None]:
        """Restores original values incrementally, without buffering the stream.

        Each choice -- and each tool call within a choice -- is its own token stream, so
        the sliding window is tracked per (choice index, tool call) pair. One shared
        window would splice the characters held back for one stream onto another. The
        windows are locals of this generator, so they are scoped to a single stream and
        cannot leak between concurrent requests.

        The two native stream shapes have no `choices` and are restored by their own
        walkers, with the same per-stream windows: Anthropic `/v1/messages` arrives as raw
        SSE frames, and the Responses API as typed events.
        """
        if self.should_run_guardrail(data=request_data, event_type=GuardrailEventHooks.post_call) is not True:
            async for chunk in response:
                yield chunk
            return

        session_id: Final = self._session_id(request_data)
        step: Final = functools.partial(self._stream_step, session_id=session_id)
        rehydrate: Final = functools.partial(self._rehydrate, session_id=session_id)
        sse: Final = AnthropicSSERestorer(step)
        events: Final = ResponsesStreamRestorer(step, rehydrate)
        carries: Final[CarryWindows] = {}
        last_chunk = None  # rebind-ok: tracks the most recent chunk for the final flush.

        async for chunk in response:
            if isinstance(chunk, (bytes, str)):
                for frames in await sse.feed(chunk):
                    yield frames
                continue
            if responses_event_type(chunk) is not None:
                for event in await events.restore(detached(chunk)):
                    yield event
                continue
            restored_chunk = detached(chunk)
            last_chunk = restored_chunk
            for choice in getattr(restored_chunk, "choices", None) or ():
                await self._restore_choice(choice, carries, session_id)
            yield restored_chunk

        for frames in await sse.finish():
            yield frames
        for event in await events.finish():
            yield event
        if last_chunk is not None and any(carries.values()):
            async for trailing in self._flush_trailing(last_chunk, carries, session_id):
                yield trailing

    async def _restore_choice(self, choice: object, carries: CarryWindows, session_id: str) -> None:
        """Restores one choice's delta, advancing that choice's own windows.

        Content and each tool call are separate token streams, so each gets its own
        window: `(choice_index, None)` for content, `(choice_index, tool_call_index)` for
        one tool call's accumulating `arguments`. A shared window would splice the text
        held back for one stream onto another.
        """
        delta: Final = getattr(choice, "delta", None)
        index: Final = choice_index(choice)
        is_final: Final = bool(getattr(choice, "finish_reason", None))
        if isinstance(choice, TextChoices):
            await self._restore_text_window(choice, (index, None), carries, session_id, is_final)
            return
        if delta is None:
            return

        await self._restore_content_window(delta, (index, None), carries, session_id, is_final)

        for tool_call in getattr(delta, "tool_calls", None) or ():
            await self._restore_tool_call_window(tool_call, index, carries, session_id)

        if is_final:
            await self._flush_finished_choice(delta, index, carries, session_id)

    async def _restore_text_window(
        self,
        choice: object,
        key: CarryKey,
        carries: CarryWindows,
        session_id: str,
        is_final: bool,
    ) -> None:
        """Restores a Completions stream choice's `text` through its window."""
        carry: Final = carries.get(key, "")
        text: Final = read_field(choice, "text")
        if not isinstance(text, str) or not text:
            if not (is_final and carry):
                return
        emitted, remaining = await self._stream_step(text if isinstance(text, str) else "", carry, is_final, session_id)
        carries[key] = remaining  # rebind-ok: this stream's window advances.
        if emitted or text:
            write_field(choice, "text", emitted)

    async def _restore_content_window(
        self,
        delta: Any,
        key: CarryKey,
        carries: CarryWindows,
        session_id: str,
        is_final: bool,
    ) -> None:
        """Restores one delta's content through its own window."""
        carry: Final = carries.get(key, "")
        text: Final = getattr(delta, "content", None)

        if not isinstance(text, str) or not text:
            if is_final and carry:
                flushed, remaining = await self._stream_step("", carry, True, session_id)
                carries[key] = remaining  # rebind-ok: this stream's window advances.
                if flushed:
                    delta.content = flushed
            return

        emitted, remaining = await self._stream_step(text, carry, is_final, session_id)
        carries[key] = remaining  # rebind-ok: this stream's window advances.
        delta.content = emitted

    async def _restore_tool_call_window(
        self,
        tool_call: Any,
        choice_index: int,
        carries: CarryWindows,
        session_id: str,
    ) -> None:
        """Restores one streamed tool call's argument fragment.

        A tool call's `arguments` is a JSON document delivered as fragments that clients
        concatenate per tool-call index, so each index gets a window of its own rather
        than sharing the content stream's.
        """
        tool_index: Final = read_field(tool_call, "index")
        if not isinstance(tool_index, int):
            return
        function: Final = read_field(tool_call, "function")
        if function is None:
            return
        arguments: Final = read_field(function, "arguments")
        if not isinstance(arguments, str) or not arguments:
            return

        key: Final = (choice_index, tool_index)
        emitted, remaining = await self._stream_step(arguments, carries.get(key, ""), False, session_id)
        carries[key] = remaining  # rebind-ok: this tool call's window advances.
        write_field(function, "arguments", emitted)

    async def _flush_finished_choice(
        self,
        delta: Any,
        choice_index: int,
        carries: CarryWindows,
        session_id: str,
    ) -> None:
        """Emits everything this finishing choice still holds, into this chunk.

        A client parses a tool call's `arguments` when the chunk carrying the
        finish_reason arrives, so a flush delivered afterwards is too late -- the client
        has already tried to parse truncated JSON. Content lands back on `content`; held
        tool-call text is appended as an index-only continuation entry, which is the shape
        clients concatenate by index, so no id or name is needed. Appending is correct
        even when this chunk already carried a fragment for that tool call.
        """
        continuations: Final[list[dict[str, object]]] = []  # mutable-ok: built into this chunk's delta.
        for key in sorted((held for held in carries if held[0] == choice_index), key=carry_sort_key):
            carry = carries[key]
            if not carry:
                continue
            _, tool_index = key
            text, remaining = await self._stream_step("", carry, True, session_id)
            carries[key] = remaining  # rebind-ok: this stream's window advances.
            if not text:
                continue
            if tool_index is None:
                delta.content = text
            else:
                continuations.extend(continuation_delta(tool_index, text))
        if continuations:
            existing: Final = tuple(getattr(delta, "tool_calls", None) or ())
            delta.tool_calls = [*existing, *continuations]

    async def _flush_trailing(
        self, last_chunk: Any, carries: CarryWindows, session_id: str
    ) -> AsyncGenerator[Any, None]:
        """Empties every window still holding text, one chunk per window.

        This is the net for a stream that ended with no finish_reason at all; a stream
        that ended with one is flushed into its own terminal chunk by
        `_flush_finished_choice`, because that is the moment a client parses tool
        arguments.

        Driven by the windows rather than by the last chunk's choices. A choice that
        finished earlier is not present in the terminal chunk, and flushing only what
        that chunk carries would drop its held text and truncate its answer.
        """
        for key in sorted(carries, key=carry_sort_key):
            carry = carries[key]
            if not carry:
                continue
            choice_index, tool_index = key
            text, remaining = await self._stream_step("", carry, True, session_id)
            carries[key] = remaining  # rebind-ok: this stream's window advances.
            if not text:
                continue
            chunk = self._chunk_for_choice(last_chunk, choice_index)
            if chunk is None:
                continue
            choice: object = chunk.choices[0]
            delta = read_field(choice, "delta")
            if isinstance(choice, TextChoices):
                choice.text = text
            elif tool_index is None:
                write_field(delta, "content", text)
            else:
                write_field(delta, "content", None)
                write_field(delta, "tool_calls", continuation_delta(tool_index, text))
            yield chunk

    @staticmethod
    def _chunk_for_choice(last_chunk: Any, index: int) -> Any:
        """A single-choice copy of the last chunk, carrying only `index`.

        Emitting one choice per chunk keeps a flush from reading as content on a
        choice it does not belong to.
        """
        chunk: Final = last_chunk.model_copy(deep=True)
        raw_choices: Final = getattr(chunk, "choices", None)
        if not raw_choices:
            return None
        choices: Final[tuple[object, ...]] = tuple(raw_choices)
        position: Final = next((at for at, choice in enumerate(choices) if choice_index(choice) == index), 0)
        kept: Final = raw_choices[position]
        if getattr(kept, "delta", None) is None and not isinstance(kept, TextChoices):
            return None
        kept.index = index
        kept.finish_reason = None
        chunk.choices = [kept]
        if hasattr(chunk, "usage"):
            del chunk.usage
        return chunk

    async def _stream_step(self, text: str, carry: str, final: bool, session_id: str) -> tuple[str, str]:
        """Returns ``(text safe to emit now, window still being held)``."""
        body: Final = await self._call_shield(
            _REHYDRATE_STREAM_PATH,
            session_id,
            {"text": text, "carry": carry, "final": final},
        )
        emitted: Final = body.get("text")
        remaining: Final = body.get("carry")
        if not isinstance(emitted, str) or not isinstance(remaining, str):
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message="LLM Shield Proxy stream rehydration returned an unexpected payload.",
            )
        return emitted, remaining

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: MutableRequest,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        """Unified entry point: what the UI's Test guardrail button and the translation
        handlers call.

        `tool_calls` is handled on the response side only. LiteLLM populates the field here,
        and on a reply it holds the model's tool arguments -- the same text the native hook
        restores, and restoring one but not the other would leave the placeholder on
        whichever path ran. The request side is left to the native pre-call hook, because
        redacting it here as well would redact it twice.
        """
        text_list: Final = tuple(inputs.get("texts") or ())
        tool_calls: Final = tuple(inputs.get("tool_calls") or ()) if input_type == "response" else ()
        if not text_list and not tool_calls:
            return inputs

        restored_calls: Final[list[object]] = [copy.deepcopy(call) for call in tool_calls]  # mutable-ok: a new list.
        spans: Final[list[str]] = list(text_list)  # mutable-ok: ordered batch, frozen before the call.
        writers: Final[list[Callable[[str], None]]] = []  # mutable-ok: one per span appended below.
        for call in restored_calls:
            function = read_field(call, "function")
            arguments = read_field(function, "arguments") if function is not None else None
            if isinstance(arguments, str) and arguments:
                spans.append(arguments)
                writers.append(functools.partial(write_field, function, "arguments"))

        replaced: Final = (
            await self._redact(tuple(spans), self._mint_session_id(request_data))
            if input_type == "request"
            else await self._rehydrate(tuple(spans), self._session_id(request_data))
        )
        restored_values: Final[list[str]] = list(replaced)  # mutable-ok: sliced into the texts list.

        for write, replacement in zip(writers, restored_values[len(text_list) :]):
            write(replacement)
        merged: Final[JsonBody] = {**inputs}
        if text_list:
            merged["texts"] = restored_values[: len(text_list)]
        if restored_calls:
            merged["tool_calls"] = restored_calls
        return merged
