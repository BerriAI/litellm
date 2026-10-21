"""
PrivAiTe guardrail for the LiteLLM proxy.

Runs PrivAiTe's engine in-process. The pre-call hook anonymizes the request and stashes
the reversible map in the request metadata, and the post-call hooks consume it to restore
the real values in the response, streaming included. The request surface scanned matches
the PrivAiTe core: chat `messages`, every Responses `input` item shape and
`prompt.variables`, the `/v1/completions` `prompt` and `suffix`, and the chat
`prediction.content` and `web_search_options.user_location` fields. Responses API
streaming restore is not implemented yet.

If `block_entities` is configured, a request containing any of those PII types is
rejected with a 400 before anything is forwarded. The error names the types, never the
values.

There is no detection or masking logic here, it all lives in the `privaite` package,
which is imported lazily so this module stays importable for the guardrail registry
without it. See https://github.com/crp4222/PrivAiTe.
"""

from __future__ import annotations

import asyncio
import copy
import itertools
from collections.abc import AsyncGenerator, AsyncIterable, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Final, TypeAlias

from typing_extensions import TypedDict, Unpack

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.guardrails import GuardrailEventHooks, Mode

from .engine import (
    PrivaiteEngine,
    RestoreMapping,
    blocked_error,
    build_engine,
    restore_buffer_factory,
    restore_mapping,
)
from .payload import JsonObject, as_array, as_object, read_field, read_list, write_field
from .request_walk import (
    aux_field_slots,
    has_aux_fields,
    prompt_variable_slots,
    prompt_variables,
    responses_input_slots,
)
from .restore import (
    StreamRestorer,
    json_escape,
    restore_message,
    restore_responses_output,
)

if TYPE_CHECKING:
    from litellm.caching.caching import DualCache
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.utils import ModelResponseStream

RESTORE_MAP_KEY: Final = "privaite_map"

_PRESETS: Final = frozenset({"light", "onnx"})

_FALSE_STRINGS: Final = frozenset({"false", "0", "no", ""})

_REQUIRED_HOOKS: Final = (GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call)

_EngineKey: TypeAlias = tuple[str, tuple[str, ...], bool, tuple[str, ...]]


class _CustomGuardrailOptions(TypedDict, total=False, extra_items=object):
    pass


def _parse_block_entities(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    entries: Final = raw.split(",") if isinstance(raw, str) else tuple(str(entry) for entry in raw or ())
    return tuple(entry.strip() for entry in entries if entry.strip())


def _with_required_hooks(
    hook: GuardrailEventHooks | Sequence[GuardrailEventHooks] | None,
) -> list[GuardrailEventHooks]:  # mutable-ok: CustomGuardrail.event_hook is a list
    """Both hooks are mandatory. A `mode: post_call` config alone would skip the
    anonymization and forward raw PII to the model."""
    configured: Final = ((hook,) if hook else ()) if isinstance(hook, str) else tuple(hook or ())
    return list(dict.fromkeys((*configured, *_REQUIRED_HOOKS)))


def _drop_restore_map(request: object) -> None:
    metadata: Final = as_object(read_field(request, "metadata"))
    if metadata is not None:
        metadata.pop(RESTORE_MAP_KEY, None)


def _pop_restore_map(request: JsonObject) -> tuple[tuple[str, str], ...]:
    """Consumes the reversible map, so the originals do not linger in metadata the proxy may persist to spend logs."""
    metadata: Final = as_object(request.get("metadata"))
    restore_map: Final = as_object(metadata.pop(RESTORE_MAP_KEY, None)) if metadata is not None else None
    pairs: Final = (restore_map or {}).items()
    return tuple((fake, original) for fake, original in pairs if isinstance(original, str))


def _pseudo_message(text: str) -> JsonObject:
    """A string leaf wrapped as a message, so it crosses the engine's single choke point like chat content."""
    return {"role": "user", "content": text}


def _json_escaped(pairs: Iterable[tuple[str, str]]) -> Iterator[tuple[str, str]]:
    """The argument channels stream JSON source text, so their originals are restored JSON-escaped.

    Placeholders need no escaping, so the fakes are the same on both sides.
    """
    for fake, original in pairs:
        yield fake, json_escape(original)


class PrivaiteGuardrail(CustomGuardrail):
    def __init__(
        self,
        preset: str | None = None,
        languages: str | None = None,
        deanonymize: bool | str | None = True,
        block_entities: str | Sequence[str] | None = None,
        supported_event_hooks: Sequence[GuardrailEventHooks] = _REQUIRED_HOOKS,
        **kwargs: Unpack[_CustomGuardrailOptions],  # noqa: LIT008  # kwargs-ok: forwarded verbatim to CustomGuardrail.__init__
    ) -> None:
        self.preset: Final = preset if preset in _PRESETS else "onnx"
        self.languages: Final = languages or "en,fr"
        self.deanonymize: Final = (
            deanonymize if isinstance(deanonymize, bool) else str(deanonymize).strip().lower() not in _FALSE_STRINGS
        )
        self.block_entities: Final = _parse_block_entities(block_entities)
        super().__init__(  # pyright: ignore[reportUnknownMemberType]  # CustomGuardrail.__init__ takes untyped **kwargs
            supported_event_hooks=list(supported_event_hooks), **kwargs
        )
        if not isinstance(self.event_hook, Mode):
            self.event_hook = _with_required_hooks(self.event_hook)
        self._engine: PrivaiteEngine | None = None
        self._engine_key: _EngineKey | None = None
        self._lock: Final = asyncio.Lock()

    def _languages(self) -> tuple[str, ...]:
        languages: Final = tuple(language.strip() for language in self.languages.split(",") if language.strip())
        return languages or ("en",)

    async def _engine_for(self, languages: Sequence[str]) -> PrivaiteEngine:
        key: Final = (self.preset, tuple(languages), self.deanonymize, self.block_entities)
        if self._engine is not None and self._engine_key == key:
            return self._engine
        async with self._lock:
            if self._engine is not None and self._engine_key == key:
                return self._engine  # pragma: no cover
            engine: Final = await build_engine(self.preset, languages, self.deanonymize, self.block_entities)
            self._engine = engine
            self._engine_key = key
            return engine

    async def _anonymize_request(self, data: JsonObject, engine: PrivaiteEngine) -> RestoreMapping | None:
        """Anonymizes every scanned request surface in place under one shared mapping.

        Chat messages go to the engine as they are. Every other string leaf is appended to
        the same batch as a pseudo-message, so one process_request pass covers a crafted
        request that carries several surfaces, under the same block gate and fail-closed policy.
        """
        messages: Final = as_array(data.get("messages")) or []
        slots: Final = tuple(
            itertools.chain(responses_input_slots(data), prompt_variable_slots(data), aux_field_slots(data))
        )
        if not messages and not slots:
            return None
        anonymized, mapping = await engine.process_request([*messages, *(_pseudo_message(text) for text, _ in slots)])
        count: Final = len(messages)
        if len(anonymized) != count + len(slots):
            raise RuntimeError(
                f"privaite returned {len(anonymized)} messages for {count + len(slots)}, "
                "refusing to forward a request it did not scrub in full"
            )
        if messages:
            messages[:] = anonymized[:count]
        for (_, write), leaf_message in zip(slots, anonymized[count:], strict=True):
            content = read_field(leaf_message, "content")
            write(content if isinstance(content, str) else "")
        return mapping

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: JsonObject,
        call_type: str,
    ) -> JsonObject:
        # metadata is caller controlled, so an incoming privaite_map is never trusted:
        # the post-call hooks may only restore from a map this pre-call pass produced.
        _drop_restore_map(data)
        if (
            not data.get("messages")
            and not data.get("input")
            and prompt_variables(data) is None
            and not has_aux_fields(data)
        ):
            return data

        engine: Final = await self._engine_for(self._languages())
        blocked: Final = blocked_error()
        try:
            mapping: Final = await self._anonymize_request(data, engine)
        except blocked as exc:
            from fastapi import HTTPException

            raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc

        # A background request's result is fetched by a later poll the post-call hooks
        # never see, so its map would stay in request state with no consumer.
        if mapping is None or not self.deanonymize or mapping.is_empty or data.get("background"):
            return data
        restore_map: Final = dict(mapping.get_all_fakes())
        metadata: Final = as_object(data.get("metadata"))
        if metadata is None:
            write_field(data, "metadata", {RESTORE_MAP_KEY: restore_map})
        else:
            metadata[RESTORE_MAP_KEY] = restore_map
        return data

    async def async_post_call_success_hook(
        self,
        data: JsonObject,
        user_api_key_dict: UserAPIKeyAuth,
        response: object,
    ) -> object:
        """Restores the original values in a copy of the response.

        LiteLLM keeps the object it hands this hook for its response cache and its
        success logs, and a cache write may serialize it after this hook ran. Restoring it
        in place would put this caller's values in the cache under a key built from the
        anonymized request, where another caller's identical-looking request hits them.
        """
        if not self.deanonymize:
            return response
        pairs: Final = _pop_restore_map(data)
        if not pairs:
            return response
        mapping: Final = restore_mapping(pairs)
        engine: Final = await self._engine_for(self._languages())
        restored: Final = copy.deepcopy(response)
        for choice in read_list(restored, "choices"):
            message = read_field(choice, "message")
            if message is not None:
                await restore_message(message, engine, mapping)
        await restore_responses_output(restored, engine, mapping)
        return restored

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: AsyncIterable[ModelResponseStream],
        request_data: JsonObject,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        """Restores the original values in copies of the chunks.

        The stream wrapper keeps every chunk it yields and rebuilds the whole response from
        them for the cache and the success logs once the stream ends, so the chunks
        themselves keep their placeholders, as in the non-streaming hook.
        """
        pairs: Final = _pop_restore_map(request_data)
        if not self.deanonymize or not pairs:
            async for chunk in response:
                yield chunk
            return

        restorer: Final = StreamRestorer(
            restore_mapping(pairs),
            restore_mapping(_json_escaped(pairs)),
            restore_buffer_factory(),
        )
        async for chunk in response:
            restored = copy.deepcopy(chunk)
            restorer.restore_chunk(restored)
            yield restored
        for drained in restorer.drain():
            yield drained

    async def async_post_call_failure_hook(
        self,
        request_data: object,
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: str | None = None,
    ) -> None:
        # The success and streaming hooks never run when the call fails, so the map is
        # dropped here too, before a failure spend-log can persist it.
        _drop_restore_map(request_data)
