"""
PrivAiTe guardrail for the LiteLLM proxy.

Runs PrivAiTe's engine in-process inside LiteLLM. The pre-call hook anonymizes the
request (chat `messages` and Responses API `input`) and stashes the reversible map
in the request metadata (consumed and popped by the post-call hook); the post-call
hook restores the real values in the response, including chat tool-call arguments,
the legacy function_call, and Responses API output_text/function_call output. Chat
streaming is restored too: text content, streamed tool-call arguments, and the
streamed function_call. (Responses API streaming restore is not yet implemented.)
On the failure path the map is dropped from metadata as well, so it cannot reach a
failure spend-log.

If `block_entities` is configured, a request containing any of those PII types is
rejected with an HTTP 400 in the pre-call hook, before anything is forwarded to the
model. The error names the offending type(s) only, never the underlying value.

It reuses PrivAiTe's engine, so there is no detection or masking logic here. The
'privaite' package is imported lazily so this module stays importable (for the
guardrail registry) even when PrivAiTe is not installed.

See https://github.com/crp4222/PrivAiTe for the package.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.guardrails import GuardrailEventHooks

_LANG_MODELS = {
    "en": "en_core_web_lg",
    "fr": "fr_core_news_md",
    "de": "de_core_news_md",
    "es": "es_core_news_md",
    "it": "it_core_news_md",
    "pt": "pt_core_news_md",
    "nl": "nl_core_news_md",
}


def _obj_get(obj: Any, key: str) -> Any:
    """Read a field from a dict-or-object response item."""
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _obj_set(obj: Any, key: str, value: Any) -> None:
    """Write a field on a dict-or-object response item."""
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)


class PrivaiteGuardrail(CustomGuardrail):
    def __init__(self, **kwargs: Any) -> None:
        self.preset = kwargs.pop("preset", None) or "onnx"
        if self.preset not in ("light", "onnx"):
            self.preset = "onnx"
        self.languages = kwargs.pop("languages", None) or "en,fr"
        deanon = kwargs.pop("deanonymize", True)
        self.deanonymize = (
            deanon if isinstance(deanon, bool) else str(deanon).strip().lower() not in ("false", "0", "no", "")
        )
        # PII TYPES to reject outright (empty = mask everything, the default).
        # Accepts a list or a comma-separated string.
        self.block_entities = self._parse_block_entities(kwargs.pop("block_entities", None))
        if "supported_event_hooks" not in kwargs:
            kwargs["supported_event_hooks"] = [
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.post_call,
            ]
        super().__init__(**kwargs)
        # Both hooks are mandatory: pre_call anonymizes the request and post_call
        # restores the response. A config such as `mode: post_call` would skip the
        # pre_call pass and silently forward raw PII to the model, so ensure both
        # hooks run for plain string/list modes. Tag-based Mode configs are left
        # untouched.
        hook = self.event_hook
        if hook is None or isinstance(hook, str):
            base = [hook] if isinstance(hook, str) and hook else []
            self.event_hook = cast(
                list[GuardrailEventHooks],
                list(dict.fromkeys(base + ["pre_call", "post_call"])),
            )
        elif isinstance(hook, list):
            normalized = [getattr(h, "value", h) for h in hook]
            self.event_hook = cast(
                list[GuardrailEventHooks],
                list(dict.fromkeys(normalized + ["pre_call", "post_call"])),
            )
        self._engine: Any = None
        self._engine_key: Any = None
        self._lock = asyncio.Lock()

    def _languages(self) -> list:
        langs = [lang.strip() for lang in self.languages.split(",") if lang.strip()]
        return langs or ["en"]

    @staticmethod
    def _parse_block_entities(raw: object) -> list:
        if isinstance(raw, str):
            return [e.strip() for e in raw.split(",") if e.strip()]
        if isinstance(raw, (list, tuple)):
            return [str(e).strip() for e in raw if str(e).strip()]
        return []

    async def _engine_for(self, languages: list) -> Any:
        key = (self.preset, tuple(languages), self.deanonymize, tuple(self.block_entities))
        if self._engine is not None and self._engine_key == key:
            return self._engine

        async with self._lock:
            if self._engine is not None and self._engine_key == key:
                return self._engine  # pragma: no cover

            try:
                from privaite.config.schema import (
                    AnonymizationConfig,
                    DeanonymizationConfig,
                    DetectorsConfig,
                    PIIConfig,
                    PresidioDetectorConfig,
                )
                from privaite.pii.engine import PIIEngine
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "The PrivAiTe guardrail requires the 'privaite' package. "
                    "Install it with: pip install 'privaite>=0.2.4'"
                ) from exc

            if self.block_entities and "block_entities" not in PIIConfig.model_fields:
                # PIIConfig uses extra="allow", so an older privaite would silently
                # swallow block_entities and forward the PII anyway. Fail closed
                # rather than give a false sense of a policy gate that is not there.
                raise RuntimeError(
                    "block_entities is set but the installed privaite does not "
                    "support it; upgrade privaite to a version that enforces "
                    "pii.block_entities."
                )

            config = PIIConfig(
                enabled=True,
                preset=self.preset,
                detectors=DetectorsConfig(presidio=PresidioDetectorConfig(enabled=True, languages=languages)),
                anonymization=AnonymizationConfig(method="placeholder"),
                deanonymization=DeanonymizationConfig(enabled=self.deanonymize),
                block_entities=self.block_entities,
            )
            engine = PIIEngine(config)
            try:
                await engine.initialize()
            except OSError:
                # spaCy models not present yet: download them once, then retry.
                # The download is synchronous pip machinery pulling hundreds of
                # MB; run it off the event loop so it does not stall every other
                # request in this proxy worker.
                from spacy.cli import download

                for lang in languages:
                    model = _LANG_MODELS.get(lang)
                    if model:
                        await asyncio.to_thread(download, model)
                engine = PIIEngine(config)
                await engine.initialize()

            self._engine = engine
            self._engine_key = key
            return engine

    # Text-bearing fields on non-message Responses input items: a tool output, a
    # streamed/echoed tool call, and a bare input_text/output_text content part.
    # Scanned individually so a mixed input list is not left raw.
    _ITEM_TEXT_FIELDS = ("output", "arguments", "text")

    def _overwrite_snapshot_input(self, data: dict, new_input: Any) -> None:
        # A plain `data["input"] = ...` rebind leaks for string input: the proxy
        # snapshots the request body by shallow-copying data before this hook,
        # so the original string stays in proxy_server_request.body["input"].
        # Overwrite the snapshot copy too. (List inputs are mutated in place, so
        # the aliased snapshot already reflects the anonymized values.)
        psr = data.get("proxy_server_request")
        body = psr.get("body") if isinstance(psr, dict) else None
        if isinstance(body, dict) and "input" in body:
            body["input"] = new_input

    def _responses_item_repr(self, item: Any) -> tuple:
        """Map ONE Responses `input` list item to (engine_message, field) so it
        can be scanned. field=None means the item IS a message (replace it whole
        with the anonymized copy); a field name means write the scrubbed content
        back into item[field]; "__str__" means the item is a bare string. Returns
        (None, None) for a shape with no scannable text."""
        if isinstance(item, str):
            return ({"role": "user", "content": item}, "__str__") if item else (None, None)
        if isinstance(item, dict):
            if "role" in item:
                # A message item; the engine scans its content natively.
                return item, None
            for field in self._ITEM_TEXT_FIELDS:
                if isinstance(item.get(field), str) and item[field]:
                    return {"role": "user", "content": item[field]}, field
            if isinstance(item.get("content"), (str, list)) and item["content"]:
                return {"role": "user", "content": item["content"]}, "content"
        return None, None

    async def _anonymize_request(self, data: dict, engine: Any) -> Any:
        """Anonymize chat `messages` AND Responses `input` in place using the
        engine (span-precise), sharing ONE mapping. Every Responses input item is
        scanned item by item (message, tool output, tool call, bare string), so a
        mixed input list no longer slips past detection and the block gate.
        Returns the mapping, or None if there was nothing to anonymize."""
        messages = data.get("messages")
        msg_list = messages if isinstance(messages, list) else []
        batch: list = list(msg_list)

        input_value = data.get("input")
        targets: list = []  # (index_or_"str", field) describing each write-back
        if isinstance(input_value, str) and input_value:
            batch.append({"role": "user", "content": input_value})
            targets.append(("str", None))
        elif isinstance(input_value, list):
            for idx, item in enumerate(input_value):
                rep, field = self._responses_item_repr(item)
                if rep is None:
                    continue  # no scannable text on this item shape
                batch.append(rep)
                targets.append((idx, field))

        if not batch:
            return None

        anonymized, mapping = await engine.process_request(batch)

        n = len(msg_list)
        if msg_list:
            # msg_list is data["messages"] (same object) -> mutate it in place.
            msg_list[:] = anonymized[:n]

        for (target, field), anon in zip(targets, anonymized[n:]):
            if target == "str":
                new_text = anon.get("content", input_value)
                data["input"] = new_text
                self._overwrite_snapshot_input(data, new_text)
                continue
            # every non-"str" target came from the isinstance(input_value, list)
            # branch above, so input_value is a list here.
            if not isinstance(input_value, list):
                continue
            if field is None:
                input_value[target] = anon
            elif field == "__str__":
                input_value[target] = anon.get("content", input_value[target])
            else:
                new_item = dict(input_value[target])
                new_item[field] = anon.get("content", new_item[field])
                input_value[target] = new_item
        return mapping

    async def async_pre_call_hook(self, user_api_key_dict: Any, cache: Any, data: dict, call_type: str) -> dict:
        # metadata is caller-controlled at the proxy boundary, so never trust an
        # incoming privaite_map: this hook is the only authority that may set it.
        # Clearing it here means the post-call hook can only ever restore from a
        # map produced by this guardrail's pre-call pass on the same request.
        metadata = data.get("metadata")
        if isinstance(metadata, dict):
            metadata.pop("privaite_map", None)

        if not data.get("messages") and not data.get("input"):
            return data

        engine = await self._engine_for(self._languages())
        from privaite.pii.engine import PIIBlockedError

        try:
            mapping = await self._anonymize_request(data, engine)
        except PIIBlockedError as exc:
            # A blocked PII type was found: reject the request outright with a 400,
            # forwarding nothing. The message names TYPES only, never the values.
            from fastapi import HTTPException

            raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc

        if (
            mapping is not None
            and self.deanonymize
            and not mapping.is_empty
            and not data.get("background")
        ):
            # Carry a plain fake->original dict to the post-call hook on the same
            # request, the same channel LiteLLM's Presidio guardrail uses. The
            # post-call hook pops it again so it does not linger in metadata.
            # Background requests are excluded: their result is fetched by a later
            # poll the post-call hooks never see, so the map would persist in
            # request state with no consumer; those responses keep placeholders.
            data.setdefault("metadata", {})["privaite_map"] = dict(mapping.get_all_fakes())
        return data

    async def _restore_message(self, message: Any, engine: Any, mapping: Any) -> None:
        """Restore originals in one response message: content, the reasoning
        trace, tool-call args and the legacy function_call args."""
        content = getattr(message, "content", None)
        if isinstance(content, str) and content:
            message.content = await engine.process_response(content, mapping)
        for field in ("reasoning_content", "reasoning"):
            value = getattr(message, field, None)
            if isinstance(value, str) and value:
                setattr(message, field, await engine.process_response(value, mapping))
        for tool_call in getattr(message, "tool_calls", None) or []:
            fn = getattr(tool_call, "function", None)
            if fn is None:
                continue
            args = getattr(fn, "arguments", None)
            if args:
                fn.arguments = await engine.process_response(args, mapping)
        function_call = getattr(message, "function_call", None)
        if function_call is not None:
            fc_args = getattr(function_call, "arguments", None)
            if fc_args:
                function_call.arguments = await engine.process_response(fc_args, mapping)

    async def _restore_responses_output(self, response: Any, engine: Any, mapping: Any) -> None:
        """Restore originals in a Responses API result: output_text content
        blocks and any function_call output-item arguments (dict or object)."""
        for item in getattr(response, "output", None) or []:
            for block in _obj_get(item, "content") or []:
                text = _obj_get(block, "text")
                if isinstance(text, str) and text:
                    _obj_set(block, "text", await engine.process_response(text, mapping))
            args = _obj_get(item, "arguments")
            if isinstance(args, str) and args:
                _obj_set(item, "arguments", await engine.process_response(args, mapping))

    async def async_post_call_success_hook(self, data: dict, user_api_key_dict: Any, response: Any) -> Any:
        if not self.deanonymize:
            return response
        # pop (not get): consume the reversible map so the originals do not linger
        # in request metadata, which the proxy may persist to spend logs.
        fakes = (data.get("metadata") or {}).pop("privaite_map", None)
        if not fakes:
            return response

        from privaite.pii.mapping import PIIMapping

        mapping = PIIMapping()
        for fake, original in fakes.items():
            mapping.add(original, fake, "PII")

        engine = await self._engine_for(self._languages())
        for choice in getattr(response, "choices", None) or []:
            message = getattr(choice, "message", None)
            if message is not None:
                await self._restore_message(message, engine, mapping)
        # Responses API results carry text under `output`, not `choices`.
        await self._restore_responses_output(response, engine, mapping)
        return response

    def _restore_delta(self, delta: Any, index: int, finished: bool, restore) -> None:
        """Restore one streamed delta in place: text content, the reasoning
        trace, streamed tool-call argument fragments (per tool_call index) and
        the legacy function_call."""
        content = getattr(delta, "content", None) or ""
        restored = restore(("content", index), content, finished)
        if content or finished:
            delta.content = restored
        for field in ("reasoning_content", "reasoning"):
            value = getattr(delta, field, None)
            if isinstance(value, str) and (value or finished):
                setattr(delta, field, restore((field, index), value, finished))
        for tool_call in getattr(delta, "tool_calls", None) or []:
            fn = getattr(tool_call, "function", None)
            if fn is None:
                continue
            args = getattr(fn, "arguments", None)
            if args:
                tc_index = getattr(tool_call, "index", 0) or 0
                # Pass `finished` so a fragment on the same chunk as finish_reason
                # flushes its held-back tail instead of dropping it.
                fn.arguments = restore(("tool", index, tc_index), args, finished)
        function_call = getattr(delta, "function_call", None)
        if function_call is not None:
            fc_args = getattr(function_call, "arguments", None)
            if fc_args:
                function_call.arguments = restore(("fc", index), fc_args, finished)

    async def async_post_call_streaming_iterator_hook(
        self, user_api_key_dict: Any, response: Any, request_data: dict
    ) -> Any:
        # pop (not get): consume the map so the originals do not linger in metadata.
        fakes = (request_data.get("metadata") or {}).pop("privaite_map", None)
        if not self.deanonymize or not fakes:
            async for chunk in response:
                yield chunk
            return

        from privaite.pii.mapping import PIIMapping
        from privaite.streaming.buffer import StreamingDeAnonymizer

        mapping = PIIMapping()
        for fake, original in fakes.items():
            mapping.add(original, fake, "PII")

        # One de-anonymizer buffer per streamed segment, keyed by (kind, choice
        # index, ...). With n>1 the provider interleaves chunks for different
        # choices, and tool-call arguments stream as fragments per tool_call
        # index; each segment keeps its own boundary buffer so a placeholder split
        # across chunks reassembles without mixing segments.
        buffers: dict = {}

        def _restore(key: tuple, text: str, finished: bool) -> str:
            deanon = buffers.get(key)
            if deanon is None:
                deanon = buffers[key] = StreamingDeAnonymizer(mapping)
            out = deanon.feed(text) if text else ""
            if finished:
                out += deanon.flush()
            return out

        # The buffer holds back partial placeholders that span chunk boundaries.
        # Whatever remains is flushed onto the chunk that carries finish_reason.
        async for chunk in response:
            for choice in getattr(chunk, "choices", None) or []:
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                index = getattr(choice, "index", 0) or 0
                finished = getattr(choice, "finish_reason", None) is not None
                self._restore_delta(delta, index, finished, _restore)
            yield chunk

    async def async_post_call_failure_hook(
        self,
        request_data: Any,
        original_exception: Any,
        user_api_key_dict: Any,
        traceback_str: Any = None,
    ) -> Any:
        # The success/streaming hooks pop the reversible map after restoring, but
        # they never run when the call fails. Drop it here too so the originals
        # are not left in metadata for a failure spend-log to persist. (LiteLLM
        # core also strips "privaite_map" from the spend-log body; this is the
        # defense for a stock install that predates that change.)
        metadata = request_data.get("metadata") if isinstance(request_data, dict) else None
        if isinstance(metadata, dict):
            metadata.pop("privaite_map", None)
        return None
