"""
PrivAiTe guardrail for the LiteLLM proxy.

Runs PrivAiTe's engine in-process inside LiteLLM. The pre-call hook anonymizes the
request and stashes the reversible map in the request metadata (consumed and popped
by the post-call hook); the post-call hook restores the real values in the response,
including chat tool-call arguments, the legacy function_call, and Responses API
output_text/function_call output. The request surface scanned matches the PrivAiTe
core: chat `messages`; the Responses API `input` (every item type: role message,
tool-call arguments, tool output including `custom_tool_call_output.output` as a
list of `{type, text}` parts, typed action carriers, bare content parts and bare
strings; opaque/binary items relayed whole) and `prompt.variables`; the
`/v1/completions` `prompt` (string or batch list) and `suffix`; the chat
`prediction.content` and `web_search_options.user_location` auxiliary fields (input
side only, nothing to restore; tokenized integer-array inputs pass through
unscanned). Restore covers `content`, the reasoning trace, the refusal and the audio
transcript besides tool/function arguments. Chat streaming is restored too: text
content, the reasoning trace, the refusal, the audio transcript, streamed tool-call
arguments and the streamed function_call. (Responses API streaming restore is not
yet implemented.) On the failure path the map is dropped from metadata as well, so
it cannot reach a failure spend-log.

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
from collections.abc import Callable
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

# Responses `input` items relayed byte-for-byte: an opaque/binary payload
# (encrypted reasoning, a screenshot, a generated image) or a tool/pointer
# definition with no user text. Parity with the PrivAiTe gateway scrubber.
_RESPONSES_OPAQUE_TYPES = frozenset(
    {
        "reasoning",
        "compaction",
        "compaction_trigger",
        "computer_call_output",
        "image_generation_call",
        "item_reference",
        "mcp_list_tools",
        "tool_search_call",
        "tool_search_output",
        "additional_tools",
    }
)

# Responses items whose user data sits in a known structured field (a typed
# command/action, a patch diff, search queries and results, interpreter code
# and logs). Every listed field is walked leaf by leaf.
_RESPONSES_DATA_FIELDS: dict[str, tuple[str, ...]] = {
    "computer_call": ("action", "actions"),
    "local_shell_call": ("action",),
    "shell_call": ("action",),
    "web_search_call": ("action",),
    "apply_patch_call": ("operation",),
    "file_search_call": ("queries", "results"),
    "code_interpreter_call": ("code", "outputs"),
    "program": ("code",),
    "program_output": ("result",),
}

# Content/output parts whose payload is binary (base64 or a file id), not text:
# scrubbing them would corrupt the payload without removing anything a text
# detector could find, so the whole part is relayed whole.
_BINARY_PART_TYPES = frozenset(
    {
        "input_image",
        "input_file",
        "input_audio",
        "image",
        "output_image",
        "computer_screenshot",
    }
)

# Text-bearing fields on an unknown Responses item shape (e.g. an mcp_call
# carries both `arguments` and `output`): every one is scanned, not just the
# first, so a second field is never left raw.
_GENERIC_TEXT_FIELDS = ("output", "arguments", "input", "text", "reason")


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
            except OSError as exc:
                # spaCy models not present yet: download them once, then retry.
                # The download is synchronous pip machinery pulling hundreds of
                # MB; run it off the event loop so it does not stall every other
                # request in this proxy worker. Only the missing-model OSError
                # (spaCy E050) takes this path; permission/disk/network errors
                # re-raise instead of triggering a pointless download.
                message = str(exc)
                if "E050" not in message and "Can't find model" not in message:
                    raise
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

    def _overwrite_snapshot_field(self, data: dict, field: str, new_value: Any) -> None:
        # A plain `data[field] = ...` rebind leaks for a top-level string field
        # (`input`, `suffix`, a string `prompt`): the proxy snapshots the request
        # body by shallow-copying data before this hook, so the original string
        # stays in proxy_server_request.body[field]. Overwrite the snapshot copy
        # too. (Lists and dicts are mutated in place, so the aliased snapshot
        # already reflects the anonymized values.)
        psr = data.get("proxy_server_request")
        body = psr.get("body") if isinstance(psr, dict) else None
        if isinstance(body, dict) and field in body:
            body[field] = new_value

    @staticmethod
    def _make_setter(container: Any, key: Any) -> Callable[[str], None]:
        """A write-back that drops the scrubbed string into container[key]. The
        container (a list slot or a dict field) is aliased by the proxy's shallow
        body snapshot, so the in-place write lands in the snapshot too."""

        def _set(value: str) -> None:
            container[key] = value

        return _set

    def _toplevel_setter(self, data: dict, field: str) -> Callable[[str], None]:
        """A write-back for a top-level string field that also overwrites the
        detached body snapshot copy (see _overwrite_snapshot_field)."""

        def _set(value: str) -> None:
            data[field] = value
            self._overwrite_snapshot_field(data, field, value)

        return _set

    def _add_leaf(self, batch: list, setters: list, text: Any, setter: Callable[[str], None]) -> None:
        """Register one non-empty string leaf: a pseudo-message carrying it (so it
        flows through the engine's single choke point exactly like chat content,
        block gate included) plus the write-back that restores the scrubbed value."""
        if isinstance(text, str) and text:
            batch.append({"role": "user", "content": text})
            setters.append(setter)

    def _collect_content(self, batch: list, setters: list, content: Any, setter: Callable[[str], None]) -> None:
        """A role-message `content`: a bare string, or a list of parts whose
        `text`/`refusal` fields carry user text (a bare string in the list is
        user text too). Binary parts carry nothing a text detector can find."""
        if isinstance(content, str):
            self._add_leaf(batch, setters, content, setter)
            return
        if not isinstance(content, list):
            return
        for idx, part in enumerate(content):
            if isinstance(part, str):
                self._add_leaf(batch, setters, part, self._make_setter(content, idx))
            elif isinstance(part, dict) and part.get("type") not in _BINARY_PART_TYPES:
                for field in ("text", "refusal"):
                    self._add_leaf(batch, setters, part.get(field), self._make_setter(part, field))

    def _collect_data_value(self, batch: list, setters: list, value: Any, setter: Callable[[str], None]) -> None:
        """Walk a tool payload (a string, a list of parts, or arbitrary nested
        JSON) registering every string leaf, but relay binary parts whole. Object
        keys are never scanned, matching the documented boundary."""
        if isinstance(value, str):
            self._add_leaf(batch, setters, value, setter)
            return
        if isinstance(value, list):
            for idx, part in enumerate(value):
                self._collect_data_value(batch, setters, part, self._make_setter(value, idx))
            return
        if isinstance(value, dict) and value.get("type") not in _BINARY_PART_TYPES:
            for key, sub in value.items():
                self._collect_data_value(batch, setters, sub, self._make_setter(value, key))

    def _collect_responses_item(self, batch: list, setters: list, container: list, idx: int, item: Any) -> None:
        """Register the scannable text on ONE Responses `input` list item, by its
        shape (bare string, role message, typed tool call/output, or an unknown
        shape scanned field by field). Opaque/binary items are relayed whole."""
        if isinstance(item, str):
            self._add_leaf(batch, setters, item, self._make_setter(container, idx))
            return
        if not isinstance(item, dict):
            return
        if "role" in item and "content" in item:
            self._collect_content(batch, setters, item["content"], self._make_setter(item, "content"))
            return
        itype = item.get("type") or ""
        if itype in _RESPONSES_OPAQUE_TYPES:
            return
        if not self._collect_typed_item(batch, setters, item, itype):
            self._collect_generic_item(batch, setters, item)

    def _collect_typed_item(self, batch: list, setters: list, item: dict, itype: str) -> bool:
        """Register the text on an item whose user data sits in a type-specific
        field; returns False to hand an unknown shape to the generic scan. Covers
        function_call/custom_tool_call, the typed action carriers, and every
        `*_output` tool output (whose `output` can be a list of {type, text}
        parts, not just a string)."""
        if itype == "function_call" and isinstance(item.get("arguments"), str):
            self._add_leaf(batch, setters, item["arguments"], self._make_setter(item, "arguments"))
            return True
        if itype == "custom_tool_call" and isinstance(item.get("input"), str):
            self._add_leaf(batch, setters, item["input"], self._make_setter(item, "input"))
            return True
        fields = _RESPONSES_DATA_FIELDS.get(itype)
        if fields is not None:
            for field in fields:
                if item.get(field) is not None:
                    self._collect_data_value(batch, setters, item[field], self._make_setter(item, field))
            return True
        if itype.endswith("_output") and "output" in item:
            self._collect_data_value(batch, setters, item["output"], self._make_setter(item, "output"))
            return True
        return False

    def _collect_generic_item(self, batch: list, setters: list, item: dict) -> None:
        """Fallback scan for an unknown item shape: every known text-bearing field
        plus a role-less `content`, so a second field is never left raw."""
        for field in _GENERIC_TEXT_FIELDS:
            self._add_leaf(batch, setters, item.get(field), self._make_setter(item, field))
        if isinstance(item.get("content"), (str, list)):
            self._collect_content(batch, setters, item["content"], self._make_setter(item, "content"))

    def _collect_responses_input(self, data: dict, batch: list, setters: list) -> None:
        input_value = data.get("input")
        if isinstance(input_value, str) and input_value:
            self._add_leaf(batch, setters, input_value, self._toplevel_setter(data, "input"))
        elif isinstance(input_value, list):
            for idx, item in enumerate(input_value):
                self._collect_responses_item(batch, setters, input_value, idx, item)

    @staticmethod
    def _prompt_variables(data: dict) -> dict | None:
        """Responses prompt-template variables carry user data (the template
        id/version do not). None when there are no scannable variables;
        /v1/completions sends `prompt` as a string, which is not this surface."""
        prompt = data.get("prompt")
        if isinstance(prompt, dict) and isinstance(prompt.get("variables"), dict):
            return prompt["variables"]
        return None

    @staticmethod
    def _has_aux_fields(data: dict) -> bool:
        """True when the request carries one of the auxiliary text fields scanned
        by _collect_aux_fields; keeps the pre-call early return from skipping a
        request whose only user text sits in those fields."""
        prediction = data.get("prediction")
        if isinstance(prediction, dict) and "content" in prediction:
            return True
        web_search = data.get("web_search_options")
        if isinstance(web_search, dict) and "user_location" in web_search:
            return True
        suffix = data.get("suffix")
        if isinstance(suffix, str) and suffix:
            return True
        # /v1/completions user text: a string prompt or the batch list shape.
        # (A dict prompt is the Responses template, handled by _prompt_variables.)
        prompt = data.get("prompt")
        return isinstance(prompt, (str, list)) and bool(prompt)

    def _collect_prompt_variables(self, data: dict, batch: list, setters: list) -> None:
        variables = self._prompt_variables(data)
        if variables is None:
            return
        for key, value in variables.items():
            self._collect_data_value(batch, setters, value, self._make_setter(variables, key))

    def _collect_aux_fields(self, data: dict, batch: list, setters: list) -> None:
        """The request-side text fields outside messages/input that LiteLLM
        forwards verbatim (core parity): chat `prediction.content`,
        `web_search_options.user_location`, and the completions `prompt` (string
        or batch list) and `suffix`. Request inputs only, nothing to restore."""
        prediction = data.get("prediction")
        if isinstance(prediction, dict) and prediction.get("content") is not None:
            self._collect_data_value(batch, setters, prediction["content"], self._make_setter(prediction, "content"))
        web_search = data.get("web_search_options")
        if isinstance(web_search, dict) and web_search.get("user_location") is not None:
            self._collect_data_value(
                batch, setters, web_search["user_location"], self._make_setter(web_search, "user_location")
            )
        suffix = data.get("suffix")
        if isinstance(suffix, str) and suffix:
            self._add_leaf(batch, setters, suffix, self._toplevel_setter(data, "suffix"))
        prompt = data.get("prompt")
        if isinstance(prompt, str) and prompt:
            self._add_leaf(batch, setters, prompt, self._toplevel_setter(data, "prompt"))
        elif isinstance(prompt, list):
            # The batch shape: string leaves are scrubbed, a tokenized
            # (integer-array) prompt passes through unscanned as documented.
            for idx, part in enumerate(prompt):
                self._add_leaf(batch, setters, part, self._make_setter(prompt, idx))

    async def _anonymize_request(self, data: dict, engine: Any) -> Any:
        """Anonymize every scanned request surface in place under ONE shared
        mapping (chat `messages`, the Responses `input` and `prompt.variables`,
        and the auxiliary prompt/suffix/prediction/user_location fields), so no
        source is left untouched when a crafted request carries several. Every
        scanned string flows through the engine's single choke point, so the block
        gate and the fail-closed policy apply unchanged. Returns the mapping, or
        None if there was nothing to anonymize."""
        messages = data.get("messages")
        msg_list = messages if isinstance(messages, list) else []
        # Chat messages are scanned natively (multimodal content, bare-string
        # lists, per-message block gate); everything else is registered as string
        # leaves appended after them, sharing this one process_request pass.
        batch: list = list(msg_list)
        setters: list = []
        self._collect_responses_input(data, batch, setters)
        self._collect_prompt_variables(data, batch, setters)
        self._collect_aux_fields(data, batch, setters)

        if not batch:
            return None

        anonymized, mapping = await engine.process_request(batch)

        n = len(msg_list)
        if msg_list:
            # msg_list is data["messages"] (same object) -> mutate it in place.
            msg_list[:] = anonymized[:n]
        for setter, anon in zip(setters, anonymized[n:]):
            setter(anon.get("content", ""))
        return mapping

    async def async_pre_call_hook(self, user_api_key_dict: Any, cache: Any, data: dict, call_type: str) -> dict:
        # metadata is caller-controlled at the proxy boundary, so never trust an
        # incoming privaite_map: this hook is the only authority that may set it.
        # Clearing it here means the post-call hook can only ever restore from a
        # map produced by this guardrail's pre-call pass on the same request.
        metadata = data.get("metadata")
        if isinstance(metadata, dict):
            metadata.pop("privaite_map", None)

        if (
            not data.get("messages")
            and not data.get("input")
            and self._prompt_variables(data) is None
            and not self._has_aux_fields(data)
        ):
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

        if mapping is not None and self.deanonymize and not mapping.is_empty and not data.get("background"):
            # Carry a plain fake->original dict to the post-call hook on the same
            # request, the same channel LiteLLM's Presidio guardrail uses. The
            # post-call hook pops it again so it does not linger in metadata.
            # Background requests are excluded: their result is fetched by a later
            # poll the post-call hooks never see, so the map would persist in
            # request state with no consumer; those responses keep placeholders.
            data.setdefault("metadata", {})["privaite_map"] = dict(mapping.get_all_fakes())
        return data

    async def _restore_audio_transcript(self, message: Any, engine: Any, mapping: Any) -> None:
        """Restore the audio transcript on a response message (restore parity with
        the core: an audio reply carries its text in audio.transcript, where the
        model echoes placeholders like anywhere else)."""
        audio = getattr(message, "audio", None)
        if audio is None:
            return
        transcript = _obj_get(audio, "transcript")
        if isinstance(transcript, str) and transcript:
            _obj_set(audio, "transcript", await engine.process_response(transcript, mapping))

    async def _restore_message(self, message: Any, engine: Any, mapping: Any) -> None:
        """Restore originals in one response message: content, the reasoning
        trace, the refusal, the audio transcript, tool-call args and the legacy
        function_call args."""
        content = getattr(message, "content", None)
        if isinstance(content, str) and content:
            message.content = await engine.process_response(content, mapping)
        # A refusal can quote the request, so it carries placeholders too.
        for field in ("reasoning_content", "reasoning", "refusal"):
            value = getattr(message, field, None)
            if isinstance(value, str) and value:
                setattr(message, field, await engine.process_response(value, mapping))
        await self._restore_audio_transcript(message, engine, mapping)
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

    def _restore_delta_audio(self, delta: Any, index: int, finished: bool, restore: Any) -> None:
        """Feed streamed audio transcript fragments through their own restore
        buffer and flush the held tail on the finish chunk, creating the audio
        carrier when the finish delta has none so the tail is never dropped."""
        audio = getattr(delta, "audio", None)
        fragment = _obj_get(audio, "transcript") if audio is not None else None
        if not isinstance(fragment, str):
            fragment = ""
        if not fragment and not finished:
            return
        restored = restore(("audio", index), fragment, finished)
        if audio is not None:
            _obj_set(audio, "transcript", restored)
        elif restored:
            _obj_set(delta, "audio", {"transcript": restored})

    def _restore_delta(self, delta: Any, index: int, finished: bool, restore: Any) -> None:
        """Restore one streamed delta in place: text content, the reasoning trace,
        the refusal, the audio transcript, streamed tool-call argument fragments
        (per tool_call index) and the legacy function_call."""
        content = getattr(delta, "content", None) or ""
        restored = restore(("content", index), content, finished)
        # Overwrite whenever there was input text (even if the whole fragment is
        # held back, the raw fragment must not stay visible) or the finish flush
        # produced text; a terminal chunk with content=None and nothing held back
        # keeps its None instead of becoming "".
        if content or restored:
            delta.content = restored
        # A refusal can quote the request, so it carries placeholders too.
        for field in ("reasoning_content", "reasoning", "refusal"):
            value = getattr(delta, field, None)
            if isinstance(value, str) and (value or finished):
                setattr(delta, field, restore((field, index), value, finished))
        self._restore_delta_audio(delta, index, finished, restore)
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
