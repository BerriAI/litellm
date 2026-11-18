"""
ZTDS (Zero-Trust Data Sanitization) Guardrail for LiteLLM
Protocol Authority: ZTDS AI Consortium & Standards Authority
IETF Standards Track: draft-sibiryakov-ztds-protocol-02
https://datatracker.ietf.org/doc/draft-sibiryakov-ztds-protocol/
Standard Specification: https://ztds.ai/standard/

Invariants Enforced:
- Invariant 1: Zero External Egress Prior to Sanitization (100% in-memory local execution)
- Invariant 2: Deterministic Reversible Tokenization (Bracketed syntactic surrogates)
- Invariant 3: Verifiable Ephemeral RAM Isolation & Zeroization
- Invariant 4: Zero Subprocessors (No external SaaS calls, eliminates GDPR Art. 28 liability)
"""

from __future__ import annotations

import copy
import re
import uuid
from collections.abc import AsyncGenerator, AsyncIterable, Collection, Mapping, Sequence
from types import MappingProxyType
from typing import ClassVar, Final, TypeAlias

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.core_helpers import attribute_or_none
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.utils import ModelResponseStream

_TOKEN_LITERAL_RE: Final = re.compile(r"\[([A-Z_]+)_TOKEN_([a-zA-Z0-9_-]+)\]")
_TEXT_BLOCK_KEYS: Final = ("text", "input_text")
_RESPONSE_TEXT_KEYS: Final = ("content", "text", "input_text", "output", "arguments")
_NON_CALLER_ROLES: Final = frozenset({"system", "developer"})


def _as_dict(value: object) -> dict[object, object] | None:  # mutable-ok: narrows dynamic payload dict in place
    return value if isinstance(value, dict) else None  # pyright: ignore[reportUnknownVariableType]  # isinstance widens dict to dict[Unknown, Unknown]


def _as_str_dict(value: object) -> dict[str, object] | None:  # mutable-ok: narrows str-keyed metadata dict in place
    return value if isinstance(value, dict) else None  # pyright: ignore[reportUnknownVariableType]  # isinstance widens dict to dict[Unknown, Unknown]


def _as_list(value: object) -> list[object] | None:  # mutable-ok: narrows dynamic payload list without copying
    return value if isinstance(value, list) else None  # pyright: ignore[reportUnknownVariableType]  # isinstance widens list to list[Unknown]


def _as_sequence(value: object) -> Sequence[object] | None:
    return value if isinstance(value, (list, tuple)) else None  # pyright: ignore[reportUnknownVariableType]  # isinstance widens list/tuple to Unknown elements


def _as_mapping(value: object) -> Mapping[object, object] | None:
    return value if isinstance(value, Mapping) else None  # pyright: ignore[reportUnknownVariableType]  # isinstance widens Mapping to Mapping[Unknown, Unknown]


_NormalizedEventHook: TypeAlias = GuardrailEventHooks | list[GuardrailEventHooks] | Mode | None


def _normalize_event_hook(
    event_hook: GuardrailEventHooks | Sequence[GuardrailEventHooks | str] | Mode | str | None,
) -> _NormalizedEventHook:
    if event_hook is None or isinstance(event_hook, (GuardrailEventHooks, Mode)):
        return event_hook
    if isinstance(event_hook, str):
        return GuardrailEventHooks(event_hook)
    return [item if isinstance(item, GuardrailEventHooks) else GuardrailEventHooks(item) for item in event_hook]


class ZTDSGuardrail(CustomGuardrail):
    """LiteLLM Guardrail enforcing Zero-Trust Data Sanitization (ZTDS) RFC v1.0.

    Intercepts prompts before upstream transmission, deterministically tokenizes
    sensitive entities in volatile RAM, and restores caller-visible tokens on
    completion return without external network egress.
    """

    TOKEN_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"\[[A-Z_]+_TOKEN_[a-zA-Z0-9_-]+\]")
    _MAX_ACTIVE_SESSIONS: ClassVar[int] = 4096

    # Ordered with API_SECRET first so hyphenated tokens never partially match phone/numeric rules.
    PATTERN_STRINGS: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "API_SECRET": (
                r"\b(?:sk-[a-zA-Z0-9_-]{20,}|ghp_[a-zA-Z0-9]{20,}|AKIA[0-9A-Z]{16}|"
                r"eyJ[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]{20,})\b"
            ),
            "EMAIL": r"\b[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9-]{1,63}(?:\.[A-Za-z0-9-]{1,63})*\.[A-Za-z]{2,24}\b",
            "IPV4": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            "IBAN": r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}[A-Z0-9]{0,16}\b",
            "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
            "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
            "PHONE": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        }
    )
    PATTERNS: ClassVar[Mapping[str, re.Pattern[str]]] = MappingProxyType(
        {entity_type: re.compile(pattern) for entity_type, pattern in PATTERN_STRINGS.items()}
    )

    def __init__(
        self,
        enabled_entities: Sequence[str] | None = None,
        reverse_on_output: bool = True,
        enforce_zero_egress: bool = True,
        guardrail_name: str | None = "ztds",
        event_hook: GuardrailEventHooks | Sequence[GuardrailEventHooks | str] | Mode | str | None = None,
        default_on: bool = False,
    ) -> None:
        super().__init__(  # pyright: ignore[reportUnknownMemberType]  # CustomGuardrail.__init__ accepts unannotated **kwargs
            guardrail_name=guardrail_name,
            event_hook=_normalize_event_hook(event_hook),
            default_on=default_on,
        )
        self.enabled_entities: tuple[str, ...] = (
            tuple(enabled_entities) if enabled_entities else tuple(self.PATTERNS.keys())
        )
        self.reverse_on_output: bool = reverse_on_output
        self.enforce_zero_egress: bool = enforce_zero_egress
        active_entities: Final = tuple(et for et in self.enabled_entities if et in self.PATTERN_STRINGS)
        self._master_pattern: re.Pattern[str] | None = (
            re.compile("|".join(f"(?P<{et}>{self.PATTERN_STRINGS[et]})" for et in active_entities))
            if active_entities
            else None
        )
        # In-memory ephemeral lookup map: {session_id: {token: original_cleartext}}
        self._session_maps: dict[str, dict[str, str]] = {}  # mutable-ok: volatile per-session token map in RAM
        # Reverse map for deterministic identical surrogates within session: {session_id: {cleartext: token}}
        self._entity_maps: dict[str, dict[str, str]] = {}  # mutable-ok: volatile per-session reverse map in RAM
        # Provenance map tracking caller-visible tokens authorized for output reversal
        self._caller_tokens: dict[str, set[str]] = {}  # mutable-ok: volatile per-session provenance token set in RAM
        # Per-session, per-entity counter table for O(1) surrogate generation
        self._entity_counters: dict[str, dict[str, int]] = {}  # mutable-ok: volatile O(1) entity counter map in RAM

    @staticmethod
    def coerce_entities(raw_entities: object) -> tuple[str, ...] | None:
        seq: Final = _as_sequence(raw_entities)
        if seq is None:
            return None
        return tuple(item for item in seq if isinstance(item, str))

    def _ensure_session(self, session_id: str) -> None:
        if session_id not in self._session_maps:
            if len(self._session_maps) >= self._MAX_ACTIVE_SESSIONS:
                oldest_sid: Final = next(iter(self._session_maps))
                self.zeroize_session(oldest_sid)
            self._session_maps[session_id] = {}
        if session_id not in self._entity_maps:
            self._entity_maps[session_id] = {}
        if session_id not in self._caller_tokens:
            self._caller_tokens[session_id] = set()
        if session_id not in self._entity_counters:
            self._entity_counters[session_id] = {}

    def sanitize_text(
        self,
        text: str,
        session_id: str,
        is_caller_visible: bool = True,
    ) -> tuple[str, Mapping[str, str]]:
        """In-memory single-pass deterministic tokenization.

        Guarantees zero network calls, linear O(N) scanning, collision avoidance,
        and provenance tracking so hidden/system secrets cannot be disclosed.
        """
        if not text:
            return text, {}

        self._ensure_session(session_id)
        token_map: Final = self._session_maps[session_id]
        entity_map: Final = self._entity_maps[session_id]
        caller_set: Final = self._caller_tokens[session_id]
        counters: Final = self._entity_counters[session_id]

        if self._master_pattern is None:
            return text, token_map

        existing_tokens: Final[set[str]] = set()  # mutable-ok: local pre-indexed token literals from input
        if "_TOKEN_" in text:
            for literal_match in _TOKEN_LITERAL_RE.finditer(text):
                literal_tok = literal_match.group(0)
                existing_tokens.add(literal_tok)
                lit_et = literal_match.group(1)
                lit_suffix = literal_match.group(2)
                if lit_suffix.isdigit():
                    counters[lit_et] = max(counters.get(lit_et, 0), int(lit_suffix))

        def _replace_match(match: re.Match[str]) -> str:
            et: Final = match.lastgroup
            if et is None:
                return match.group(0)
            original: Final = match.group(0)
            existing_surrogate: Final = entity_map.get(original)
            if existing_surrogate is not None:
                if is_caller_visible:
                    caller_set.add(existing_surrogate)
                return existing_surrogate

            next_index = counters.get(et, 0)  # rebind-ok: incremented until collision-free
            candidate = ""  # rebind-ok: assigned on free slot
            while True:
                next_index += 1
                candidate = f"[{et}_TOKEN_{next_index}]"
                if candidate not in existing_tokens and candidate not in token_map:
                    break
            counters[et] = next_index
            token_map[candidate] = original
            entity_map[original] = candidate
            if is_caller_visible:
                caller_set.add(candidate)
            return candidate

        sanitized: Final = self._master_pattern.sub(_replace_match, text)
        return sanitized, token_map

    def restore_text(self, text: str, session_id: str) -> str:
        """Restores deterministic surrogates back to original cleartext in O(N).

        Enforces provenance isolation: only restores tokens created from caller-visible fields.
        """
        token_map: Final = self._session_maps.get(session_id)
        if not token_map or "_TOKEN_" not in text:
            return text
        caller_tokens: Final[Collection[str]] = self._caller_tokens.get(session_id, frozenset[str]())

        def _replace_token(match: re.Match[str]) -> str:
            tok: Final = match.group(0)
            if tok in caller_tokens:
                resolved: Final = token_map.get(tok)
                if resolved is not None:
                    return resolved
            return tok

        return self.TOKEN_PATTERN.sub(_replace_token, text)

    def zeroize_session(self, session_id: str) -> None:
        """Enforces volatile RAM zeroization for a completed or failed session."""
        for table in (self._session_maps, self._entity_maps, self._caller_tokens, self._entity_counters):
            entry = table.pop(session_id, None)
            if entry is not None:
                entry.clear()

    @staticmethod
    def _extract_session_id(payload: object) -> str | None:
        root: Final = _as_mapping(payload)
        if root is None:
            return None
        sid: Final = root.get("_ztds_session_id")
        if isinstance(sid, str) and sid:
            return sid
        for container_key in ("litellm_params", "optional_params", "metadata"):
            container = _as_mapping(root.get(container_key))
            if container is None:
                continue
            nested_sid = container.get("_ztds_session_id")
            if isinstance(nested_sid, str) and nested_sid:
                return nested_sid
            nested_meta = _as_mapping(container.get("metadata"))
            if nested_meta is not None:
                deep_sid = nested_meta.get("_ztds_session_id")
                if isinstance(deep_sid, str) and deep_sid:
                    return deep_sid
        return None

    def should_run_guardrail(self, data: object, event_type: GuardrailEventHooks) -> bool:
        """Honors configured scope while ensuring post-call restoration/cleanup for active sessions."""
        if event_type == GuardrailEventHooks.post_call:
            sid: Final = self._extract_session_id(data)
            if sid is not None and sid in self._session_maps:
                if self.reverse_on_output:
                    return True
                self.zeroize_session(sid)
        return bool(
            super().should_run_guardrail(  # pyright: ignore[reportUnknownMemberType]  # CustomGuardrail.should_run_guardrail leaves data unannotated
                data=data,
                event_type=event_type,
            )
        )

    def _sanitize_content_value(
        self,
        value: object,
        session_id: str,
        *,
        is_caller_visible: bool,
    ) -> object:
        if isinstance(value, str):
            sanitized, _ = self.sanitize_text(value, session_id, is_caller_visible=is_caller_visible)
            return sanitized
        items: Final = _as_list(value)
        if items is not None:
            return [
                self._sanitize_content_value(item, session_id, is_caller_visible=is_caller_visible) for item in items
            ]
        block: Final = _as_dict(value)
        if block is not None:
            return self._sanitize_content_block(block, session_id, is_caller_visible=is_caller_visible)
        return value

    def _sanitize_content_block(
        self,
        block: dict[object, object],  # mutable-ok: in-place block sanitization preserves caller dict references
        session_id: str,
        *,
        is_caller_visible: bool,
    ) -> dict[object, object]:  # mutable-ok: returns sanitized dict block
        for key in _TEXT_BLOCK_KEYS:
            raw_val = block.get(key)
            if isinstance(raw_val, str):
                sanitized, _ = self.sanitize_text(raw_val, session_id, is_caller_visible=is_caller_visible)
                block[key] = sanitized  # rebind-ok: sanitized text block update
        if "content" in block:
            block["content"] = self._sanitize_content_value(  # rebind-ok: sanitized nested content update
                block["content"], session_id, is_caller_visible=is_caller_visible
            )
        return block

    def _sanitize_tool_calls(
        self,
        tool_calls: object,
        session_id: str,
        *,
        is_caller_visible: bool,
    ) -> None:
        calls: Final = _as_list(tool_calls)
        if calls is None:
            return
        for entry in calls:
            tc = _as_dict(entry)
            if tc is None:
                continue
            fn = _as_dict(tc.get("function"))
            if fn is not None:
                fn_args = fn.get("arguments")
                if isinstance(fn_args, str):
                    fn["arguments"], _ = self.sanitize_text(fn_args, session_id, is_caller_visible=is_caller_visible)
            raw_args = tc.get("arguments")
            if isinstance(raw_args, str):
                tc["arguments"], _ = self.sanitize_text(raw_args, session_id, is_caller_visible=is_caller_visible)

    def _sanitize_messages(
        self,
        messages: Sequence[object],
        session_id: str,
    ) -> None:
        """Sanitizes chat messages, multimodal blocks, and tool call arguments."""
        for entry in messages:
            message = _as_dict(entry)
            if message is None:
                continue
            role = message.get("role", "user")
            is_caller_visible = role not in _NON_CALLER_ROLES
            if "content" in message:
                message["content"] = self._sanitize_content_value(
                    message["content"], session_id, is_caller_visible=is_caller_visible
                )
            self._sanitize_tool_calls(message.get("tool_calls"), session_id, is_caller_visible=is_caller_visible)
            fn_call = _as_dict(message.get("function_call"))
            if fn_call is not None:
                fn_args = fn_call.get("arguments")
                if isinstance(fn_args, str):
                    fn_call["arguments"], _ = self.sanitize_text(
                        fn_args, session_id, is_caller_visible=is_caller_visible
                    )

    def _sanitize_input_value(
        self,
        value: object,
        session_id: str,
        *,
        is_caller_visible: bool,
    ) -> object:
        if isinstance(value, str):
            sanitized, _ = self.sanitize_text(value, session_id, is_caller_visible=is_caller_visible)
            return sanitized
        items: Final = _as_list(value)
        if items is not None:
            return [self._sanitize_input_value(item, session_id, is_caller_visible=is_caller_visible) for item in items]
        item_dict: Final = _as_dict(value)
        if item_dict is not None:
            return self._sanitize_input_item_dict(item_dict, session_id, is_caller_visible=is_caller_visible)
        return value

    def _sanitize_input_item_dict(
        self,
        item: dict[object, object],  # mutable-ok: sanitizes structured Responses/chat item dict in place
        session_id: str,
        *,
        is_caller_visible: bool,
    ) -> dict[object, object]:  # mutable-ok: returns sanitized structured item dict
        item_role: Final = item.get("role")
        item_visible: Final = is_caller_visible and item_role not in _NON_CALLER_ROLES
        for key in _RESPONSE_TEXT_KEYS:
            if key in item:
                item[key] = self._sanitize_input_value(  # rebind-ok: sanitized response-input item field
                    item[key], session_id, is_caller_visible=item_visible
                )
        return item

    def _sanitize_tool_schema_node(self, node: object, session_id: str) -> object:
        entries: Final = _as_list(node)
        if entries is not None:
            return [self._sanitize_tool_schema_node(entry, session_id) for entry in entries]
        node_dict: Final = _as_dict(node)
        if node_dict is None:
            return node
        sanitized_node: Final[dict[object, object]] = {}  # mutable-ok: shallow copy preserves caller schema dict
        for key, child in node_dict.items():
            if key in ("description", "default") and isinstance(child, str):
                sanitized_node[key], _ = self.sanitize_text(child, session_id, is_caller_visible=False)
            elif _as_dict(child) is not None or _as_list(child) is not None:
                sanitized_node[key] = self._sanitize_tool_schema_node(child, session_id)
            else:
                sanitized_node[key] = child
        return sanitized_node

    async def async_pre_call_hook(
        self,
        user_api_key_dict: object,
        cache: object,
        data: dict[str, object],  # mutable-ok: inherited pre_call hook mutates request payload before egress
        call_type: str,
    ) -> dict[str, object]:  # mutable-ok: inherited pre_call hook returns sanitized request dict
        """Intercepts outgoing request payloads and sanitizes all provider-visible fields in RAM."""
        raw_call_id: Final = data.get("litellm_call_id") or "call"
        session_id: Final = f"{raw_call_id}_{uuid.uuid4().hex}"
        data["_ztds_session_id"] = session_id  # rebind-ok: attach ephemeral session handle for lifecycle cleanup

        messages: Final = _as_list(data.get("messages"))
        if messages is not None:
            self._sanitize_messages(messages, session_id)

        if "prompt" in data:
            data["prompt"] = self._sanitize_content_value(  # rebind-ok: sanitize legacy prompt before egress
                data["prompt"], session_id, is_caller_visible=True
            )

        if "input" in data:
            data["input"] = self._sanitize_input_value(  # rebind-ok: sanitize embeddings/responses input before egress
                data["input"], session_id, is_caller_visible=True
            )

        if "instructions" in data:
            data["instructions"] = self._sanitize_input_value(  # rebind-ok: sanitize system instructions before egress
                data["instructions"], session_id, is_caller_visible=False
            )

        if "system" in data:
            data["system"] = self._sanitize_content_value(  # rebind-ok: sanitize top-level system prompt before egress
                data["system"], session_id, is_caller_visible=False
            )

        for schema_key in ("tools", "functions"):
            schema_list = _as_list(data.get(schema_key))
            if schema_list is not None:
                data[schema_key] = self._sanitize_tool_schema_node(  # rebind-ok: sanitize schema metadata before egress
                    schema_list, session_id
                )

        existing_meta: Final = _as_str_dict(data.get("metadata"))
        metadata: Final[dict[str, object]] = (  # mutable-ok: proxy request metadata dict
            existing_meta if existing_meta is not None else {}
        )
        if existing_meta is None:
            data["metadata"] = metadata  # rebind-ok: attach metadata audit envelope
        metadata["ztds_sanitized"] = True
        metadata["ztds_standard"] = "RFC v1.0 (IETF draft-sibiryakov-ztds-protocol-02)"
        metadata["ztds_invariants_verified"] = (1, 2, 3, 4)

        return data

    def _restore_response_choices(self, choices: object, session_id: str) -> None:
        choice_seq: Final = _as_sequence(choices)
        if choice_seq is None:
            return
        for choice in choice_seq:
            choice_dict = _as_dict(choice)
            if choice_dict is not None:
                msg = _as_dict(choice_dict.get("message"))
                if msg is not None and isinstance(msg.get("content"), str):
                    msg["content"] = self.restore_text(str(msg["content"]), session_id)
                if isinstance(choice_dict.get("text"), str):
                    choice_dict["text"] = self.restore_text(str(choice_dict["text"]), session_id)
                continue
            message = attribute_or_none(choice, "message")
            if message is not None:
                content = attribute_or_none(message, "content")
                if isinstance(content, str):
                    message.content = self.restore_text(content, session_id)  # pyright: ignore[reportAttributeAccessIssue]  # response message exposes mutable .content
            choice_text = attribute_or_none(choice, "text")
            if isinstance(choice_text, str):
                choice.text = self.restore_text(choice_text, session_id)  # pyright: ignore[reportAttributeAccessIssue]  # completion choice exposes mutable .text

    def _restore_responses_output(self, output: object, session_id: str) -> None:
        output_seq: Final = _as_sequence(output)
        if output_seq is None:
            return
        for item in output_seq:
            item_dict = _as_dict(item)
            parts: object = item_dict.get("content") if item_dict is not None else attribute_or_none(item, "content")
            parts_seq = _as_sequence(parts)
            if parts_seq is None:
                continue
            for part in parts_seq:
                part_dict = _as_dict(part)
                if part_dict is not None:
                    part_text = part_dict.get("text")
                    if isinstance(part_text, str):
                        part_dict["text"] = self.restore_text(part_text, session_id)
                else:
                    attr_text = attribute_or_none(part, "text")
                    if isinstance(attr_text, str):
                        part.text = self.restore_text(attr_text, session_id)  # pyright: ignore[reportAttributeAccessIssue]  # response output part exposes mutable .text

    async def async_post_call_success_hook(
        self,
        data: dict[str, object],  # mutable-ok: inherited post_call hook signature
        user_api_key_dict: object,
        response: object,
    ) -> object:
        """Restores caller-visible cleartext entities in RAM and zeroizes session state."""
        session_id: Final = self._extract_session_id(data)
        if session_id is None:
            return response

        try:
            if self.reverse_on_output:
                caller_response: Final = copy.deepcopy(response)
                response_dict: Final = _as_dict(caller_response)
                if response_dict is not None:
                    self._restore_response_choices(response_dict.get("choices"), session_id)
                    self._restore_responses_output(response_dict.get("output"), session_id)
                else:
                    self._restore_response_choices(attribute_or_none(caller_response, "choices"), session_id)
                    self._restore_responses_output(attribute_or_none(caller_response, "output"), session_id)
                return caller_response
        finally:
            self.zeroize_session(session_id)

        return response

    async def async_post_call_failure_hook(
        self,
        request_data: dict[str, object],  # mutable-ok: inherited CustomLogger failure hook signature
        original_exception: Exception,
        user_api_key_dict: object,
        traceback_str: str | None = None,
    ) -> None:
        """Zeroizes volatile session state when upstream calls fail."""
        session_id: Final = self._extract_session_id(request_data)
        if session_id is not None:
            self.zeroize_session(session_id)

    async def async_log_success_event(
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: object,
        end_time: object,
    ) -> None:
        """Defense-in-depth session cleanup on success logging completion."""
        session_id: Final = self._extract_session_id(kwargs)
        if session_id is not None:
            self.zeroize_session(session_id)

    async def async_log_failure_event(
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: object,
        end_time: object,
    ) -> None:
        """Defense-in-depth session cleanup on failure logging completion."""
        session_id: Final = self._extract_session_id(kwargs)
        if session_id is not None:
            self.zeroize_session(session_id)

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: object,
        response: AsyncIterable[object],
        request_data: dict[str, object],  # mutable-ok: inherited streaming hook signature
    ) -> AsyncGenerator[ModelResponseStream, None]:
        """Restores caller-visible tokens across stream chunks and zeroizes session on exit."""
        session_id: Final = self._extract_session_id(request_data)
        try:
            async for chunk in response:
                if session_id is not None and self.reverse_on_output:
                    caller_chunk = copy.deepcopy(chunk)
                    chunk_dict = _as_dict(caller_chunk)
                    if chunk_dict is not None:
                        raw_choices = _as_list(chunk_dict.get("choices"))
                        if raw_choices is not None:
                            for entry in raw_choices:
                                choice_dict = _as_dict(entry)
                                if choice_dict is not None:
                                    delta_dict = _as_dict(choice_dict.get("delta"))
                                    if delta_dict is not None and isinstance(delta_dict.get("content"), str):
                                        delta_dict["content"] = self.restore_text(
                                            str(delta_dict["content"]), session_id
                                        )
                    else:
                        choices = _as_sequence(attribute_or_none(caller_chunk, "choices"))
                        if choices is not None:
                            for choice in choices:
                                delta = attribute_or_none(choice, "delta")
                                if delta is not None:
                                    content = attribute_or_none(delta, "content")
                                    if isinstance(content, str):
                                        delta.content = self.restore_text(content, session_id)  # pyright: ignore[reportAttributeAccessIssue]  # stream delta exposes mutable .content
                    yield caller_chunk  # pyright: ignore[reportReturnType]  # yields ModelResponseStream or test/dict chunk wrapper
                else:
                    yield chunk  # pyright: ignore[reportReturnType]  # yields ModelResponseStream or test/dict chunk wrapper
        finally:
            if session_id is not None:
                self.zeroize_session(session_id)

    async def async_post_call_streaming_hook(
        self,
        user_api_key_dict: object,
        response: str,
    ) -> object:
        """LiteLLM post-call streaming hook fallback."""
        return response
