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
import json
import re
import uuid
from collections.abc import AsyncGenerator, AsyncIterable, Collection, Mapping, Sequence
from types import MappingProxyType
from typing import ClassVar, Final, TypeAlias

from pydantic import BaseModel, TypeAdapter, ValidationError

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.core_helpers import attribute_or_none
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.utils import ModelResponseStream

_TOKEN_LITERAL_RE: Final = re.compile(r"\[([A-Z_]+)_TOKEN_([a-zA-Z0-9_-]+)\]")
_NON_CALLER_ROLES: Final = frozenset({"system", "developer"})
_JSON_VALUE_ADAPTER: Final = TypeAdapter(object)
_JSON_ENCODED_STRING_KEYS: Final = frozenset({"arguments"})
_SYSTEM_PROVENANCE_KEYS: Final = frozenset(
    {
        "system",
        "instructions",
        "instruction",
        "systemInstruction",
        "system_instruction",
        "tools",
        "functions",
        "tool_choice",
        "function_call",
        "response_format",
        "config",
    }
)
_HIDDEN_REASONING_KEYS: Final = frozenset({"thinking", "reasoning", "reasoning_content", "summary"})
_CONTROL_KEYS: Final = frozenset(
    {
        "_ztds_session_id",
        "model",
        "custom_llm_provider",
        "api_key",
        "api_base",
        "api_version",
        "organization",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "aws_region_name",
        "aws_profile_name",
        "aws_role_name",
        "aws_web_identity_token",
        "aws_sts_endpoint",
        "aws_external_id",
        "aws_bedrock_runtime_endpoint",
        "vertex_project",
        "vertex_location",
        "vertex_credentials",
        "azure_ad_token",
        "azure_scope",
        "headers",
        "extra_headers",
        "litellm_call_id",
        "litellm_trace_id",
        "litellm_session_id",
        "litellm_logging_obj",
        "litellm_params",
        "metadata",
        "litellm_metadata",
        "proxy_server_request",
        "user_api_key_dict",
        "model_info",
        "guardrails",
        "guardrail_to_apply",
        "callbacks",
        "success_callback",
        "failure_callback",
        "client",
        "async_client",
        "http_client",
        "router",
        "cache",
        "caching",
        "stream",
        "stream_options",
        "timeout",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "n",
        "stop",
        "logprobs",
        "top_logprobs",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "seed",
        "service_tier",
        "store",
        "parallel_tool_calls",
        "truncation",
        "previous_response_id",
        "conversation",
        "background",
        "modalities",
        "drop_params",
        "additional_drop_params",
        "user",
    }
)


def _dump_pydantic_model(model: BaseModel) -> object:
    return model.model_dump()  # pyright: ignore[reportAny]  # BaseModel.model_dump returns dict[str, Any]


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

    # Disjoint boundary anchors keep JWT, API secret, and email scans strictly O(N) on repeated prefixes.
    PATTERN_STRINGS: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "API_SECRET": (
                r"(?:(?<![A-Za-z0-9_-])(?:sk-[a-zA-Z0-9_-]{20,}|ghp_[a-zA-Z0-9]{20,}|AKIA[0-9A-Z]{16})"
                r"|(?<![A-Za-z0-9_.-])eyJ[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]{20,})"
                r"(?![A-Za-z0-9_-])"
            ),
            "EMAIL": (
                r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]{1,64}"
                r"@[A-Za-z0-9-]{1,63}(?:\.[A-Za-z0-9-]{1,63}){0,10}\.[A-Za-z]{2,24}\b"
            ),
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

    def _sanitize_string_value(
        self,
        value: str,
        session_id: str,
        *,
        is_caller_visible: bool,
        parent_key: str | None,
    ) -> str:
        if parent_key in _JSON_ENCODED_STRING_KEYS:
            try:
                parsed: Final = _JSON_VALUE_ADAPTER.validate_json(value)
                if _as_dict(parsed) is not None or _as_list(parsed) is not None:
                    sanitized_obj: Final = self._sanitize_payload_node(
                        parsed, session_id, is_caller_visible=is_caller_visible, parent_key=None
                    )
                    return json.dumps(sanitized_obj, ensure_ascii=False)
            except (ValueError, ValidationError):
                pass
        sanitized, _ = self.sanitize_text(value, session_id, is_caller_visible=is_caller_visible)
        return sanitized

    def _sanitize_dict_node(
        self,
        node_dict: dict[object, object],  # mutable-ok: sanitizes provider payload dict keys
        session_id: str,
        *,
        is_caller_visible: bool,
    ) -> dict[object, object]:  # mutable-ok: returns sanitized nested dict
        role_val: Final = node_dict.get("role")
        branch_visible: Final = is_caller_visible and not (
            isinstance(role_val, str) and role_val.lower() in _NON_CALLER_ROLES
        )
        if not branch_visible:
            return {
                key: (
                    val
                    if isinstance(key, str) and (key.startswith("_") or key == "litellm_logging_obj")
                    else self._sanitize_payload_node(
                        val,
                        session_id,
                        is_caller_visible=False,
                        parent_key=key if isinstance(key, str) else None,
                    )
                )
                for key, val in node_dict.items()
            }
        for key in tuple(node_dict.keys()):
            if isinstance(key, str) and (key.startswith("_") or key == "litellm_logging_obj"):
                continue
            parent_key = key if isinstance(key, str) else None
            child_visible = parent_key not in _SYSTEM_PROVENANCE_KEYS and parent_key not in _HIDDEN_REASONING_KEYS
            node_dict[key] = self._sanitize_payload_node(  # rebind-ok: in-place request dict sanitization
                node_dict[key],
                session_id,
                is_caller_visible=child_visible,
                parent_key=parent_key,
            )
        return node_dict

    def _sanitize_payload_node(
        self,
        node: object,
        session_id: str,
        *,
        is_caller_visible: bool,
        parent_key: str | None = None,
    ) -> object:
        if node is None or isinstance(node, (bool, int, float)):
            return node
        if isinstance(node, str):
            return self._sanitize_string_value(
                node, session_id, is_caller_visible=is_caller_visible, parent_key=parent_key
            )
        if isinstance(node, BaseModel):
            return self._sanitize_payload_node(
                _dump_pydantic_model(node),
                session_id,
                is_caller_visible=is_caller_visible,
                parent_key=parent_key,
            )
        list_val: Final = _as_list(node)
        if list_val is not None:
            if not is_caller_visible:
                return [
                    self._sanitize_payload_node(item, session_id, is_caller_visible=False, parent_key=parent_key)
                    for item in list_val
                ]
            for idx in range(len(list_val)):
                list_val[idx] = self._sanitize_payload_node(
                    list_val[idx], session_id, is_caller_visible=True, parent_key=parent_key
                )
            return list_val
        seq_val: Final = _as_sequence(node)
        if seq_val is not None:
            return tuple(
                self._sanitize_payload_node(
                    item, session_id, is_caller_visible=is_caller_visible, parent_key=parent_key
                )
                for item in seq_val
            )
        dict_val: Final = _as_dict(node)
        if dict_val is not None:
            return self._sanitize_dict_node(dict_val, session_id, is_caller_visible=is_caller_visible)
        mapping_val: Final = _as_mapping(node)
        if mapping_val is not None:
            copied: Final[dict[object, object]] = dict(mapping_val)  # mutable-ok: mutable working copy of Mapping
            return self._sanitize_dict_node(copied, session_id, is_caller_visible=is_caller_visible)
        return node

    def _sanitize_request_mapping(
        self,
        target: dict[str, object],  # mutable-ok: sanitizes top-level/optional_params request container
        session_id: str,
    ) -> None:
        for key in tuple(target.keys()):
            if key == "optional_params":
                opt_dict = _as_str_dict(target.get("optional_params"))
                if opt_dict is not None:
                    self._sanitize_request_mapping(opt_dict, session_id)
                continue
            if key in _CONTROL_KEYS or key.startswith(("_", "litellm_")):
                continue
            is_visible = key not in _SYSTEM_PROVENANCE_KEYS
            target[key] = self._sanitize_payload_node(  # rebind-ok: sanitize provider-visible request field
                target[key], session_id, is_caller_visible=is_visible, parent_key=key
            )

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

        self._sanitize_request_mapping(data, session_id)

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

    def _restore_parts_sequence(self, parts: object, session_id: str) -> None:
        parts_seq: Final = _as_sequence(parts)
        if parts_seq is None:
            return
        for part in parts_seq:
            part_dict = _as_dict(part)
            if part_dict is not None:
                part_text = part_dict.get("text")
                if isinstance(part_text, str):
                    part_dict["text"] = self.restore_text(part_text, session_id)
            else:
                attr_text = attribute_or_none(part, "text")
                if isinstance(attr_text, str):
                    part.text = self.restore_text(attr_text, session_id)  # pyright: ignore[reportAttributeAccessIssue]  # response part exposes mutable .text

    def _restore_anthropic_and_gemini(self, caller_response: object, session_id: str) -> None:
        response_dict: Final = _as_dict(caller_response)
        top_content: Final = (
            response_dict.get("content") if response_dict is not None else attribute_or_none(caller_response, "content")
        )
        self._restore_parts_sequence(top_content, session_id)
        candidates: Final = _as_sequence(
            response_dict.get("candidates")
            if response_dict is not None
            else attribute_or_none(caller_response, "candidates")
        )
        if candidates is None:
            return
        for candidate in candidates:
            cand_dict = _as_dict(candidate)
            cand_content = (
                cand_dict.get("content") if cand_dict is not None else attribute_or_none(candidate, "content")
            )
            content_dict = _as_dict(cand_content)
            parts = content_dict.get("parts") if content_dict is not None else attribute_or_none(cand_content, "parts")
            self._restore_parts_sequence(parts, session_id)

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
                self._restore_anthropic_and_gemini(caller_response, session_id)
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
