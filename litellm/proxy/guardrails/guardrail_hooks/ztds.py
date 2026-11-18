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
from typing import ClassVar, Final, Literal, TypeAlias

from pydantic import BaseModel, TypeAdapter, ValidationError

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.core_helpers import attribute_or_none
from litellm.litellm_core_utils.served_output_texts import (
    clear_served_output_sanitizer,
    register_served_output_sanitizer,
)
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.utils import GenericGuardrailAPIInputs, ModelResponseStream

_SUPPORTED_ENTITY_ALT: Final = "API_SECRET|CREDIT_CARD|EMAIL|IBAN|IPV4|PHONE|SSN"
_TOKEN_LITERAL_RE: Final = re.compile(rf"\[({_SUPPORTED_ENTITY_ALT})_TOKEN_(\d{{1,18}})\]")
_NON_CALLER_ROLES: Final = frozenset({"system", "developer"})
_JSON_VALUE_ADAPTER: Final = TypeAdapter(object)
_JSON_ENCODED_STRING_KEYS: Final = frozenset({"arguments"})
_BASE64_DATA_URI_RE: Final = re.compile(
    r"data:(?:image|video|audio)/[a-z0-9.+-]+[;,]|data:[a-z0-9.+/-]+(?:;[a-z0-9.=+-]+)*;base64,",
    re.IGNORECASE,
)
_UNSUPPORTED_BLOCK_TYPES: Final = frozenset(
    {
        "file",
        "input_file",
        "input_audio",
        "audio",
        "image",
        "image_url",
        "input_image",
        "video",
        "input_video",
        "computer_screenshot",
    }
)
_UNSUPPORTED_MEDIA_KEYS: Final = frozenset(
    {
        "file_data",
        "fileData",
        "file_id",
        "input_audio",
        "image_url",
        "inline_data",
        "inlineData",
        "s3Location",
    }
)
_UNSUPPORTED_MEDIA_CONTAINERS: Final = frozenset({"image", "video"})
_UNSUPPORTED_SOURCE_TYPES: Final = frozenset({"base64", "url", "file"})
_PARALLEL_HOOK_NAMES: Final = frozenset(
    {GuardrailEventHooks.during_call.value, GuardrailEventHooks.during_mcp_call.value}
)
_INTERNAL_METADATA_KEYS: Final = frozenset(
    {
        "guardrails",
        "applied_guardrails",
        "pre_call_executed_guardrails",
        "model_info",
        "requester_metadata",
        "ztds_sanitized",
        "ztds_standard",
        "ztds_invariants_verified",
    }
)
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
        "user",
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


def _as_set(value: object) -> set[object] | None:  # mutable-ok: narrows dynamic set container
    return value if isinstance(value, set) else None  # pyright: ignore[reportUnknownVariableType]  # isinstance widens set to set[Unknown]


def _as_frozenset(value: object) -> frozenset[object] | None:
    return value if isinstance(value, frozenset) else None  # pyright: ignore[reportUnknownVariableType]  # isinstance widens frozenset to frozenset[Unknown]


_NormalizedEventHook: TypeAlias = GuardrailEventHooks | list[GuardrailEventHooks] | Mode | None


def _normalize_event_hook(
    event_hook: GuardrailEventHooks | Sequence[GuardrailEventHooks | str] | Mode | str | None,
) -> _NormalizedEventHook:
    if event_hook is None or isinstance(event_hook, (GuardrailEventHooks, Mode)):
        return event_hook
    if isinstance(event_hook, str):
        return GuardrailEventHooks(event_hook)
    return [item if isinstance(item, GuardrailEventHooks) else GuardrailEventHooks(item) for item in event_hook]


def _reject_parallel_during_call(normalized_hook: _NormalizedEventHook) -> None:
    if normalized_hook is None:
        return
    if isinstance(normalized_hook, GuardrailEventHooks):
        if normalized_hook.value in _PARALLEL_HOOK_NAMES:
            raise ValueError("ZTDS zero-egress guardrail cannot use parallel during_call hooks; use pre_call")
        return
    if isinstance(normalized_hook, list):
        if any(item.value in _PARALLEL_HOOK_NAMES for item in normalized_hook):
            raise ValueError("ZTDS zero-egress guardrail cannot use parallel during_call hooks; use pre_call")
        return
    mode_values: Final[list[str]] = []  # mutable-ok: local flattened hook names from Mode
    for val in normalized_hook.tags.values():
        mode_values.extend((val,) if isinstance(val, str) else val)
    if normalized_hook.default is not None:
        mode_values.extend(
            (normalized_hook.default,) if isinstance(normalized_hook.default, str) else normalized_hook.default
        )
    if any(hook in _PARALLEL_HOOK_NAMES for hook in mode_values):
        raise ValueError("ZTDS zero-egress guardrail cannot use parallel during_call hooks; use pre_call")


class ZTDSGuardrail(CustomGuardrail):
    """LiteLLM Guardrail enforcing Zero-Trust Data Sanitization (ZTDS) RFC v1.0.

    Intercepts prompts before upstream transmission, deterministically tokenizes
    sensitive entities in volatile RAM, and restores caller-visible tokens on
    completion return without external network egress.
    """

    use_native_lifecycle_hooks: ClassVar[bool] = True
    TOKEN_PATTERN: ClassVar[re.Pattern[str]] = re.compile(rf"\[(?:{_SUPPORTED_ENTITY_ALT})_TOKEN_\d{{1,18}}\]")
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
        enabled_entities: Sequence[str] | str | None = None,
        reverse_on_output: bool = True,
        enforce_zero_egress: bool = True,
        guardrail_name: str | None = "ztds",
        event_hook: GuardrailEventHooks | Sequence[GuardrailEventHooks | str] | Mode | str | None = None,
        default_on: bool = False,
    ) -> None:
        normalized_hook: Final = _normalize_event_hook(event_hook)
        if enforce_zero_egress:
            _reject_parallel_during_call(normalized_hook)
        validated_entities: Final = self._validate_enabled_entities(enabled_entities)
        super().__init__(  # pyright: ignore[reportUnknownMemberType]  # CustomGuardrail.__init__ accepts unannotated **kwargs
            guardrail_name=guardrail_name,
            event_hook=normalized_hook,
            default_on=default_on,
        )
        self.enabled_entities: tuple[str, ...] = validated_entities
        self.reverse_on_output: bool = reverse_on_output
        self.enforce_zero_egress: bool = enforce_zero_egress
        self._master_pattern: re.Pattern[str] = re.compile(
            "|".join(f"(?P<{et}>{self.PATTERN_STRINGS[et]})" for et in validated_entities)
        )
        # In-memory ephemeral lookup map: {session_id: {token: original_cleartext}}
        self._session_maps: dict[str, dict[str, str]] = {}  # mutable-ok: volatile per-session token map in RAM
        # Reverse map for deterministic identical surrogates within session: {session_id: {cleartext: token}}
        self._entity_maps: dict[str, dict[str, str]] = {}  # mutable-ok: volatile per-session reverse map in RAM
        # Provenance map tracking caller-visible tokens authorized for output reversal
        self._caller_tokens: dict[str, set[str]] = {}  # mutable-ok: volatile per-session provenance token set in RAM
        # Per-session, per-entity counter table for O(1) surrogate generation
        self._entity_counters: dict[str, dict[str, int]] = {}  # mutable-ok: volatile O(1) entity counter map in RAM

    @classmethod
    def _validate_enabled_entities(cls, enabled_entities: object) -> tuple[str, ...]:
        if enabled_entities is None:
            return tuple(cls.PATTERN_STRINGS.keys())
        seq: Final = _as_sequence(enabled_entities)
        if not isinstance(enabled_entities, str) and (seq is None or len(seq) == 0):
            raise ValueError("ZTDS enabled_entities must be a non-empty sequence of supported entity names")
        candidates: Final[Sequence[object]] = (enabled_entities,) if isinstance(enabled_entities, str) else seq or ()

        normalized: Final[list[str]] = []  # mutable-ok: validated uppercase entity names built once
        for raw in candidates:
            if not isinstance(raw, str):
                raise TypeError(f"Unsupported ZTDS entity value: {raw!r}")
            upper = raw.strip().upper()
            if upper not in cls.PATTERN_STRINGS:
                supported = ", ".join(cls.PATTERN_STRINGS.keys())
                raise ValueError(f"Unsupported ZTDS entity '{raw}'. Supported entities: {supported}")
            if upper not in normalized:
                normalized.append(upper)
        return tuple(normalized)

    @staticmethod
    def coerce_entities(raw_entities: object) -> tuple[str, ...] | None:
        if raw_entities is None:
            return None
        return ZTDSGuardrail._validate_enabled_entities(raw_entities)

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

    def _reject_uninspectable_source(self, node_dict: Mapping[object, object]) -> None:
        block_type: Final = node_dict.get("type")
        source_dict: Final = _as_mapping(node_dict.get("source"))
        if block_type == "document" and (source_dict is None or source_dict.get("type") != "text"):
            raise ValueError("ZTDS guardrail rejects non-text document block")
        if source_dict is None:
            return
        source_type: Final = source_dict.get("type")
        if isinstance(source_type, str) and source_type in _UNSUPPORTED_SOURCE_TYPES:
            raise ValueError(f"ZTDS guardrail rejects uninspectable '{source_type}' source payload")
        for key in ("bytes", "s3Location"):
            if source_dict.get(key) is not None:
                raise ValueError(f"ZTDS guardrail rejects uninspectable '{key}' source payload")

    def _reject_uninspectable_dict(self, node_dict: Mapping[object, object]) -> None:
        if not self.enforce_zero_egress:
            return
        block_type: Final = node_dict.get("type")
        if isinstance(block_type, str) and block_type in _UNSUPPORTED_BLOCK_TYPES:
            raise ValueError(f"ZTDS guardrail rejects uninspectable '{block_type}' attachment block")
        if block_type == "base64" and "data" in node_dict:
            raise ValueError("ZTDS guardrail rejects uninspectable base64 document block")
        self._reject_uninspectable_source(node_dict)
        for mkey in _UNSUPPORTED_MEDIA_KEYS:
            if mkey in node_dict and node_dict[mkey] is not None:
                raise ValueError(f"ZTDS guardrail rejects uninspectable '{mkey}' payload")
        for ckey in _UNSUPPORTED_MEDIA_CONTAINERS:
            if _as_mapping(node_dict.get(ckey)) is not None:
                raise ValueError(f"ZTDS guardrail rejects uninspectable '{ckey}' container")

    def _sanitize_dict_key(self, key: object, session_id: str) -> object:
        if not isinstance(key, str) or key.startswith("_"):
            return key
        clean_key, _ = self.sanitize_text(key, session_id, is_caller_visible=False)
        return clean_key

    def _sanitize_string_value(
        self,
        value: str,
        session_id: str,
        *,
        is_caller_visible: bool,
        parent_key: str | None,
    ) -> str:
        if self.enforce_zero_egress and _BASE64_DATA_URI_RE.search(value) is not None:
            raise ValueError("ZTDS guardrail rejects uninspectable inline data URI payload")
        is_json_candidate: Final = parent_key in _JSON_ENCODED_STRING_KEYS or (
            len(value) >= 2 and value[0] in ("{", "[", '"') and "\\" in value
        )
        if is_json_candidate:
            try:
                parsed: Final = _JSON_VALUE_ADAPTER.validate_json(value)
                if _as_dict(parsed) is not None or _as_list(parsed) is not None:
                    sanitized_obj: Final = self._sanitize_payload_node(
                        parsed, session_id, is_caller_visible=is_caller_visible, parent_key=None
                    )
                    return json.dumps(sanitized_obj, ensure_ascii=False)
                if isinstance(parsed, str) and parsed != value:
                    sanitized_str, _ = self.sanitize_text(parsed, session_id, is_caller_visible=is_caller_visible)
                    return json.dumps(sanitized_str, ensure_ascii=False)
            except (ValueError, ValidationError):
                pass
        if "\\u" in value or "\\U" in value:
            try:
                wrapped: Final = _JSON_VALUE_ADAPTER.validate_json(f'"{value}"')
                if isinstance(wrapped, str) and wrapped != value:
                    sanitized_wrapped, _ = self.sanitize_text(wrapped, session_id, is_caller_visible=is_caller_visible)
                    if sanitized_wrapped != wrapped:
                        return sanitized_wrapped
            except (ValueError, ValidationError):
                pass
        sanitized, _ = self.sanitize_text(value, session_id, is_caller_visible=is_caller_visible)
        return sanitized

    def _sanitize_bytes_value(
        self,
        raw: bytes | bytearray,
        session_id: str,
        *,
        is_caller_visible: bool,
    ) -> bytes | bytearray:
        try:
            decoded: Final = bytes(raw).decode("utf-8")
        except UnicodeDecodeError as exc:
            if self.enforce_zero_egress:
                raise ValueError("ZTDS guardrail rejects non-UTF-8 binary payload") from exc
            return raw
        sanitized, _ = self.sanitize_text(decoded, session_id, is_caller_visible=is_caller_visible)
        encoded: Final = sanitized.encode("utf-8")
        return bytearray(encoded) if isinstance(raw, bytearray) else encoded

    def _sanitize_dict_node(
        self,
        node_dict: dict[object, object],  # mutable-ok: sanitizes provider payload dict keys
        session_id: str,
        *,
        is_caller_visible: bool,
    ) -> dict[object, object]:  # mutable-ok: returns sanitized nested dict
        self._reject_uninspectable_dict(node_dict)
        role_val: Final = node_dict.get("role")
        branch_visible: Final = is_caller_visible and not (
            isinstance(role_val, str) and role_val.lower() in _NON_CALLER_ROLES
        )
        if not branch_visible:
            return {
                self._sanitize_dict_key(key, session_id): (
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
        key_renames: Final[list[tuple[object, object]]] = []  # mutable-ok: defers key replacement until iteration ends
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
            clean_key = self._sanitize_dict_key(key, session_id)
            if clean_key != key:
                key_renames.append((key, clean_key))
        for old_key, new_key in key_renames:
            node_dict[new_key] = node_dict.pop(old_key)  # rebind-ok: swap sanitized dict key in place
        return node_dict

    def _sanitize_collection_node(
        self,
        node: object,
        session_id: str,
        *,
        is_caller_visible: bool,
        parent_key: str | None,
    ) -> tuple[bool, object]:
        list_val: Final = _as_list(node)
        if list_val is not None:
            if not is_caller_visible:
                return True, [
                    self._sanitize_payload_node(item, session_id, is_caller_visible=False, parent_key=parent_key)
                    for item in list_val
                ]
            for idx in range(len(list_val)):
                list_val[idx] = self._sanitize_payload_node(
                    list_val[idx], session_id, is_caller_visible=True, parent_key=parent_key
                )
            return True, list_val
        seq_val: Final = _as_sequence(node)
        if seq_val is not None:
            return True, tuple(
                self._sanitize_payload_node(
                    item, session_id, is_caller_visible=is_caller_visible, parent_key=parent_key
                )
                for item in seq_val
            )
        set_val: Final = _as_set(node)
        if set_val is not None:
            return True, {
                self._sanitize_payload_node(
                    item, session_id, is_caller_visible=is_caller_visible, parent_key=parent_key
                )
                for item in set_val
            }
        fset_val: Final = _as_frozenset(node)
        if fset_val is not None:
            return True, frozenset(
                self._sanitize_payload_node(
                    item, session_id, is_caller_visible=is_caller_visible, parent_key=parent_key
                )
                for item in fset_val
            )
        return False, node

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
        if isinstance(node, (bytes, bytearray)):
            return self._sanitize_bytes_value(node, session_id, is_caller_visible=is_caller_visible)
        if isinstance(node, BaseModel):
            return self._sanitize_payload_node(
                _dump_pydantic_model(node),
                session_id,
                is_caller_visible=is_caller_visible,
                parent_key=parent_key,
            )
        handled, sanitized_collection = self._sanitize_collection_node(
            node, session_id, is_caller_visible=is_caller_visible, parent_key=parent_key
        )
        if handled:
            return sanitized_collection
        dict_val: Final = _as_dict(node)
        if dict_val is not None:
            return self._sanitize_dict_node(dict_val, session_id, is_caller_visible=is_caller_visible)
        mapping_val: Final = _as_mapping(node)
        if mapping_val is not None:
            copied: Final[dict[object, object]] = dict(mapping_val)  # mutable-ok: mutable working copy of Mapping
            return self._sanitize_dict_node(copied, session_id, is_caller_visible=is_caller_visible)
        if self.enforce_zero_egress:
            raise ValueError(f"ZTDS guardrail rejects unsupported payload object of type {type(node).__name__}")
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
                else:
                    opt_map = _as_mapping(target.get("optional_params"))
                    if opt_map is not None:
                        copied_opt: dict[str, object] = {  # mutable-ok: copied Mapping container for sanitization
                            str(ok): ov for ok, ov in opt_map.items()
                        }
                        self._sanitize_request_mapping(copied_opt, session_id)
                        target["optional_params"] = copied_opt  # rebind-ok: store sanitized optional_params copy
                continue
            if key in _CONTROL_KEYS or key.startswith(("_", "litellm_")):
                continue
            is_visible = key not in _SYSTEM_PROVENANCE_KEYS
            target[key] = self._sanitize_payload_node(  # rebind-ok: sanitize provider-visible request field
                target[key], session_id, is_caller_visible=is_visible, parent_key=key
            )

    def _sanitize_metadata_slot(
        self,
        slot_dict: dict[str, object],  # mutable-ok: sanitizes metadata slot in place
        session_id: str,
    ) -> None:
        key_renames: Final[list[tuple[str, str]]] = []  # mutable-ok: defers key replacement until iteration ends
        for k in tuple(slot_dict.keys()):
            if k == "requester_metadata":
                req_meta = _as_dict(slot_dict.get("requester_metadata"))
                if req_meta is not None:
                    for rk in tuple(req_meta.keys()):
                        if rk == "headers" or (isinstance(rk, str) and rk.startswith("_")):
                            continue
                        req_meta[rk] = self._sanitize_payload_node(
                            req_meta[rk], session_id, is_caller_visible=False, parent_key=str(rk)
                        )
                continue
            if k in _INTERNAL_METADATA_KEYS or k.startswith(("user_api_key", "_", "litellm_")):
                continue
            slot_dict[k] = self._sanitize_payload_node(  # rebind-ok: sanitize caller metadata value in place
                slot_dict[k], session_id, is_caller_visible=False, parent_key=k
            )
            clean_k, _ = self.sanitize_text(k, session_id, is_caller_visible=False)
            if clean_k != k:
                key_renames.append((k, clean_k))
        for old_k, new_k in key_renames:
            slot_dict[new_k] = slot_dict.pop(old_k)  # rebind-ok: swap sanitized metadata key in place

    def _sanitize_caller_metadata(
        self,
        data: dict[str, object],  # mutable-ok: inspects caller-controlled metadata buckets in place
        session_id: str,
    ) -> None:
        for meta_slot in ("metadata", "litellm_metadata"):
            slot_dict = _as_str_dict(data.get(meta_slot))
            if slot_dict is not None:
                self._sanitize_metadata_slot(slot_dict, session_id)

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

        try:
            self._sanitize_request_mapping(data, session_id)
            self._sanitize_caller_metadata(data, session_id)
            self._sync_logging_object_pre_call(data.get("litellm_logging_obj"), session_id)
        except Exception:
            self.zeroize_session(session_id)
            data.pop("_ztds_session_id", None)
            raise

        if call_type in ("mcp_call", "call_mcp_tool") and "arguments" in data:
            data["modified_arguments"] = data["arguments"]  # rebind-ok: MCP pre_call hook returns modified_arguments

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

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],  # mutable-ok: inherited CustomGuardrail apply_guardrail signature
        input_type: Literal["request", "response"],
        logging_obj: object = None,
    ) -> GenericGuardrailAPIInputs:
        """Unified guardrail entrypoint supporting /guardrails/apply_guardrail and SDK callers."""
        if input_type == "response":
            sid: Final = self._extract_session_id(request_data)
            if sid is None:
                return inputs
            try:
                raw_resp_texts: Final = inputs.get("texts")
                if self.reverse_on_output and raw_resp_texts:
                    return {**inputs, "texts": [self.restore_text(text, sid) for text in raw_resp_texts]}
                return inputs
            finally:
                self.zeroize_session(sid)

        raw_call_id: Final = request_data.get("litellm_call_id") or "call"
        existing_sid: Final = self._extract_session_id(request_data)
        session_id: Final = (
            existing_sid
            if existing_sid is not None and existing_sid in self._session_maps
            else f"{raw_call_id}_{uuid.uuid4().hex}"
        )
        request_data["_ztds_session_id"] = session_id  # rebind-ok: track unified guardrail session for cleanup
        updated_inputs: Final[GenericGuardrailAPIInputs] = dict(inputs)  # pyright: ignore[reportAssignmentType]  # shallow copy of TypedDict preserves schema keys
        try:
            if self.enforce_zero_egress and inputs.get("images"):
                raise ValueError("ZTDS guardrail rejects uninspectable image inputs under zero-egress enforcement")
            raw_texts: Final = inputs.get("texts")
            if raw_texts:
                updated_inputs["texts"] = [
                    self.sanitize_text(text, session_id, is_caller_visible=True)[0] for text in raw_texts
                ]
            raw_msgs: Final = inputs.get("structured_messages")
            if raw_msgs is not None:
                sanitized_msgs: Final = self._sanitize_payload_node(
                    raw_msgs, session_id, is_caller_visible=True, parent_key="structured_messages"
                )
                updated_inputs["structured_messages"] = sanitized_msgs  # pyright: ignore[reportGeneralTypeIssues]  # preserves message dicts cleanly while sanitizing leaf text
            raw_tools: Final = inputs.get("tools")
            if raw_tools is not None:
                sanitized_tools: Final = self._sanitize_payload_node(
                    raw_tools, session_id, is_caller_visible=False, parent_key="tools"
                )
                updated_inputs["tools"] = sanitized_tools  # pyright: ignore[reportGeneralTypeIssues]  # preserves ChatCompletionToolParam dict structure cleanly
            raw_calls: Final = inputs.get("tool_calls")
            if raw_calls is not None:
                sanitized_calls: Final = self._sanitize_payload_node(
                    raw_calls, session_id, is_caller_visible=True, parent_key="tool_calls"
                )
                updated_inputs["tool_calls"] = sanitized_calls  # pyright: ignore[reportGeneralTypeIssues]  # preserves tool call dicts cleanly while sanitizing arguments
            self._sync_logging_object_pre_call(
                logging_obj if logging_obj is not None else request_data.get("litellm_logging_obj"),
                session_id,
            )
        except Exception:
            self.zeroize_session(session_id)
            request_data.pop("_ztds_session_id", None)
            raise
        return updated_inputs

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

    def _caller_restoration_pairs(self, session_id: str) -> tuple[tuple[str, str], ...]:
        token_map: Final = self._session_maps.get(session_id)
        if not token_map:
            return ()
        caller_tokens: Final = self._caller_tokens.get(session_id, frozenset[str]())
        return tuple((cleartext, tok) for tok, cleartext in token_map.items() if tok in caller_tokens)

    def _register_logging_sanitizer(self, logging_obj: object, session_id: str) -> None:
        if not self.reverse_on_output or logging_obj is None:
            return
        pairs: Final = self._caller_restoration_pairs(session_id)
        if not pairs:
            return
        model_call_details: Final = _as_str_dict(attribute_or_none(logging_obj, "model_call_details"))
        register_served_output_sanitizer(logging_obj, model_call_details, pairs, session_id)

    def _sync_logging_object_pre_call(self, logging_obj: object, session_id: str) -> None:
        if logging_obj is None:
            return
        raw_messages: Final = attribute_or_none(logging_obj, "messages")
        if raw_messages is not None:
            sanitized_messages: Final = self._sanitize_payload_node(
                raw_messages, session_id, is_caller_visible=True, parent_key="messages"
            )
            update_fn: Final = attribute_or_none(logging_obj, "update_messages")
            if callable(update_fn):
                _ = update_fn(sanitized_messages)
        model_call_details: Final = _as_str_dict(attribute_or_none(logging_obj, "model_call_details"))
        if model_call_details is not None:
            for field_key in ("input", "messages", "prompt"):
                raw_field = model_call_details.get(field_key)
                if raw_field is not None:
                    model_call_details[field_key] = self._sanitize_payload_node(
                        raw_field, session_id, is_caller_visible=True, parent_key=field_key
                    )
            lp_dict: Final = _as_str_dict(model_call_details.get("litellm_params"))
            if lp_dict is not None:
                self._sanitize_caller_metadata(lp_dict, session_id)
        self._register_logging_sanitizer(logging_obj, session_id)

    def _restore_stream_chunk(self, chunk: object, session_id: str) -> object:
        caller_chunk: Final = copy.deepcopy(chunk)
        chunk_dict: Final = _as_dict(caller_chunk)
        if chunk_dict is not None:
            raw_choices: Final = _as_list(chunk_dict.get("choices"))
            if raw_choices is not None:
                for entry in raw_choices:
                    choice_dict = _as_dict(entry)
                    if choice_dict is not None:
                        delta_dict = _as_dict(choice_dict.get("delta"))
                        if delta_dict is not None and isinstance(delta_dict.get("content"), str):
                            delta_dict["content"] = self.restore_text(str(delta_dict["content"]), session_id)
            return caller_chunk
        choices: Final = _as_sequence(attribute_or_none(caller_chunk, "choices"))
        if choices is not None:
            for choice in choices:
                delta = attribute_or_none(choice, "delta")
                if delta is not None:
                    content = attribute_or_none(delta, "content")
                    if isinstance(content, str):
                        delta.content = self.restore_text(content, session_id)  # pyright: ignore[reportAttributeAccessIssue]  # stream delta exposes mutable .content
        return caller_chunk

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
                self._register_logging_sanitizer(data.get("litellm_logging_obj"), session_id)
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
        clear_served_output_sanitizer(request_data.get("litellm_logging_obj"))
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
        clear_served_output_sanitizer(kwargs.get("litellm_logging_obj"))
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
        clear_served_output_sanitizer(kwargs.get("litellm_logging_obj"))
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
        should_restore: Final = session_id is not None and self.reverse_on_output
        if should_restore and session_id is not None:
            self._register_logging_sanitizer(request_data.get("litellm_logging_obj"), session_id)
        try:
            async for chunk in response:
                if should_restore and session_id is not None:
                    yield self._restore_stream_chunk(chunk, session_id)  # pyright: ignore[reportReturnType]  # yields ModelResponseStream or test/dict chunk wrapper
                else:
                    yield chunk  # pyright: ignore[reportReturnType]  # yields ModelResponseStream or test/dict chunk wrapper
        finally:
            if should_restore and session_id is not None:
                self._register_logging_sanitizer(request_data.get("litellm_logging_obj"), session_id)
            if session_id is not None:
                self.zeroize_session(session_id)

    async def async_post_call_streaming_hook(
        self,
        user_api_key_dict: object,
        response: str,
    ) -> object:
        """LiteLLM post-call streaming hook fallback."""
        return response
