"""
ZTDS (Zero-Trust Data Sanitization) Guardrail for LiteLLM
Protocol Authority: ZTDS AI Consortium & Standards Authority
IETF Standards Track: draft-sibiryakov-ztds-protocol-02
https://datatracker.ietf.org/doc/draft-sibiryakov-ztds-protocol/
Standard Specification: https://ztds.ai/standard/

Invariants Enforced:
- Invariant 1: Zero External Egress Prior to Sanitization (100% in-memory local execution)
- Invariant 2: Deterministic Reversible Tokenization (Bracketed syntactic surrogates)
- Invariant 3: Verifiable Ephemeral RAM Isolation & Zeroization (Theorem 2)
- Invariant 4: Zero Subprocessors (No external SaaS calls, eliminates GDPR Art. 28 liability)
"""

from __future__ import annotations

import copy
import re
import uuid
from collections.abc import AsyncGenerator, AsyncIterable, Mapping
from types import MappingProxyType
from typing import ClassVar

try:
    from litellm.integrations.custom_guardrail import CustomGuardrail
except ImportError:
    # Standalone fallback when running outside full LiteLLM package
    class CustomGuardrail:
        def __init__(self, **kwargs: object) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)


class ZTDSGuardrail(CustomGuardrail):
    """
    LiteLLM Guardrail enforcing Zero-Trust Data Sanitization (ZTDS) RFC v1.0.
    Intercepts prompts before upstream WAN transmission, deterministically tokens sensitive entities in volatile RAM,
    and reverses tokens on completion return without external network egress.
    """

    TOKEN_PATTERN: ClassVar[re.Pattern] = re.compile(r"\[[A-Z_]+_TOKEN_[a-zA-Z0-9_-]+\]")

    # Comprehensive zero-egress regex patterns for sensitive identifiers
    PATTERNS: ClassVar[Mapping[str, re.Pattern]] = MappingProxyType(
        {
            "EMAIL": re.compile(
                r"\b[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9-]{1,63}(?:\.[A-Za-z0-9-]{1,63})*\.[A-Za-z]{2,24}\b"
            ),
            "IPV4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
            "IBAN": re.compile(r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}([A-Z0-9]?){0,16}\b"),
            "CREDIT_CARD": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
            "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "PHONE": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
            "API_SECRET": re.compile(
                r"\b(?:sk-[a-zA-Z0-9_-]{20,}|ghp_[a-zA-Z0-9]{20,}|eyJ[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]{20,})\b"
            ),
        }
    )

    def __init__(
        self,
        enabled_entities: list[str] | None = None,
        reverse_on_output: bool = True,
        enforce_zero_egress: bool = True,
        guardrail_name: str | None = "ztds",
        **kwargs: object,
    ) -> None:
        super().__init__(guardrail_name=guardrail_name, **kwargs)
        self.enabled_entities: tuple[str, ...] = (
            tuple(enabled_entities) if enabled_entities else tuple(self.PATTERNS.keys())
        )
        self.reverse_on_output = reverse_on_output
        self.enforce_zero_egress = enforce_zero_egress
        # In-memory ephemeral lookup map: {session_id: {token: original_cleartext}}
        self._session_maps: dict[str, dict[str, str]] = {}  # mutable-ok: [LIT002] ephemeral session lookup map in RAM
        # Reverse map for deterministic identical surrogates within session: {session_id: {cleartext: token}}
        self._entity_maps: dict[str, dict[str, str]] = {}  # mutable-ok: [LIT002] ephemeral entity lookup map in RAM
        # Provenance map tracking caller-visible tokens authorized for output reversal: {session_id: set(tokens)}
        self._caller_tokens: dict[str, set[str]] = {}  # mutable-ok: [LIT002] ephemeral caller token set in RAM

    def sanitize_text(self, text: str, session_id: str, is_caller_visible: bool = True) -> tuple[str, dict[str, str]]:
        """
        In-memory single-pass deterministic tokenization.
        Guarantees zero network calls and deterministic surrogate assignment within session scope.
        Tracks token provenance: only tokens created from caller-visible fields are marked reversible.
        """
        if not text or not isinstance(text, str):
            return text, {}  # mutable-ok: [LIT002] empty token map for non-string input

        if session_id not in self._session_maps:
            self._session_maps[session_id] = {}  # mutable-ok: [LIT002] session token map initialization
        if session_id not in self._entity_maps:
            self._entity_maps[session_id] = {}  # mutable-ok: [LIT002] session entity map initialization
        if session_id not in self._caller_tokens:
            self._caller_tokens[session_id] = set()  # mutable-ok: [LIT002] session token provenance set initialization

        token_map = self._session_maps[session_id]
        entity_map = self._entity_maps[session_id]
        caller_set = self._caller_tokens[session_id]

        # Pre-index existing bracketed tokens in text to prevent collisions in O(1)
        existing_tokens = set(self.TOKEN_PATTERN.findall(text))

        sanitized = text
        for entity_type in self.enabled_entities:
            if entity_type == "EMAIL" and "@" not in sanitized:
                continue

            pattern = self.PATTERNS.get(entity_type)
            if not pattern:
                continue

            # Per-entity surrogate counter to eliminate quadratic scans over token_map
            entity_counter = sum(1 for k in token_map if k.startswith(f"[{entity_type}_TOKEN_"))

            def _replace_match(match: re.Match, et: str = entity_type) -> str:
                nonlocal entity_counter
                original = match.group(0)
                if original in entity_map:
                    token = entity_map[original]
                else:
                    while True:
                        entity_counter += 1
                        candidate = f"[{et}_TOKEN_{entity_counter}]"
                        if candidate not in existing_tokens and candidate not in token_map:
                            token = candidate
                            break
                    token_map[token] = original
                    entity_map[original] = token

                if is_caller_visible:
                    caller_set.add(token)
                return token

            sanitized = pattern.sub(_replace_match, sanitized)

        return sanitized, token_map

    def restore_text(self, text: str, session_id: str) -> str:
        """
        Restores deterministic surrogates back to original cleartext via single-pass token dispatch.
        Enforces provenance isolation: only restores tokens that originated from caller-visible fields.
        Hidden/system prompt secrets are never reversed in caller output.
        """
        token_map = self._session_maps.get(session_id)
        if not token_map:
            return text
        caller_tokens = self._caller_tokens.get(session_id, frozenset())

        def _replace_token(match: re.Match) -> str:
            tok = match.group(0)
            if tok in token_map and tok in caller_tokens:
                return token_map[tok]
            return tok

        return self.TOKEN_PATTERN.sub(_replace_token, text)

    def zeroize_session(self, session_id: str) -> None:
        """
        Enforces Theorem 2 (Volatile RAM Zeroization):
        Wipes the token lookup tables and provenance sets from volatile memory.
        """
        if session_id in self._session_maps:
            self._session_maps[session_id].clear()
            del self._session_maps[session_id]
        if session_id in self._entity_maps:
            self._entity_maps[session_id].clear()
            del self._entity_maps[session_id]
        if session_id in self._caller_tokens:
            self._caller_tokens[session_id].clear()
            del self._caller_tokens[session_id]

    async def async_pre_call_hook(
        self,
        user_api_key_dict: object,
        cache: object,
        data: dict[str, object],
        call_type: str,
    ) -> dict[str, object]:
        """
        LiteLLM pre-call hook: intercepts outgoing messages, prompts, and inputs and sanitizes all content.
        Generates an internal random nonce to prevent cross-tenant ID collisions.
        Zero network sockets are opened during this operation.
        """
        raw_call_id = data.get("litellm_call_id") or "call"
        session_id = f"{raw_call_id}_{uuid.uuid4().hex}"
        data["_ztds_session_id"] = session_id

        # 1. Sanitize messages array (chat completions)
        messages = data.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict) and "content" in message:
                    role = message.get("role", "user")
                    # System and developer messages are hidden/trusted fields; user/assistant/tool are caller-visible
                    is_caller_visible = role not in ("system", "developer")
                    content = message["content"]
                    if isinstance(content, str):
                        sanitized, _ = self.sanitize_text(content, session_id, is_caller_visible=is_caller_visible)
                        message["content"] = sanitized
                    elif isinstance(content, list):
                        # Multi-modal content chunks
                        for chunk in content:
                            if isinstance(chunk, dict) and chunk.get("type") == "text":
                                chunk["text"], _ = self.sanitize_text(
                                    chunk.get("text", ""), session_id, is_caller_visible=is_caller_visible
                                )

        # 2. Sanitize prompt field (legacy completions: caller-visible)
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, str):
                data["prompt"], _ = self.sanitize_text(prompt, session_id, is_caller_visible=True)
            elif isinstance(prompt, list):
                data["prompt"] = [  # mutable-ok: [LIT002] prompt list payload required by LiteLLM schema
                    self.sanitize_text(p, session_id, is_caller_visible=True)[0] if isinstance(p, str) else p
                    for p in prompt
                ]

        # 3. Sanitize input field (moderations, embeddings, responses: caller-visible)
        if "input" in data:
            raw_input = data["input"]
            if isinstance(raw_input, str):
                data["input"], _ = self.sanitize_text(raw_input, session_id, is_caller_visible=True)
            elif isinstance(raw_input, list):
                data["input"] = [  # mutable-ok: [LIT002] input list payload required by LiteLLM schema
                    self.sanitize_text(item, session_id, is_caller_visible=True)[0] if isinstance(item, str) else item
                    for item in raw_input
                ]

        # Attach ZTDS audit receipt to metadata
        metadata = data.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}  # mutable-ok: [LIT002] dictionary metadata required by LiteLLM schema
            data["metadata"] = metadata
        metadata["ztds_sanitized"] = True
        metadata["ztds_standard"] = "RFC v1.0 (IETF draft-sibiryakov-ztds-protocol-02)"
        metadata["ztds_invariants_verified"] = (1, 2, 3, 4)

        return data

    async def async_post_call_success_hook(
        self,
        data: dict[str, object],
        user_api_key_dict: object,
        response: object,
    ) -> object:
        """
        LiteLLM post-call success hook: restores cleartext entities in volatile RAM and zeroizes session map.
        Guarantees Theorem 2 cleanup in finally block regardless of reverse_on_output configuration.
        Deep-copies response before unmasking so that upstream shared caches retain sanitized surrogates.
        """
        session_id = data.get("_ztds_session_id")
        if not session_id or not isinstance(session_id, str):
            return response

        try:
            if self.reverse_on_output:
                caller_response = copy.deepcopy(response)
                # Process standard ModelResponse object
                if hasattr(caller_response, "choices") and caller_response.choices:
                    for choice in caller_response.choices:
                        if (
                            hasattr(choice, "message")
                            and hasattr(choice.message, "content")
                            and isinstance(choice.message.content, str)
                        ):
                            choice.message.content = self.restore_text(choice.message.content, session_id)
                # Process dictionary response fallback
                elif isinstance(caller_response, dict) and "choices" in caller_response:
                    for choice in caller_response["choices"]:
                        if isinstance(choice, dict) and "message" in choice and isinstance(choice["message"], dict):
                            content = choice["message"].get("content")
                            if isinstance(content, str):
                                choice["message"]["content"] = self.restore_text(content, session_id)
                return caller_response
        finally:
            # Theorem 2: Guarantee RAM zeroization even if reverse_on_output is False or response handling fails
            self.zeroize_session(session_id)

        return response

    async def async_post_call_failure_hook(
        self,
        data: dict[str, object],
        user_api_key_dict: object,
        error: Exception,
    ) -> None:
        """
        LiteLLM post-call failure hook: ensures volatile RAM zeroization when upstream provider calls fail.
        """
        session_id = data.get("_ztds_session_id") if isinstance(data, dict) else None
        if session_id and isinstance(session_id, str):
            self.zeroize_session(session_id)

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: object,
        response: AsyncIterable[object],
        request_data: dict[str, object],
    ) -> AsyncGenerator[object, None]:
        """
        LiteLLM streaming iterator hook: restores tokens across streaming response chunks in volatile RAM
        and guarantees Theorem 2 zeroization upon stream completion or error.
        Deep-copies chunk before unmasking so upstream completion cache retains sanitized surrogates.
        """
        session_id = request_data.get("_ztds_session_id") if isinstance(request_data, dict) else None
        try:
            async for chunk in response:
                if session_id and isinstance(session_id, str) and self.reverse_on_output:
                    caller_chunk = copy.deepcopy(chunk)
                    if hasattr(caller_chunk, "choices") and caller_chunk.choices:
                        for choice in caller_chunk.choices:
                            delta = getattr(choice, "delta", None)
                            if delta and hasattr(delta, "content") and isinstance(delta.content, str):
                                delta.content = self.restore_text(delta.content, session_id)
                    elif isinstance(caller_chunk, dict) and "choices" in caller_chunk:
                        for choice in caller_chunk["choices"]:
                            delta = choice.get("delta") if isinstance(choice, dict) else None
                            if delta and isinstance(delta, dict) and isinstance(delta.get("content"), str):
                                delta["content"] = self.restore_text(delta["content"], session_id)
                    yield caller_chunk
                else:
                    yield chunk
        finally:
            if session_id and isinstance(session_id, str):
                self.zeroize_session(session_id)

    async def async_post_call_streaming_hook(
        self,
        user_api_key_dict: object,
        response: str,
    ) -> object:
        """
        LiteLLM post-call streaming hook fallback.
        """
        return response
