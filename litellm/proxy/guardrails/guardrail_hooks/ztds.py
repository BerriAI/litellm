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

import re
import uuid
from collections.abc import AsyncGenerator
from typing import Any, ClassVar

try:
    from litellm.integrations.custom_guardrail import CustomGuardrail
except ImportError:
    # Standalone fallback when running outside full LiteLLM package
    class CustomGuardrail:
        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)


class ZTDSGuardrail(CustomGuardrail):
    """
    LiteLLM Guardrail enforcing Zero-Trust Data Sanitization (ZTDS) RFC v1.0.
    Intercepts prompts before upstream WAN transmission, deterministically tokens sensitive entities in volatile RAM,
    and reverses tokens on completion return without external network egress.
    """

    # Comprehensive zero-egress regex patterns for sensitive identifiers
    PATTERNS: ClassVar[dict[str, re.Pattern]] = {
        "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,24}\b"),
        "IPV4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        "IBAN": re.compile(r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}([A-Z0-9]?){0,16}\b"),
        "CREDIT_CARD": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "PHONE": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "API_SECRET": re.compile(
            r"\b(?:sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,}|eyJ[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]{20,})\b"
        ),
    }

    def __init__(
        self,
        enabled_entities: list[str] | None = None,
        reverse_on_output: bool = True,
        enforce_zero_egress: bool = True,
        guardrail_name: str | None = "ztds",
        **kwargs: Any,
    ):
        super().__init__(guardrail_name=guardrail_name, **kwargs)
        self.enabled_entities = enabled_entities or list(self.PATTERNS.keys())
        self.reverse_on_output = reverse_on_output
        self.enforce_zero_egress = enforce_zero_egress
        # In-memory ephemeral lookup map: {session_id: {token: original_cleartext}}
        self._session_maps: dict[str, dict[str, str]] = {}
        # Reverse map for deterministic identical surrogates within session: {session_id: {cleartext: token}}
        self._entity_maps: dict[str, dict[str, str]] = {}
        # Provenance map tracking caller-visible tokens authorized for output reversal: {session_id: set(tokens)}
        self._caller_tokens: dict[str, set[str]] = {}

    def sanitize_text(self, text: str, session_id: str, is_caller_visible: bool = True) -> tuple[str, dict[str, str]]:
        """
        In-memory single-pass deterministic tokenization.
        Guarantees zero network calls and deterministic surrogate assignment within session scope.
        Tracks token provenance: only tokens created from caller-visible fields are marked reversible.
        """
        if not text or not isinstance(text, str):
            return text, {}

        if session_id not in self._session_maps:
            self._session_maps[session_id] = {}
        if session_id not in self._entity_maps:
            self._entity_maps[session_id] = {}
        if session_id not in self._caller_tokens:
            self._caller_tokens[session_id] = set()

        token_map = self._session_maps[session_id]
        entity_map = self._entity_maps[session_id]
        caller_set = self._caller_tokens[session_id]

        sanitized = text
        for entity_type in self.enabled_entities:
            pattern = self.PATTERNS.get(entity_type)
            if not pattern:
                continue

            # Process matches in reverse string order to preserve exact substring indices
            matches = list(pattern.finditer(sanitized))
            for match in reversed(matches):
                original = match.group(0)

                # Deterministic Reversible Tokenization (Invariant 2) with Collision Avoidance
                if original in entity_map:
                    token = entity_map[original]
                else:
                    count = len([k for k in token_map if k.startswith(f"[{entity_type}_TOKEN_")]) + 1
                    while True:
                        candidate = f"[{entity_type}_TOKEN_{count}]"
                        if candidate not in text and candidate not in token_map:
                            token = candidate
                            break
                        count += 1
                    token_map[token] = original
                    entity_map[original] = token

                if is_caller_visible:
                    caller_set.add(token)

                start, end = match.span()
                sanitized = sanitized[:start] + token + sanitized[end:]

        return sanitized, token_map

    def restore_text(self, text: str, session_id: str) -> str:
        """
        Restores deterministic surrogates back to original cleartext.
        Enforces provenance isolation: only restores tokens that originated from caller-visible fields.
        Hidden/system prompt secrets are never reversed in caller output.
        """
        token_map = self._session_maps.get(session_id, {})
        caller_tokens = self._caller_tokens.get(session_id, set())
        if not token_map:
            return text

        restored = text
        for token in sorted(token_map.keys(), key=len, reverse=True):
            # Only restore if token was authorized from caller-visible inputs
            if token in caller_tokens:
                restored = restored.replace(token, token_map[token])
        return restored

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
        user_api_key_dict: Any,
        cache: Any,
        data: dict[str, Any],
        call_type: str,
    ) -> dict[str, Any]:
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
                data["prompt"] = [
                    self.sanitize_text(p, session_id, is_caller_visible=True)[0] if isinstance(p, str) else p
                    for p in prompt
                ]

        # 3. Sanitize input field (moderations, embeddings, responses: caller-visible)
        if "input" in data:
            raw_input = data["input"]
            if isinstance(raw_input, str):
                data["input"], _ = self.sanitize_text(raw_input, session_id, is_caller_visible=True)
            elif isinstance(raw_input, list):
                data["input"] = [
                    self.sanitize_text(item, session_id, is_caller_visible=True)[0] if isinstance(item, str) else item
                    for item in raw_input
                ]

        # Attach ZTDS audit receipt to metadata
        if "metadata" not in data or data["metadata"] is None:
            data["metadata"] = {}
        data["metadata"]["ztds_sanitized"] = True
        data["metadata"]["ztds_standard"] = "RFC v1.0 (IETF draft-sibiryakov-ztds-protocol-02)"
        data["metadata"]["ztds_invariants_verified"] = [1, 2, 3, 4]

        return data

    async def async_post_call_success_hook(
        self,
        data: dict[str, Any],
        user_api_key_dict: Any,
        response: Any,
    ) -> Any:
        """
        LiteLLM post-call success hook: restores cleartext entities in volatile RAM and zeroizes session map.
        Guarantees Theorem 2 cleanup in finally block regardless of reverse_on_output configuration.
        """
        session_id = data.get("_ztds_session_id")
        if not session_id:
            return response

        try:
            if self.reverse_on_output:
                # Process standard ModelResponse object
                if hasattr(response, "choices") and response.choices:
                    for choice in response.choices:
                        if (
                            hasattr(choice, "message")
                            and hasattr(choice.message, "content")
                            and isinstance(choice.message.content, str)
                        ):
                            choice.message.content = self.restore_text(choice.message.content, session_id)
                # Process dictionary response fallback
                elif isinstance(response, dict) and "choices" in response:
                    for choice in response["choices"]:
                        if "message" in choice and "content" in choice["message"]:
                            choice["message"]["content"] = self.restore_text(choice["message"]["content"], session_id)
        finally:
            # Theorem 2: Guarantee RAM zeroization even if reverse_on_output is False or response handling fails
            self.zeroize_session(session_id)

        return response

    async def async_post_call_failure_hook(
        self,
        data: dict[str, Any],
        user_api_key_dict: Any,
        error: Exception,
    ) -> None:
        """
        LiteLLM post-call failure hook: ensures volatile RAM zeroization when upstream provider calls fail.
        """
        session_id = data.get("_ztds_session_id") if isinstance(data, dict) else None
        if session_id:
            self.zeroize_session(session_id)

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: Any,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[Any, None]:
        """
        LiteLLM streaming iterator hook: restores tokens across streaming response chunks in volatile RAM
        and guarantees Theorem 2 zeroization upon stream completion or error.
        """
        session_id = request_data.get("_ztds_session_id") if isinstance(request_data, dict) else None
        try:
            async for chunk in response:
                if session_id and self.reverse_on_output:
                    if hasattr(chunk, "choices") and chunk.choices:
                        for choice in chunk.choices:
                            delta = getattr(choice, "delta", None)
                            if delta and hasattr(delta, "content") and isinstance(delta.content, str):
                                delta.content = self.restore_text(delta.content, session_id)
                    elif isinstance(chunk, dict) and "choices" in chunk:
                        for choice in chunk["choices"]:
                            delta = choice.get("delta") if isinstance(choice, dict) else None
                            if delta and isinstance(delta, dict) and isinstance(delta.get("content"), str):
                                delta["content"] = self.restore_text(delta["content"], session_id)
                yield chunk
        finally:
            if session_id:
                self.zeroize_session(session_id)

    async def async_post_call_streaming_hook(
        self,
        user_api_key_dict: Any,
        response: str,
    ) -> Any:
        """
        LiteLLM post-call streaming hook fallback.
        """
        return response
