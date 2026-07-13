"""Compresr guardrail — query-aware, recoverable context compression.

Compresses bulky message content (tool outputs by default) through the
Compresr API before the request reaches the LLM. Each compressed message
carries a hash marker; a ``compresr_retrieve`` tool is injected so the model
can fetch the original content back through the agentic loop when the
compressed version is not enough — making compression recoverable instead
of lossy.

Unlike gateway-side compressors that operate on whole message lists, each
target is compressed *query-aware*: the query sent to Compresr is the intent
of the tool call that produced the message (``name + arguments``, resolved
via ``tool_call_id``), falling back to the last user message.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException
from httpx import Response as HttpxResponse
from typing_extensions import TypeGuard

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.litellm_core_utils.prompt_templates.factory import (
    get_attribute_or_key,
    get_tool_calls_from_response,
    has_tool_with_name,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # helper is untyped in http_handler
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.secret_managers.main import get_secret_str
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.integrations.custom_logger import (
    AgenticLoopPlan,
    AgenticLoopRequestPatch,
)
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LiteLLMLoggingObj,
    )
    from litellm.types.proxy.guardrails.guardrail_hooks.base import (
        GuardrailConfigModel,
    )

BYPASS_HEADER = "x-compresr-bypass"
COMPRESR_RETRIEVE_TOOL_NAME = "compresr_retrieve"
DEFAULT_API_BASE = "https://api.compresr.ai"
DEFAULT_COMPRESSION_MODEL = "latte_v2"
DEFAULT_TARGET_COMPRESSION_RATIO = 0.5
DEFAULT_MIN_CHARS_TO_COMPRESS = 500
_ORIGINALS_TTL_SECONDS = 15 * 60
_MAX_TRACKED_CALLS = 256
_DEFAULT_MAX_BYTES_PER_CALL = 10 * 1024 * 1024
# Aggregate ceiling across all recovery-store entries. max_bytes_per_call only
# bounds a single call; this caps the whole store so many calls cannot exhaust it.
_MAX_TOTAL_STORE_BYTES = 256 * 1024 * 1024
# Max compresr_retrieve calls expanded into a single follow-up (repeats deduped).
_MAX_RETRIEVALS_PER_LOOP = 8
# The shared HTTP client defaults to a 600s read timeout; that is far too long
# for an inline, on-request-path guardrail (a hung Compresr backend would pin
# the request coroutine and a pooled connection for 10 minutes). Bound it so a
# stall is routed through the fail policy quickly instead.
_COMPRESS_TIMEOUT_SECONDS = 60.0
_SOURCE_TAG = "integration:litellm"
# Request-content fields the compression_params passthrough must never
# override — they carry the actual message content/queries being compressed.
_RESERVED_COMPRESSION_PARAM_KEYS = frozenset({"context", "query", "inputs"})
_BLOCKED_METADATA_HOSTS = frozenset(
    {
        "metadata.google.internal",
        "metadata.goog",
    }
)
_BLOCKED_METADATA_IPS = frozenset(
    ipaddress.ip_address(ip) for ip in ("169.254.169.254", "fd00:ec2::254", "100.100.100.200")
)


def _parse_ip_literal(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Parse ``host`` as an IP literal, covering the alternate spellings the
    socket layer accepts (decimal/hex single-integer IPv4, IPv4-mapped IPv6)
    so a blocked address cannot be smuggled past a string comparison."""
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        try:
            addr = ipaddress.ip_address(int(host, 0))
        except (TypeError, ValueError):
            return None
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        return addr.ipv4_mapped
    return addr


def _validate_api_base(url: str) -> str:
    """Return ``url`` if it passes basic outbound-target checks, else raise.

    Best-effort defense in depth against a mis- or maliciously-configured
    ``api_base``: rejects non-http(s) schemes and the well-known cloud-metadata
    IPs/hostnames (including alternate IP-literal encodings). Private-range
    hosts are allowed because on-prem Compresr deployments legitimately live
    there.

    This is NOT a complete SSRF control, and deliberately does no DNS resolution
    (which would block the event loop at construction and still not close the
    gap): the shared outbound client follows redirects and re-resolves DNS on
    every request, so a host that passes here can still redirect to, or later
    resolve to, a blocked address (TOCTOU / DNS rebinding). ``api_base`` is
    trusted operator config rather than end-user input, so this is an accepted
    limitation; closing it fully requires the shared HTTP handler to expose a
    no-redirect / IP-pinned request path.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Compresr guardrail api_base must be http or https, got scheme={parsed.scheme!r}")
    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("Compresr guardrail api_base has no host")
    ip_literal = _parse_ip_literal(host)
    if host in _BLOCKED_METADATA_HOSTS or (ip_literal is not None and ip_literal in _BLOCKED_METADATA_IPS):
        raise ValueError(f"Compresr guardrail api_base {host!r} is a blocked cloud-metadata host")
    return url


def _is_str_object_dict(value: object) -> TypeGuard[dict[str, object]]:  # guard-ok: isinstance narrows correctly; predicate is trivially correct  # fmt: skip
    return isinstance(value, dict)


def _is_object_list(value: object) -> TypeGuard[list[object]]:  # guard-ok: isinstance narrows correctly; predicate is trivially correct  # fmt: skip
    return isinstance(value, list)


def _content_to_text(content: object) -> str:
    """Collapse a message ``content`` (str or list-of-parts) to plain text.

    For the multimodal list shape, joins ``{type: "text", text: ...}`` parts
    with blank-line separators; non-text parts are ignored.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n\n".join(parts)
    return ""


def _replace_text_in_content(content: object, new_text: str) -> object:
    """Write ``new_text`` back into a ``content`` value, preserving shape.

    ``str`` content is replaced directly. For list-of-parts content the first
    text part carries ``new_text``, later text parts are dropped, and
    non-text parts (images, audio, files) pass through untouched.
    """
    if isinstance(content, str):
        return new_text
    if isinstance(content, list):
        out: list[object] = []
        replaced = False
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                if not replaced:
                    out.append({**part, "text": new_text})
                    replaced = True
                continue
            out.append(part)
        if not replaced:
            out.insert(0, {"type": "text", "text": new_text})
        return out
    return new_text


def _render_tool_intent(fn: dict[str, object]) -> str:
    name = str(fn.get("name") or "").strip()
    args = fn.get("arguments")
    if isinstance(args, dict):
        try:
            args_str = json.dumps(args, separators=(",", ":"))
        except (TypeError, ValueError):
            args_str = str(args)
    else:
        args_str = str(args).strip() if args is not None else ""
    if name and args_str:
        return f"{name}: {args_str}"
    return name or args_str


def _query_for_target(messages: list[dict[str, object]], target_idx: int, fallback: str) -> str:
    """Query used to compress ``messages[target_idx]``.

    Tool/function outputs are compressed against the intent of the tool call
    that produced them (found via ``tool_call_id`` on a prior assistant
    message); everything else uses the last user message.
    """
    msg = messages[target_idx]
    if msg.get("role") not in ("tool", "function"):
        return fallback

    tool_call_id = msg.get("tool_call_id")
    fn_name = msg.get("name")
    for j in range(target_idx - 1, -1, -1):
        prev = messages[j]
        if prev.get("role") != "assistant":
            continue
        tool_calls = prev.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                if tool_call_id and tc.get("id") == tool_call_id:
                    fn = tc.get("function")
                    intent = _render_tool_intent(fn if isinstance(fn, dict) else {})
                    if intent:
                        return intent
        # Legacy function_call fallback: only trust a name match. Without a
        # name on the function-result message, any earlier assistant turn with
        # a function_call would match and attribute the wrong intent.
        fc = prev.get("function_call")
        if isinstance(fc, dict) and fn_name and fc.get("name") == fn_name:
            intent = _render_tool_intent(fc)
            if intent:
                return intent
    return fallback


def _safe_int(value: object) -> int:
    """Parse a token-stat field defensively.

    A malformed-but-200 response must not raise here: ``_call_compress`` has
    already returned successfully, so the fail_open/fail_closed decision is
    behind us. A bare ``int()`` on a non-numeric field would surface as an
    unhandled 500 even when ``fail_open`` is configured.
    """
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def _safe_response_text(response: object, limit: int = 500) -> str:
    """Read a response body for error logging without letting the read itself
    raise. A corrupt ``Content-Encoding`` makes ``httpx``'s ``.text`` raise a
    ``DecodingError``; if that happened while building a failure detail it would
    turn an already-handled error into an unhandled 500."""
    try:
        text = getattr(response, "text", "")
    except httpx.DecodingError:
        return "<undecodable response body>"
    return (text or "")[:limit]


def _content_hash(text: str) -> str:
    # surrogatepass so a lone surrogate in untrusted content (valid via a JSON
    # \uXXXX escape) hashes instead of raising past the fail policy.
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:24]


def _entry_bytes(originals: dict[str, str]) -> int:
    """UTF-8 byte size of one recovery-store entry (surrogatepass, like _content_hash)."""
    return sum(len(value.encode("utf-8", "surrogatepass")) for value in originals.values())


def _display_hash(hash_value: str) -> str:
    """Bound a model-supplied hash for logs/fallback text. A real marker hash is
    24 hex chars; a prompt-injected ``compresr_retrieve`` call could pass a huge
    or control-character-laden string, so strip non-printables (no forged log
    lines / ANSI escapes) and cap length before echoing into logs and the
    conversation."""
    printable = "".join(ch for ch in hash_value if ch.isprintable())
    return printable if len(printable) <= 32 else f"{printable[:32]}…"


def _recovery_marker(hash_value: str) -> str:
    return (
        f"\n\n[compresr hash={hash_value}: parts of this content were compressed "
        f"away. If you need the full original, call the "
        f"{COMPRESR_RETRIEVE_TOOL_NAME} tool with this hash.]"
    )


def _build_compresr_retrieve_tool() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": COMPRESR_RETRIEVE_TOOL_NAME,
            "description": (
                "Retrieve the original, uncompressed content behind a Compresr "
                "compression marker. Call this when a compression marker's hash "
                "points at content you need in full."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "hash": {
                        "type": "string",
                        "description": "The 24-character hex hash from the compression marker.",
                    },
                },
                "required": ["hash"],
            },
        },
    }


def has_compresr_retrieve_tool(tools: object) -> bool:
    return has_tool_with_name(tools, COMPRESR_RETRIEVE_TOOL_NAME)


def _extract_compresr_tool_calls(response: object) -> list[dict[str, object]]:
    return [
        {"id": tc["id"], "type": "function", "name": tc["name"], "arguments": tc["arguments"]}
        for tc in get_tool_calls_from_response(response)
        if tc["name"] == COMPRESR_RETRIEVE_TOOL_NAME
    ]


def _resolve_call_id(logging_obj: object) -> str | None:
    """The call id from the framework logging object.

    This value ultimately derives from the client-settable ``x-litellm-call-id``
    header and is echoed back in responses, so it is NOT a trust boundary on its
    own — ``_scoped_store_key`` prefixes it with the caller's virtual-key hash to
    partition the recovery store per tenant. Request-body/kwargs call ids are
    deliberately not consulted here.
    """
    logging_call_id = getattr(logging_obj, "litellm_call_id", None)
    if isinstance(logging_call_id, str) and logging_call_id:
        return logging_call_id
    return None


def _caller_scope(logging_obj: object) -> str:
    """The caller's virtual-key hash, used to partition the recovery store.

    Trust is anchored on the ``UserAPIKeyAuth`` object the proxy sets
    server-side (``metadata.user_api_key_auth``, litellm_pre_call_utils). Its
    ``api_key`` is the hash of the authenticated key. Both metadata spellings
    are scanned (``/v1/messages`` and ``/v1/responses`` carry it under
    ``litellm_metadata``), but the bare ``user_api_key`` *string* is never
    trusted on its own: a JSON request body can place one in the client-supplied
    ``metadata`` field, which is only sanitized on the route's canonical
    container. Returns "" when the proxy runs without per-key auth, in which case
    all traffic is a single trust domain and the call id alone suffices.
    """
    details = getattr(logging_obj, "model_call_details", None)
    if not _is_str_object_dict(details):
        return ""
    litellm_params = details.get("litellm_params")
    for container in (litellm_params, details):
        if not _is_str_object_dict(container):
            continue
        for meta_key in ("metadata", "litellm_metadata"):
            metadata = container.get(meta_key)
            if not _is_str_object_dict(metadata):
                continue
            auth = metadata.get("user_api_key_auth")
            if isinstance(auth, UserAPIKeyAuth) and isinstance(auth.api_key, str) and auth.api_key:
                return auth.api_key
    return ""


def _scoped_store_key(logging_obj: object) -> str | None:
    """Key for the recovery store: caller identity plus framework call id.

    Keying on the call id alone is unsafe: it comes from the client-settable
    ``x-litellm-call-id`` header and is echoed back in responses, so one caller
    could read or evict another's originals by reusing the id. Prefixing the
    unforgeable virtual-key hash binds each entry to the tenant that created it.
    Returns None when there is no call id, which disables recovery for the call.
    """
    call_id = _resolve_call_id(logging_obj)
    if call_id is None:
        return None
    scope = _caller_scope(logging_obj)
    return f"{scope}\x00{call_id}" if scope else call_id


def _is_responses_api_response(response: object) -> bool:
    return isinstance(get_attribute_or_key(response, "output", None), list)


def _is_anthropic_messages_response(response: object) -> bool:
    return isinstance(get_attribute_or_key(response, "content", None), list)


def _assistant_text_from_response(response: object) -> str | None:
    """The assistant's natural-language text from a model response, across chat,
    Anthropic, and Responses shapes. Preserved when the turn is rebuilt for the
    retrieval follow-up so the model's reasoning is not lost."""
    choices = get_attribute_or_key(response, "choices", None)
    if isinstance(choices, list) and choices:
        message = get_attribute_or_key(choices[0], "message", None)
        if message is not None:
            text = _content_to_text(get_attribute_or_key(message, "content", None))
            if text:
                return text
    content = get_attribute_or_key(response, "content", None)
    if isinstance(content, list):
        parts = [
            text
            for block in content
            if get_attribute_or_key(block, "type", None) == "text"
            for text in (get_attribute_or_key(block, "text", None),)
            if isinstance(text, str) and text
        ]
        if parts:
            return "".join(parts)
    output = get_attribute_or_key(response, "output", None)
    if isinstance(output, list):
        parts = []
        for item in output:
            if get_attribute_or_key(item, "type", None) != "message":
                continue
            item_content = get_attribute_or_key(item, "content", None)
            if not isinstance(item_content, list):
                continue
            for chunk in item_content:
                if get_attribute_or_key(chunk, "type", None) == "output_text":
                    text = get_attribute_or_key(chunk, "text", None)
                    if isinstance(text, str) and text:
                        parts.append(text)
        if parts:
            return "".join(parts)
    return None


def _build_assistant_message_from_response(
    response: object,
    retrieved: list[tuple[dict[str, object], str]],
) -> dict[str, object]:
    """Rebuild the chat-completions assistant turn for the retrieval follow-up.

    Only the ``compresr_retrieve`` calls are echoed, each answered by a tool
    result below. Other tool calls made in the same turn are omitted on purpose:
    the follow-up re-runs the model with the recovered content so it re-plans
    them. Echoing them would leave tool_calls with no matching tool result and
    the provider would reject the request.
    """
    return {
        "role": "assistant",
        "content": _assistant_text_from_response(response),
        "tool_calls": [
            {
                "id": tool_call.get("id"),
                "type": "function",
                "function": {
                    "name": tool_call.get("name"),
                    "arguments": json.dumps(tool_call.get("arguments", {})),
                },
            }
            for tool_call, _ in retrieved
        ],
    }


def _build_anthropic_followup_messages(
    response: object,
    retrieved: list[tuple[dict[str, object], str]],
) -> list[dict[str, object]]:
    """Anthropic requires the tool_use block echoed back in an assistant
    message paired with a tool_result block keyed by the same tool_use_id. The
    assistant text is preserved; non-retrieve tool calls are re-planned by the
    follow-up (see _build_assistant_message_from_response)."""
    assistant_content: list[dict[str, object]] = []
    text = _assistant_text_from_response(response)
    if text:
        assistant_content.append({"type": "text", "text": text})
    assistant_content.extend(
        {
            "type": "tool_use",
            "id": tool_call.get("id"),
            "name": tool_call.get("name"),
            "input": tool_call.get("arguments", {}),
        }
        for tool_call, _ in retrieved
    )
    assistant_message: dict[str, object] = {"role": "assistant", "content": assistant_content}
    user_message: dict[str, object] = {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": tool_call.get("id"), "content": content}
            for tool_call, content in retrieved
        ],
    }
    return [assistant_message, user_message]


def _build_responses_followup_items(
    response: object,
    retrieved: list[tuple[dict[str, object], str]],
) -> list[dict[str, object]]:
    """The Responses API requires the model's function_call echoed back paired
    with a function_call_output keyed by the same call_id. The assistant text is
    preserved; non-retrieve tool calls are re-planned by the follow-up."""
    items: list[dict[str, object]] = []
    text = _assistant_text_from_response(response)
    if text:
        items.append({"role": "assistant", "content": text})
    for tool_call, content in retrieved:
        call_id = tool_call.get("id")
        items.append(
            {
                "type": "function_call",
                "call_id": call_id,
                "name": tool_call.get("name"),
                "arguments": json.dumps(tool_call.get("arguments", {})),
            }
        )
        items.append({"type": "function_call_output", "call_id": call_id, "output": content})
    return items


@dataclass
class _CompressionResult:
    """Outcome of applying compression results to a message list."""

    compressed_messages: list[dict[str, object]]
    originals: dict[str, str] = field(default_factory=dict)
    # original text -> compressed text, plus the machinery the Responses `texts`
    # mirror needs to replace only where it is unambiguous.
    text_replacements: dict[str, str] = field(default_factory=dict)
    replaced_text_counts: dict[str, int] = field(default_factory=dict)
    ambiguous_texts: set[str] = field(default_factory=set)
    messages_compressed: int = 0
    tokens_before: int = 0
    tokens_after: int = 0


class CompresrGuardrail(CustomGuardrail):
    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        target_compression_ratio: float | None = None,
        coarse: bool | None = None,
        min_chars_to_compress: int | None = None,
        compress_tool_outputs: bool | None = None,
        compress_system: bool | None = None,
        compress_history: bool | None = None,
        compress_last_user: bool | None = None,
        enable_retrieval: bool | None = None,
        guardrail_name: str | None = None,
        event_hook: GuardrailEventHooks | list[GuardrailEventHooks] | Mode | None = None,
        default_on: bool = False,
        unreachable_fallback: str | None = None,
        max_bytes_per_call: int | None = None,
        allow_bypass_header: bool | None = None,
        dynamic: bool | None = None,
        dynamic_min_ratio: float | None = None,
        dynamic_max_ratio: float | None = None,
        compression_params: dict[str, object] | None = None,
    ):
        raw_api_base = (api_base or get_secret_str("COMPRESR_API_BASE") or DEFAULT_API_BASE).rstrip("/")
        self.compresr_api_base = _validate_api_base(raw_api_base)
        self.compresr_api_key = api_key or get_secret_str("COMPRESR_API_KEY")
        if not self.compresr_api_key:
            raise ValueError(
                "Compresr guardrail requires an API key. Set `api_key` in the "
                "guardrail config or the COMPRESR_API_KEY env var."
            )
        self.compression_model = model or DEFAULT_COMPRESSION_MODEL
        self.target_compression_ratio = (
            DEFAULT_TARGET_COMPRESSION_RATIO if target_compression_ratio is None else target_compression_ratio
        )
        self.coarse = True if coarse is None else coarse
        self.min_chars_to_compress = (
            DEFAULT_MIN_CHARS_TO_COMPRESS if min_chars_to_compress is None else min_chars_to_compress
        )
        self.compress_tool_outputs = True if compress_tool_outputs is None else compress_tool_outputs
        self.compress_system = False if compress_system is None else compress_system
        self.compress_history = False if compress_history is None else compress_history
        self.compress_last_user = False if compress_last_user is None else compress_last_user
        self.enable_retrieval = True if enable_retrieval is None else enable_retrieval
        self.unreachable_fallback: Literal["fail_closed", "fail_open"] = (
            "fail_open" if unreachable_fallback == "fail_open" else "fail_closed"
        )
        self.max_bytes_per_call = _DEFAULT_MAX_BYTES_PER_CALL if max_bytes_per_call is None else max_bytes_per_call
        if self.max_bytes_per_call < 0:
            raise ValueError("max_bytes_per_call must be >= 0 (0 disables the cap; positive values enforce it)")
        self.allow_bypass_header = False if allow_bypass_header is None else allow_bypass_header
        # Dynamic (adaptive) compression — latte_v2 only, on by default. When on,
        # the server picks the ratio per input (Kneedle elbow) instead of honoring
        # target_compression_ratio; None bounds let the server default apply.
        self.dynamic = True if dynamic is None else dynamic
        self.dynamic_min_ratio = dynamic_min_ratio
        self.dynamic_max_ratio = dynamic_max_ratio
        # Passthrough of extra compression params (e.g. heuristic_chunking, or a
        # newer knob) forwarded verbatim in the compress payload, so a new
        # Compresr feature works without changing this guardrail. Named fields
        # win on collision; request-content fields are stripped outright.
        reserved_keys = _RESERVED_COMPRESSION_PARAM_KEYS.intersection(compression_params or {})
        if reserved_keys:
            verbose_proxy_logger.warning(
                "Compresr: ignoring reserved compression_params keys %s", sorted(reserved_keys)
            )
        self.compression_params: dict[str, object] = {
            k: v for k, v in (compression_params or {}).items() if k not in _RESERVED_COMPRESSION_PARAM_KEYS
        }
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
        )
        self._originals_by_call_id: OrderedDict[str, tuple[dict[str, str], float]] = OrderedDict()
        # Running byte size of the store, kept in sync to enforce the global cap cheaply.
        self._store_total_bytes = 0
        if self.enable_retrieval:
            verbose_proxy_logger.warning(
                "Compresr: enable_retrieval is on; the recovery store is per-process. "
                "For multi-worker deployments, set enable_retrieval=false or run with --workers 1."
            )
        super().__init__(  # pyright: ignore[reportUnknownMemberType]  # CustomGuardrail.__init__ is untyped
            guardrail_name=guardrail_name,
            event_hook=event_hook,
            default_on=default_on,
        )

    def _should_bypass(self, request_data: dict) -> bool:
        if not self.allow_bypass_header:
            return False
        psr = request_data.get("proxy_server_request")
        if not _is_str_object_dict(psr):
            return False
        headers = psr.get("headers")
        if not _is_str_object_dict(headers):
            return False
        return str(headers.get(BYPASS_HEADER)).lower() == "true"

    def _request_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "X-API-Key": self.compresr_api_key or "",
        }

    def _handle_compress_failure(self, error: str, log_detail: dict[str, object]) -> None:
        """fail_open logs and returns (caller forwards uncompressed);
        fail_closed raises. ``log_detail`` may include upstream response bodies
        and is written only to server logs; the raised ``HTTPException`` carries
        a generic message so a malicious ``api_base`` cannot exfiltrate response
        bytes through the client-visible error."""
        if self.unreachable_fallback == "fail_open":
            verbose_proxy_logger.warning(
                "Compresr: %s; fail_open configured, forwarding request uncompressed. detail=%s",
                error,
                log_detail,
            )
            return
        verbose_proxy_logger.error("Compresr: %s. detail=%s", error, log_detail)
        raise HTTPException(status_code=502, detail={"error": error})

    def _evict_oldest(self) -> None:
        """Drop the front (oldest) entry and decrement the running byte total."""
        _key, (evicted, _expiry) = self._originals_by_call_id.popitem(last=False)
        self._store_total_bytes -= _entry_bytes(evicted)

    def _prune_originals(self) -> None:
        # Insertion order == expiry order (shared TTL); prune from the front.
        now = time.monotonic()
        store = self._originals_by_call_id
        while store and store[next(iter(store))][1] <= now:
            self._evict_oldest()
        while len(store) > _MAX_TRACKED_CALLS:
            self._evict_oldest()
        # Global byte budget; keep the most-recent entry so the current call's
        # originals survive (a single call is already bounded by max_bytes_per_call).
        while len(store) > 1 and self._store_total_bytes > _MAX_TOTAL_STORE_BYTES:
            self._evict_oldest()

    def _store_originals(self, store_key: str, originals: dict[str, str]) -> None:
        existing, _ = self._originals_by_call_id.get(store_key, ({}, 0.0))
        merged = self._bound_call_bytes({**existing, **originals})
        # Keep the running total in sync: drop the overwritten entry, add the new one.
        self._store_total_bytes += _entry_bytes(merged) - _entry_bytes(existing)
        self._originals_by_call_id[store_key] = (
            merged,
            time.monotonic() + _ORIGINALS_TTL_SECONDS,
        )
        self._originals_by_call_id.move_to_end(store_key)
        self._prune_originals()

    def _bound_call_bytes(self, merged: dict[str, str]) -> dict[str, str]:
        """Drop oldest entries (dict insertion order) until the aggregate byte
        size fits ``self.max_bytes_per_call``. Prevents one call with many
        large tool outputs from growing proxy memory without bound."""
        if self.max_bytes_per_call <= 0:
            return merged
        total = _entry_bytes(merged)
        if total <= self.max_bytes_per_call:
            return merged
        bounded = dict(merged)
        for key in list(bounded.keys()):
            if total <= self.max_bytes_per_call:
                break
            total -= len(bounded[key].encode("utf-8", "surrogatepass"))
            del bounded[key]
            verbose_proxy_logger.warning("Compresr: originals-store byte cap hit, evicted hash=%s", key)
        return bounded

    def _retrieve_original(self, store_key: str | None, hash_value: str) -> str | None:
        """Stored original for a marker hash, or None if not issued for this
        request (unknown, expired, or from another caller's scope)."""
        if store_key:
            originals, expiry = self._originals_by_call_id.get(store_key, ({}, 0.0))
            if expiry > time.monotonic() and hash_value in originals:
                return originals[hash_value]
            verbose_proxy_logger.warning(
                "Compresr retrieve: rejecting hash=%s (not issued for this request, or expired)",
                _display_hash(hash_value),
            )
        return None

    def _resolve_retrievals(
        self, store_key: str | None, tool_calls: list[dict[str, object]]
    ) -> tuple[list[tuple[dict[str, object], str]], bool]:
        """Resolve compresr_retrieve calls to (call, result_text) pairs, deduping
        repeated hashes and capping the count so the follow-up cannot be amplified.
        The bool is True iff at least one call resolved to real stored content."""
        retrieved: list[tuple[dict[str, object], str]] = []
        seen: set[str] = set()
        resolved_any = False
        for idx, tc in enumerate(tool_calls):
            arguments = tc.get("arguments", {})
            hash_value = str(arguments.get("hash", "")) if isinstance(arguments, dict) else ""
            if idx >= _MAX_RETRIEVALS_PER_LOOP:
                result = "[compresr: retrieval limit reached for this turn]"
            elif hash_value in seen:
                result = "[compresr: already retrieved above for this hash]"
            else:
                content = self._retrieve_original(store_key, hash_value)
                if content is None:
                    result = f"[compresr: hash={_display_hash(hash_value)} not found, expired, or not issued for this request]"
                else:
                    seen.add(hash_value)
                    resolved_any = True
                    result = content
            verbose_proxy_logger.debug("Compresr retrieve: hash=%s -> %d chars", _display_hash(hash_value), len(result))
            retrieved.append((tc, result))
        return retrieved, resolved_any

    async def _call_compress(
        self,
        contexts: list[str],
        queries: list[str],
    ) -> list[dict[str, object]] | None:
        """Compress ``contexts`` (query-aware). Returns one result dict per
        context, or None when the service failed and fail_open applies."""
        common: dict[str, object] = {
            # Passthrough first so the named fields below always win on collision.
            **self.compression_params,
            "compression_model_name": self.compression_model,
            "target_compression_ratio": self.target_compression_ratio,
            "coarse": self.coarse,
            "dynamic": self.dynamic,
            "source": _SOURCE_TAG,
        }
        # Only send the bounds the operator actually set; otherwise let the
        # server apply its own floor/ceiling.
        if self.dynamic_min_ratio is not None:
            common["dynamic_min_ratio"] = self.dynamic_min_ratio
        if self.dynamic_max_ratio is not None:
            common["dynamic_max_ratio"] = self.dynamic_max_ratio
        if len(contexts) == 1:
            url = f"{self.compresr_api_base}/api/compress/question-specific/"
            payload: dict[str, object] = {
                "context": contexts[0],
                "query": queries[0],
                **common,
            }
        else:
            url = f"{self.compresr_api_base}/api/compress/question-specific/batch"
            payload = {
                "inputs": [{"context": ctx, "query": q} for ctx, q in zip(contexts, queries)],
                **common,
            }

        try:
            raw_response: HttpxResponse | None = await self.async_handler.post(  # pyright: ignore[reportUnknownMemberType]  # AsyncHTTPHandler.post is untyped
                url=url,
                json=payload,
                headers=self._request_headers(),
                timeout=_COMPRESS_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            raise
        except httpx.HTTPStatusError as e:
            # The shared handler calls raise_for_status(), so a non-2xx reply
            # arrives here rather than as a returned Response — and the raised
            # error carries the upstream body and our request headers (API key).
            # Route it through the fail policy so none of that reaches the client.
            resp = getattr(e, "response", None)
            self._handle_compress_failure(
                "Compresr compression service returned an error",
                {
                    "status_code": getattr(resp, "status_code", None),
                    "body": _safe_response_text(resp),
                },
            )
            return None
        except (httpx.RequestError, litellm.Timeout) as e:
            # Every request-side httpx failure (connect, timeout, transport,
            # redirect loops, response-decoding) is a RequestError; route the
            # whole class through the fail policy so none escapes as a 500 when
            # fail_open is configured. HTTPStatusError is handled above and is
            # not a RequestError, so it is not swallowed here.
            self._handle_compress_failure(
                "Compresr compression service request failed",
                {"detail": str(e)},
            )
            return None
        if raw_response is None or not 200 <= raw_response.status_code < 300:
            self._handle_compress_failure(
                "Compresr compression service returned an error",
                {
                    "status_code": getattr(raw_response, "status_code", None),
                    "body": _safe_response_text(raw_response),
                },
            )
            return None

        try:
            body: object = raw_response.json()
        except (ValueError, httpx.DecodingError, RecursionError):
            # RecursionError: a deeply nested JSON body overflows the parser;
            # route it through the fail policy rather than let it escape as a 500.
            self._handle_compress_failure(
                "Compresr compression service returned an unreadable response",
                {"body": _safe_response_text(raw_response)},
            )
            return None
        if not _is_str_object_dict(body) or not _is_str_object_dict(body.get("data")):
            self._handle_compress_failure(
                "Compresr compression service returned unexpected response shape",
                {"body": _safe_response_text(raw_response)},
            )
            return None
        data: dict[str, object] = body["data"]  # pyright: ignore[reportAssignmentType]  # dict-guarded above; subscript does not narrow

        if len(contexts) == 1:
            return [data]
        results = data.get("results")
        if (
            not _is_object_list(results)
            or len(results) != len(contexts)
            or not all(_is_str_object_dict(r) for r in results)
        ):
            # Anything but a 1:1 dict-per-context mapping would misalign
            # results with their target messages.
            self._handle_compress_failure(
                "Compresr batch response missing or mismatched 'results'",
                {"expected": len(contexts), "got": len(results) if _is_object_list(results) else None},
            )
            return None
        return results  # pyright: ignore[reportReturnType]  # every element dict-checked above; list[object] does not narrow

    def _select_targets(self, messages: list[dict[str, object]], query_idx: int | None) -> list[int]:
        """Indices of messages whose text content should be compressed."""
        targets: list[int] = []
        for idx, msg in enumerate(messages):
            if idx == query_idx and not self.compress_last_user:
                continue
            role = msg.get("role")
            if role in ("tool", "function"):
                if not self.compress_tool_outputs:
                    continue
            elif role == "system":
                if not self.compress_system:
                    continue
            elif role == "user":
                if idx != query_idx and not self.compress_history:
                    continue
            else:
                continue
            if len(_content_to_text(msg.get("content"))) < self.min_chars_to_compress:
                continue
            targets.append(idx)
        return targets

    @staticmethod
    def _extract_fallback_query(
        messages: list[dict[str, object]],
    ) -> tuple[str, int | None]:
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "user":
                return _content_to_text(messages[idx].get("content")), idx
        return "", None

    def _apply_compression_results(
        self,
        messages: list[dict[str, object]],
        targets: list[int],
        contexts: list[str],
        results: list[dict[str, object]],
        recovery_enabled: bool,
    ) -> _CompressionResult:
        """Write each compression result into a copy of ``messages``.

        A result is a real compression only when it is a non-empty string that
        differs from the original; identical text is treated as a no-op so an
        untouched request is not needlessly rewritten downstream.
        """
        out = _CompressionResult(compressed_messages=list(messages))
        for target_idx, original_text, result in zip(targets, contexts, results):
            compressed_text = result.get("compressed_context")
            if not isinstance(compressed_text, str) or not compressed_text or compressed_text == original_text:
                continue
            out.messages_compressed += 1
            if recovery_enabled:
                hash_value = _content_hash(original_text)
                out.originals[hash_value] = original_text
                compressed_text += _recovery_marker(hash_value)
            previous = out.text_replacements.get(original_text)
            if previous is not None and previous != compressed_text:
                # Two targets with identical text but different query-specific
                # compressions; a value-keyed replacement cannot tell them apart.
                out.ambiguous_texts.add(original_text)
            else:
                out.text_replacements[original_text] = compressed_text
            out.replaced_text_counts[original_text] = out.replaced_text_counts.get(original_text, 0) + 1
            original_msg = out.compressed_messages[target_idx]
            out.compressed_messages[target_idx] = {
                **original_msg,
                "content": _replace_text_in_content(original_msg.get("content"), compressed_text),
            }
            out.tokens_before += _safe_int(result.get("original_tokens"))
            out.tokens_after += _safe_int(result.get("compressed_tokens"))
        return out

    @staticmethod
    def _mirror_texts_channel(input_texts: object, applied: _CompressionResult) -> list[object] | None:
        """Compressed content mirrored into the Responses `texts` channel.

        The chat/Anthropic handlers round-trip ``structured_messages``; the
        Responses translation cannot rebuild its input from chat messages and
        instead writes back through ``texts``. This matches by value, so a
        replacement is applied only when it is unambiguous: one compression per
        text, and every occurrence in ``texts`` accounted for by a compressed
        target. Anything else is left uncompressed rather than risk a wrong or
        out-of-policy replacement. Returns None when nothing safe applies.
        """
        if not applied.text_replacements or not isinstance(input_texts, list):
            return None
        counts = Counter(text for text in input_texts if isinstance(text, str))
        safe = {
            text: replacement
            for text, replacement in applied.text_replacements.items()
            if text not in applied.ambiguous_texts and counts.get(text) == applied.replaced_text_counts.get(text)
        }
        if not safe:
            return None
        return [safe.get(text, text) if isinstance(text, str) else text for text in input_texts]

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        if input_type != "request":
            return inputs

        if self._should_bypass(request_data):
            verbose_proxy_logger.debug("Compresr: %s header set; skipping compression", BYPASS_HEADER)
            return inputs

        structured_messages = inputs.get("structured_messages")
        if not _is_object_list(structured_messages) or not structured_messages:
            return inputs
        messages = [m for m in structured_messages if _is_str_object_dict(m)]
        if len(messages) != len(structured_messages):
            return inputs

        fallback_query, query_idx = self._extract_fallback_query(messages)
        targets: list[int] = []
        queries: list[str] = []
        for idx in self._select_targets(messages, query_idx):
            query = _query_for_target(messages, idx, fallback_query)
            # latte models require a non-empty query; leave targets we cannot
            # derive one for uncompressed rather than erroring.
            if not query.strip():
                continue
            targets.append(idx)
            queries.append(query)
        if not targets:
            verbose_proxy_logger.debug("Compresr: no messages eligible for compression")
            return inputs

        contexts = [_content_to_text(messages[idx].get("content")) for idx in targets]

        start_time = time.monotonic()
        results = await self._call_compress(contexts=contexts, queries=queries)
        end_time = time.monotonic()
        if results is None:  # service failed, fail_open configured
            return inputs

        # Recovery needs a per-tenant store key. Without a caller scope (proxy
        # runs without per-key auth) the store key would fall back to the
        # client-settable call id, letting one caller retrieve another's
        # originals by reusing the id; skip retrieval instead.
        store_key = _scoped_store_key(logging_obj)
        scope = _caller_scope(logging_obj)
        recovery_enabled = self.enable_retrieval and store_key is not None and bool(scope)

        applied = self._apply_compression_results(messages, targets, contexts, results, recovery_enabled)
        if applied.messages_compressed == 0:
            # Nothing was replaced. Return the original inputs object: handlers
            # detect guardrail edits by identity, and a fresh structured_messages
            # list would force a full write-back of an untouched request (on
            # Anthropic that reconversion strips cache_control from thinking
            # blocks).
            verbose_proxy_logger.debug("Compresr: service returned no compressed content; request unchanged")
            return inputs

        stats: dict[str, object] = {
            "messages_compressed": applied.messages_compressed,
            "tokens_before": applied.tokens_before,
            "tokens_after": applied.tokens_after,
            "tokens_saved": applied.tokens_before - applied.tokens_after,
            "compression_model": self.compression_model,
        }
        verbose_proxy_logger.debug(
            "Compresr: compressed %s message(s), %s -> %s tokens",
            applied.messages_compressed,
            applied.tokens_before,
            applied.tokens_after,
        )
        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_json_response=stats,
            request_data=request_data,
            guardrail_status="success",
            guardrail_provider="compresr",
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
        )

        compressed_inputs: dict[str, object] = {**inputs, "structured_messages": applied.compressed_messages}
        mirrored_texts = self._mirror_texts_channel(inputs.get("texts"), applied)
        if mirrored_texts is not None:
            compressed_inputs["texts"] = mirrored_texts

        originals = applied.originals
        if not recovery_enabled or not originals or store_key is None:
            return compressed_inputs  # pyright: ignore[reportReturnType]  # plain dicts satisfy AllMessageValues at runtime

        self._store_originals(store_key, originals)

        existing_tools = inputs.get("tools")
        retrieve_tool = _build_compresr_retrieve_tool()
        if isinstance(existing_tools, list) and not has_compresr_retrieve_tool(existing_tools):
            merged_tools: list[object] = list(existing_tools) + [retrieve_tool]
        elif existing_tools is None:
            merged_tools = [retrieve_tool]
        else:
            merged_tools = list(existing_tools) if isinstance(existing_tools, list) else [retrieve_tool]

        compressed_inputs["tools"] = merged_tools
        return compressed_inputs  # pyright: ignore[reportReturnType]  # plain dicts satisfy AllMessageValues at runtime

    async def async_should_run_agentic_loop(
        self,
        response: Any,
        model: str,
        messages: list[dict],
        tools: list[dict] | None,
        stream: bool,
        custom_llm_provider: str,
        kwargs: dict,
    ) -> tuple[bool, dict]:
        if not has_compresr_retrieve_tool(tools):
            return False, {}
        tool_calls = _extract_compresr_tool_calls(response)
        if not tool_calls:
            return False, {}
        return True, {"tool_calls": tool_calls}

    async def async_build_agentic_loop_plan(
        self,
        tools: dict,
        model: str,
        messages: list[dict],
        response: Any,
        anthropic_messages_provider_config: Any,
        anthropic_messages_optional_request_params: dict,
        logging_obj: Any,
        stream: bool,
        kwargs: dict,
    ) -> AgenticLoopPlan:
        tool_calls: list[dict[str, object]] = tools.get("tool_calls", [])  # pyright: ignore[reportAssignmentType]  # gate hook builds this dict with list values only

        self._prune_originals()
        store_key = _scoped_store_key(logging_obj)
        retrieved, resolved_any = self._resolve_retrievals(store_key, tool_calls)
        if not resolved_any:
            # Nothing this guardrail stored resolved; skip the extra provider round-trip.
            return AgenticLoopPlan(run_agentic_loop=False)

        if _is_responses_api_response(response):
            follow_up_messages = list(messages) + _build_responses_followup_items(response, retrieved)
        elif _is_anthropic_messages_response(response):
            follow_up_messages = list(messages) + _build_anthropic_followup_messages(response, retrieved)
        else:
            assistant_message = _build_assistant_message_from_response(response, retrieved)
            tool_results = [
                {"role": "tool", "tool_call_id": tc.get("id"), "content": content} for tc, content in retrieved
            ]
            follow_up_messages = list(messages) + [assistant_message] + tool_results

        anthropic_max = anthropic_messages_optional_request_params.get("max_tokens")
        max_tokens: int | None = anthropic_max if anthropic_max is not None else kwargs.get("max_tokens")
        optional_params_without_max_tokens = {
            k: v for k, v in anthropic_messages_optional_request_params.items() if k != "max_tokens"
        }

        full_model_name = model
        if logging_obj is not None:
            agentic_params = getattr(logging_obj, "model_call_details", {}).get("agentic_loop_params", {})
            candidate = agentic_params.get("model", model)
            if isinstance(candidate, str) and candidate:
                full_model_name = candidate

        return AgenticLoopPlan(
            run_agentic_loop=True,
            request_patch=AgenticLoopRequestPatch(
                model=full_model_name,
                messages=follow_up_messages,
                max_tokens=max_tokens,
                optional_params=optional_params_without_max_tokens,
                kwargs={
                    k: v for k, v in kwargs.items() if not k.startswith("_compresr") and k != "litellm_logging_obj"
                },
            ),
            metadata={"tool_type": "compresr_retrieve"},
        )

    @staticmethod
    def get_config_model() -> type[GuardrailConfigModel[object]] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.compresr import (
            CompresrGuardrailConfigModel,
        )

        return CompresrGuardrailConfigModel
