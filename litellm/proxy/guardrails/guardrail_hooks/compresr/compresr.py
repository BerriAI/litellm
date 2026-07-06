"""Compresr guardrail — query-aware context compression with lossless recovery.

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

# NOTE: originals are stored in process memory. Multi-worker deployments
# (gunicorn/uvicorn --workers N > 1) will silently lose originals when
# pre- and post-call hooks land on different workers. Set --workers 1
# or disable recovery (enable_retrieval=False) in multi-worker setups.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import socket
import time
import uuid
from typing import TYPE_CHECKING, Any, Literal, Optional
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
_HASH_PATTERN = re.compile(r"compresr hash=([a-f0-9]{24})")
_ORIGINALS_TTL_SECONDS = 15 * 60
_MAX_TRACKED_CALLS = 256
_DEFAULT_MAX_BYTES_PER_CALL = 10 * 1024 * 1024
_SOURCE_TAG = "gateway:unknown"
_BLOCKED_METADATA_HOSTS = frozenset(
    {
        "169.254.169.254",
        "fd00:ec2::254",
        "100.100.100.200",
        "168.63.129.16",  # Azure IMDS / wire-server
        "metadata.google.internal",
        "metadata.goog",
    }
)
_CGNAT_RANGE = ipaddress.ip_network("100.64.0.0/10")


def _validate_api_base(url: str) -> str:
    """Return ``url`` if it is a safe outbound target, else raise ``ValueError``.

    Blocks non-http(s) schemes and the well-known cloud-metadata IPs so that
    an attacker with config-write access cannot turn the guardrail into an
    SSRF probe against instance metadata. Private-range hosts are allowed
    because on-prem Compresr deployments legitimately live there.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Compresr guardrail api_base must be http or https, got scheme={parsed.scheme!r}"
        )
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        raise ValueError("Compresr guardrail api_base has no host")
    if host in _BLOCKED_METADATA_HOSTS:
        raise ValueError(f"Compresr guardrail api_base {host!r} is a blocked cloud-metadata host")
    try:
        for info in socket.getaddrinfo(host, None):
            addr = info[4][0]
            ip = ipaddress.ip_address(addr)
            if addr in _BLOCKED_METADATA_HOSTS or ip.is_link_local or ip.is_loopback or ip.is_unspecified:
                raise ValueError(
                    f"Compresr guardrail api_base {host!r} resolves to blocked address {addr!r}"
                )
            if ip in _CGNAT_RANGE:
                raise ValueError(
                    f"Compresr guardrail api_base {host!r} resolves to CGNAT address {addr!r} (100.64.0.0/10)"
                )
    except socket.gaierror:
        verbose_proxy_logger.warning(
            "compresr: DNS resolution failed for %s at startup — SSRF validation skipped. "
            "Ensure api_base is reachable before handling live traffic.", host
        )
        return url
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
        fc = prev.get("function_call")
        if isinstance(fc, dict) and (not fn_name or fc.get("name") == fn_name):
            intent = _render_tool_intent(fc)
            if intent:
                return intent
    return fallback


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _recovery_marker(hash_value: str) -> str:
    return (
        f"\n\n[compresr hash={hash_value}: parts of this content were compressed "
        f"away. If you need the full original, call the "
        f"{COMPRESR_RETRIEVE_TOOL_NAME} tool with this hash.]"
    )


def extract_hashes_from_messages(messages: list[dict[str, object]]) -> list[str]:
    hashes: list[str] = []
    for msg in messages:
        text = _content_to_text(msg.get("content"))
        hashes.extend(_HASH_PATTERN.findall(text))
    return hashes


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


def _resolve_call_id(logging_obj: object) -> Optional[str]:
    """Framework-issued call id only; user-supplied ids from the request body
    are ignored so one tenant cannot retrieve another tenant's originals by
    guessing or observing a call id."""
    logging_call_id = getattr(logging_obj, "litellm_call_id", None)
    if isinstance(logging_call_id, str) and logging_call_id:
        return logging_call_id
    return None


def _is_responses_api_response(response: object) -> bool:
    return isinstance(get_attribute_or_key(response, "output", None), list)


def _is_anthropic_messages_response(response: object) -> bool:
    return isinstance(get_attribute_or_key(response, "content", None), list)


def _build_assistant_message_from_response(response: object) -> dict[str, object]:
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        return {"role": "assistant", "content": None, "tool_calls": []}
    message = getattr(choices[0], "message", None)
    if message is None:
        return {"role": "assistant", "content": None, "tool_calls": []}
    content = getattr(message, "content", None)
    tool_calls = getattr(message, "tool_calls", None)
    raw_tool_calls: list[dict[str, object]] = []
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            fn = getattr(tc, "function", None)
            raw_tool_calls.append(
                {
                    "id": getattr(tc, "id", None),
                    "type": "function",
                    "function": {
                        "name": getattr(fn, "name", None) if fn else None,
                        "arguments": getattr(fn, "arguments", "{}") if fn else "{}",
                    },
                }
            )
    return {"role": "assistant", "content": content, "tool_calls": raw_tool_calls}


def _build_anthropic_followup_messages(
    retrieved: list[tuple[dict[str, object], str]],
) -> list[dict[str, object]]:
    """Anthropic requires the tool_use block echoed back in an assistant
    message paired with a tool_result block keyed by the same tool_use_id."""
    assistant_message: dict[str, object] = {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": tool_call.get("id"),
                "name": tool_call.get("name"),
                "input": tool_call.get("arguments", {}),
            }
            for tool_call, _ in retrieved
        ],
    }
    user_message: dict[str, object] = {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": tool_call.get("id"), "content": content}
            for tool_call, content in retrieved
        ],
    }
    return [assistant_message, user_message]


def _build_responses_followup_items(
    retrieved: list[tuple[dict[str, object], str]],
) -> list[dict[str, object]]:
    """The Responses API requires the model's function_call echoed back paired
    with a function_call_output keyed by the same call_id."""
    items: list[dict[str, object]] = []
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
        self.max_bytes_per_call = (
            _DEFAULT_MAX_BYTES_PER_CALL if max_bytes_per_call is None else max_bytes_per_call
        )
        self.allow_bypass_header = False if allow_bypass_header is None else allow_bypass_header
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
        )
        self._originals_by_call_id: dict[str, tuple[dict[str, str], float]] = {}
        super().__init__(  # pyright: ignore[reportUnknownMemberType]  # CustomGuardrail.__init__ is untyped
            guardrail_name=guardrail_name,
            event_hook=event_hook,
            default_on=default_on,
        )

    # ── request plumbing ──────────────────────────────────────────────

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
            verbose_proxy_logger.critical(
                "Compresr: %s; fail_open configured, forwarding request uncompressed. detail=%s",
                error,
                log_detail,
            )
            return
        verbose_proxy_logger.error("Compresr: %s. detail=%s", error, log_detail)
        raise HTTPException(status_code=502, detail={"error": error})

    # ── originals store (recovery) ────────────────────────────────────

    def _prune_originals(self) -> None:
        now = time.monotonic()
        alive = {
            call_id: (originals, expiry)
            for call_id, (originals, expiry) in self._originals_by_call_id.items()
            if expiry > now
        }
        if len(alive) > _MAX_TRACKED_CALLS:
            # Evict soonest-to-expire first — a hard cap so a burst of huge
            # tool outputs cannot grow proxy memory unbounded.
            by_expiry = sorted(alive.items(), key=lambda item: item[1][1])
            alive = dict(by_expiry[len(alive) - _MAX_TRACKED_CALLS :])
        self._originals_by_call_id = alive

    def _store_originals(self, call_id: str, originals: dict[str, str]) -> None:
        existing, _ = self._originals_by_call_id.get(call_id, ({}, 0.0))
        merged = self._bound_call_bytes({**existing, **originals})
        self._originals_by_call_id[call_id] = (
            merged,
            time.monotonic() + _ORIGINALS_TTL_SECONDS,
        )
        # Prune after inserting so the cap holds; the entry just added has the
        # latest expiry and always survives eviction.
        self._prune_originals()

    def _bound_call_bytes(self, merged: dict[str, str]) -> dict[str, str]:
        """Drop oldest entries (dict insertion order) until the aggregate byte
        size fits ``self.max_bytes_per_call``. Prevents one call with many
        large tool outputs from growing proxy memory without bound."""
        if self.max_bytes_per_call <= 0:
            return merged
        total = sum(len(value.encode("utf-8")) for value in merged.values())
        if total <= self.max_bytes_per_call:
            return merged
        bounded = dict(merged)
        for key in list(bounded.keys()):
            if total <= self.max_bytes_per_call:
                break
            total -= len(bounded[key].encode("utf-8"))
            del bounded[key]
            verbose_proxy_logger.warning(
                "Compresr: originals-store byte cap hit, evicted hash=%s", key
            )
        return bounded

    def _retrieve_original(self, call_id: Optional[str], hash_value: str) -> str:
        if call_id:
            originals, expiry = self._originals_by_call_id.get(call_id, ({}, 0.0))
            if expiry > time.monotonic() and hash_value in originals:
                return originals[hash_value]
            # A hash is only honored if it was issued by *this request's own*
            # compression, scoped by litellm_call_id. Honoring any hash-shaped
            # string in message text would be forgeable across requests.
            verbose_proxy_logger.warning(
                "Compresr retrieve: rejecting hash=%s (not issued for this request, or expired)",
                hash_value,
            )
        return f"[compresr: hash={hash_value} not found, expired, or not issued for this request]"

    # ── Compresr API (dependency-free) ────────────────────────────────

    async def _call_compress(
        self,
        contexts: list[str],
        queries: list[str],
    ) -> Optional[list[dict[str, object]]]:
        """Compress ``contexts`` (query-aware). Returns one result dict per
        context, or None when the service failed and fail_open applies."""
        try:
            _validate_api_base(self.compresr_api_base)
        except ValueError as exc:
            self._handle_compress_failure(
                "Compresr api_base failed SSRF re-validation at request time",
                {"detail": str(exc)},
            )
            return None

        common: dict[str, object] = {
            "compression_model_name": self.compression_model,
            "target_compression_ratio": self.target_compression_ratio,
            "coarse": self.coarse,
            "source": _SOURCE_TAG,
        }
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
            )
        except (httpx.ConnectError, httpx.TimeoutException, httpx.TransportError, litellm.Timeout) as e:
            self._handle_compress_failure(
                "Compresr compression service unreachable",
                {"detail": str(e)},
            )
            return None
        if raw_response is None or raw_response.status_code != 200:
            self._handle_compress_failure(
                "Compresr compression service returned an error",
                {
                    "status_code": getattr(raw_response, "status_code", None),
                    "body": getattr(raw_response, "text", "")[:500],
                },
            )
            return None

        try:
            body: object = raw_response.json()
        except ValueError:
            self._handle_compress_failure(
                "Compresr compression service returned non-JSON response",
                {"body": raw_response.text[:500]},
            )
            return None
        if not _is_str_object_dict(body) or not _is_str_object_dict(body.get("data")):
            self._handle_compress_failure(
                "Compresr compression service returned unexpected response shape",
                {"body": raw_response.text[:500]},
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

    # ── target selection ──────────────────────────────────────────────

    def _select_targets(self, messages: list[dict[str, object]], query_idx: Optional[int]) -> list[int]:
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
    ) -> tuple[str, Optional[int]]:
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "user":
                return _content_to_text(messages[idx].get("content")), idx
        return "", None

    # ── the guardrail hook ────────────────────────────────────────────

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
        targets = [
            idx
            for idx in self._select_targets(messages, query_idx)
            # latte models require a non-empty query; leave targets we cannot
            # derive one for uncompressed rather than erroring.
            if _query_for_target(messages, idx, fallback_query).strip()
        ]
        if not targets:
            verbose_proxy_logger.debug("Compresr: no messages eligible for compression")
            return inputs

        contexts = [_content_to_text(messages[idx].get("content")) for idx in targets]
        queries = [_query_for_target(messages, idx, fallback_query) for idx in targets]

        start_time = time.time()
        results = await self._call_compress(contexts=contexts, queries=queries)
        end_time = time.time()
        if results is None:  # service failed, fail_open configured
            return inputs

        compressed_messages: list[dict[str, object]] = list(messages)
        originals: dict[str, str] = {}
        messages_compressed = 0
        tokens_before = 0
        tokens_after = 0
        for target_idx, original_text, result in zip(targets, contexts, results):
            compressed_text = result.get("compressed_context")
            if not isinstance(compressed_text, str) or not compressed_text.strip():
                continue
            if len(compressed_text) >= len(original_text):
                continue  # compression made it worse — keep original
            messages_compressed += 1
            if self.enable_retrieval:
                hash_value = _content_hash(original_text)
                originals[hash_value] = original_text
                compressed_text += _recovery_marker(hash_value)
            original_msg = compressed_messages[target_idx]
            compressed_messages[target_idx] = {
                **original_msg,
                "content": _replace_text_in_content(original_msg.get("content"), compressed_text),
            }
            tokens_before += int(result.get("original_tokens") or 0)
            tokens_after += int(result.get("compressed_tokens") or 0)

        stats: dict[str, object] = {
            "messages_compressed": messages_compressed,
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "tokens_saved": tokens_before - tokens_after,
            "compression_model": self.compression_model,
        }
        verbose_proxy_logger.debug(
            "Compresr: compressed %s message(s), %s -> %s tokens",
            stats["messages_compressed"],
            tokens_before,
            tokens_after,
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

        if not self.enable_retrieval or not originals:
            return {**inputs, "structured_messages": compressed_messages}  # pyright: ignore[reportReturnType]  # plain dicts satisfy AllMessageValues at runtime

        call_id = _resolve_call_id(logging_obj) or str(uuid.uuid4())
        self._store_originals(call_id, originals)

        existing_tools = inputs.get("tools")
        retrieve_tool = _build_compresr_retrieve_tool()
        if isinstance(existing_tools, list) and not has_compresr_retrieve_tool(existing_tools):
            merged_tools: list[object] = list(existing_tools) + [retrieve_tool]
        elif existing_tools is None:
            merged_tools = [retrieve_tool]
        else:
            merged_tools = list(existing_tools) if isinstance(existing_tools, list) else [retrieve_tool]

        return {  # pyright: ignore[reportReturnType]  # plain dicts satisfy AllMessageValues at runtime
            **inputs,
            "structured_messages": compressed_messages,
            "tools": merged_tools,
        }

    # ── agentic loop (recovery round-trip) ────────────────────────────

    async def async_should_run_agentic_loop(
        self,
        response: Any,
        model: str,
        messages: list[dict],
        tools: Optional[list[dict]],
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

        call_id = _resolve_call_id(logging_obj)
        retrieved: list[tuple[dict[str, object], str]] = []
        for tc in tool_calls:
            arguments = tc.get("arguments", {})
            hash_value = str(arguments.get("hash", "")) if isinstance(arguments, dict) else ""
            content = self._retrieve_original(call_id, hash_value)
            verbose_proxy_logger.debug("Compresr retrieve: hash=%s -> %d chars", hash_value, len(content))
            retrieved.append((tc, content))

        if _is_responses_api_response(response):
            follow_up_messages = list(messages) + _build_responses_followup_items(retrieved)
        elif _is_anthropic_messages_response(response):
            follow_up_messages = list(messages) + _build_anthropic_followup_messages(retrieved)
        else:
            assistant_message = _build_assistant_message_from_response(response)
            tool_results = [
                {"role": "tool", "tool_call_id": tc.get("id"), "content": content} for tc, content in retrieved
            ]
            follow_up_messages = list(messages) + [assistant_message] + tool_results

        max_tokens: Optional[int] = anthropic_messages_optional_request_params.get("max_tokens") or kwargs.get(
            "max_tokens"
        )
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
