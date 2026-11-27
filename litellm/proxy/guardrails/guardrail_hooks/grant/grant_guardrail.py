"""Grant post-call guardrail for LiteLLM.

Upstream target (PR package):
  litellm/proxy/guardrails/guardrail_hooks/grant/grant_guardrail.py

Zero new dependencies: this module uses ``httpx``, which LiteLLM already ships,
plus the Python standard library. Zero custody: only tool names, a short summary,
and redacted scope metadata (argument key count plus a keyed HMAC digest) are
sent to Grant. Credential values, payment details, raw arguments, raw PII, and
raw argument keys are never sent.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any, AsyncIterable, AsyncIterator, Iterable

import httpx

try:  # pragma: no cover - the fallback only applies outside a LiteLLM install
    from litellm.integrations.custom_guardrail import CustomGuardrail
except ImportError:  # pragma: no cover
    class CustomGuardrail:  # minimal local shim so upstream tests run anywhere
        def __init__(self, guardrail_name: str | None = None, **kwargs: Any) -> None:
            self.guardrail_name = guardrail_name
            self.event_hook = kwargs.get("event_hook")


DEFAULT_GRANT_BASE_URL = "https://grant-approve.vercel.app"
ALLOWING_DECISIONS = frozenset({"approve", "always", "auto_rule", "approved"})
BLOCKING = "blocking"
CHECKPOINT = "checkpoint"
STREAMING_REJECTION_MESSAGE = (
    "GrantGuardrail does not permit unbuffered stream=True; "
    "set stream=False or disable streaming for protected tool calls"
)
DEFAULT_PROTECTED_TOOLS = frozenset(
    {
        "send_email",
        "send_message",
        "issue_refund",
        "create_charge",
        "transfer_funds",
        "publish_content",
        "write_record",
        "delete_record",
        "execute_payment",
    }
)
_EPHEMERAL_HMAC_SECRET = secrets.token_bytes(32)


def resolve_grant_base_url(explicit: str | None = None) -> str:
    """Explicit setting → env vars → documented hosted default.

    Deliberately does not read ``.vercel/project.json``: a file in the proxy's
    working directory must never become the trust anchor for approval requests
    or redirect the Grant API key to an unintended host.
    """
    candidate = (
        explicit
        or os.environ.get("GRANT_BASE_URL")
        or os.environ.get("GRANT_PUBLIC_BASE_URL")
        or ""
    ).strip()
    return candidate.rstrip("/") if candidate else DEFAULT_GRANT_BASE_URL


def _as_dict(value: Any) -> dict[str, Any]:
    """Best-effort conversion of pydantic models and LiteLLM objects to dicts."""
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(mode="json")
        except TypeError:  # pragma: no cover - older pydantic
            dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    try:
        return {key: val for key, val in vars(value).items() if not str(key).startswith("_")}
    except TypeError:
        return {}


def _tool_function(call: Any) -> tuple[str, Any, str]:
    """Normalize OpenAI tool calls, legacy function_call, and Anthropic tool_use."""
    data = _as_dict(call)
    function = _as_dict(data.get("function"))
    name = str(function.get("name") or data.get("name") or data.get("tool_name") or "")
    arguments = function.get("arguments")
    if arguments is None:
        arguments = data.get("arguments", data.get("input", {}))
    call_id = str(data.get("id") or data.get("call_id") or data.get("tool_call_id") or "")
    return name, arguments if arguments is not None else {}, call_id


def _normalized_call(call: Any) -> dict[str, Any]:
    name, arguments, call_id = _tool_function(call)
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _iter_tool_calls(response: Any) -> Iterable[Any]:
    """Yield normalized protected-call candidates from every supported format."""
    data = _as_dict(response)
    if not data:
        return

    choices = data.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            choice_data = _as_dict(choice)
            message = _as_dict(choice_data.get("message") or choice_data.get("delta") or {})
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    yield _normalized_call(call)
            # Legacy OpenAI functions/function_call response format.
            function_call = message.get("function_call")
            if function_call:
                yield _normalized_call(
                    {"id": message.get("tool_call_id") or "", "function": function_call}
                )

    top_level_calls = data.get("tool_calls")
    if isinstance(top_level_calls, list):
        for call in top_level_calls:
            yield _normalized_call(call)

    # Anthropic /v1/messages top-level content blocks.
    for container in (data, _as_dict(data.get("message"))):
        content = container.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            block_data = _as_dict(block)
            if str(block_data.get("type") or "").casefold() != "tool_use":
                continue
            arguments = block_data.get("input", block_data.get("arguments", {}))
            yield _normalized_call(
                {
                    "id": block_data.get("id") or "",
                    "name": block_data.get("name") or "",
                    "arguments": arguments,
                }
            )

    # Responses API function_call output items.
    output = data.get("output")
    if isinstance(output, list):
        for item in output:
            item_data = _as_dict(item)
            if str(item_data.get("type") or "").casefold() not in {"function_call", "tool_use"}:
                continue
            arguments = item_data.get("arguments", item_data.get("input", {}))
            yield _normalized_call(
                {
                    "id": item_data.get("call_id") or item_data.get("id") or "",
                    "name": item_data.get("name") or "",
                    "arguments": arguments,
                }
            )


def _iter_sse_events(payload: Any) -> Iterable[dict[str, Any]]:
    """Decode raw SSE chunks (Anthropic native streaming) into JSON events."""
    if isinstance(payload, (bytes, bytearray)):
        text = bytes(payload).decode("utf-8", errors="replace")
    elif isinstance(payload, str):
        text = payload
    else:
        return
    for raw_event in text.replace("\r\n", "\n").split("\n\n"):
        data_lines = [line[5:].strip() for line in raw_event.splitlines() if line.startswith("data:")]
        if not data_lines:
            continue
        joined = "\n".join(data_lines)
        if joined == "[DONE]":
            continue
        try:
            event = json.loads(joined)
        except ValueError:
            continue
        if isinstance(event, dict):
            yield event


def _append_arguments(current: Any, addition: Any) -> Any:
    """Accumulate streamed JSON-argument fragments without interpreting values."""
    if addition is None:
        return current
    if isinstance(addition, str):
        return (current if isinstance(current, str) else "") + addition
    if isinstance(current, dict) and isinstance(addition, dict):
        return {**current, **addition}
    return addition


def _iter_tool_calls_from_chunks(chunks: Iterable[Any]) -> Iterable[Any]:
    """Rebuild tool calls from buffered stream chunks before any chunk is yielded."""
    fragments: dict[str, dict[str, Any]] = {}
    legacy: dict[str, dict[str, Any]] = {}
    complete: list[Any] = []
    anthropic: dict[str, dict[str, Any]] = {}

    def consume(data: dict[str, Any]) -> None:
        choices = data.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                choice_data = _as_dict(choice)
                delta = _as_dict(choice_data.get("delta"))
                streamed_calls = delta.get("tool_calls")
                if isinstance(streamed_calls, list):
                    for call in streamed_calls:
                        call_data = _as_dict(call)
                        index = str(call_data.get("index") or "0")
                        entry = fragments.setdefault(index, {"id": "", "name": "", "arguments": ""})
                        if call_data.get("id"):
                            entry["id"] = str(call_data["id"])
                        function = _as_dict(call_data.get("function"))
                        if function.get("name"):
                            entry["name"] += str(function["name"])
                        entry["arguments"] = _append_arguments(entry["arguments"], function.get("arguments"))
                function_call = delta.get("function_call")
                if function_call:
                    call_data = _as_dict(function_call)
                    entry = legacy.setdefault("legacy", {"id": "", "name": "", "arguments": ""})
                    if call_data.get("name"):
                        entry["name"] += str(call_data["name"])
                    entry["arguments"] = _append_arguments(entry["arguments"], call_data.get("arguments"))
                message = _as_dict(choice_data.get("message"))
                if isinstance(message.get("tool_calls"), list):
                    complete.extend(message["tool_calls"])
                if message.get("function_call"):
                    complete.append(
                        {"id": message.get("tool_call_id") or "", "function": message["function_call"]}
                    )

        for container in (data, _as_dict(data.get("message"))):
            content = container.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                block_data = _as_dict(block)
                if str(block_data.get("type") or "").casefold() != "tool_use":
                    continue
                key = str(block_data.get("id") or block_data.get("name") or len(anthropic))
                entry = anthropic.setdefault(
                    key,
                    {
                        "id": str(block_data.get("id") or ""),
                        "name": str(block_data.get("name") or ""),
                        "arguments": block_data.get("input", block_data.get("arguments", {})),
                    },
                )
                if block_data.get("name"):
                    entry["name"] = str(block_data["name"])
                if "input" in block_data:
                    entry["arguments"] = _append_arguments(entry["arguments"], block_data["input"])

        event_type = str(data.get("type") or "")
        if event_type == "content_block_start":
            block_data = _as_dict(data.get("content_block"))
            if str(block_data.get("type") or "").casefold() == "tool_use":
                index = str(data.get("index") or block_data.get("id") or "0")
                anthropic[index] = {
                    "id": str(block_data.get("id") or ""),
                    "name": str(block_data.get("name") or ""),
                    "arguments": block_data.get("input", {}),
                }
        elif event_type == "content_block_delta":
            delta = _as_dict(data.get("delta"))
            if str(delta.get("type") or "").casefold() == "input_json_delta":
                index = str(data.get("index") or "0")
                entry = anthropic.setdefault(index, {"id": "", "name": "", "arguments": ""})
                if delta.get("partial_json") is not None:
                    entry["arguments"] = _append_arguments(entry["arguments"], delta["partial_json"])

    # Raw SSE frames can be split across chunks; concatenate before decoding so
    # buffered enforcement reconstructs the same tool call the client will see.
    raw_parts: list[str] = []
    for chunk in chunks:
        if isinstance(chunk, (bytes, bytearray)):
            raw_parts.append(bytes(chunk).decode("utf-8", errors="replace"))
        elif isinstance(chunk, str):
            raw_parts.append(chunk)
        else:
            consume(_as_dict(chunk))
    if raw_parts:
        for event in _iter_sse_events("".join(raw_parts)):
            consume(event)

    for call in complete:
        yield _normalized_call(call)
    for entry in fragments.values():
        if entry["name"]:
            yield _normalized_call(
                {"id": entry["id"], "function": {"name": entry["name"], "arguments": entry["arguments"]}}
            )
    for entry in legacy.values():
        if entry["name"]:
            yield _normalized_call(
                {"id": entry["id"], "function": {"name": entry["name"], "arguments": entry["arguments"]}}
            )
    for entry in anthropic.values():
        if entry["name"]:
            yield _normalized_call(
                {"id": entry["id"], "name": entry["name"], "arguments": entry["arguments"]}
            )


def _local_hmac_key() -> bytes:
    secret = os.environ.get("GRANT_LOCAL_HMAC_SECRET") or os.environ.get("GRANT_API_KEY") or ""
    return secret.encode("utf-8") if secret else _EPHEMERAL_HMAC_SECRET


def _redacted_scope(arguments: Any, call_id: str) -> dict[str, Any]:
    """Derive non-sensitive metadata only; values and key names never leave."""
    parsed: Any = arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except (TypeError, ValueError):
            parsed = {"unparsed": True}
    if not isinstance(parsed, dict):
        parsed = {"value": parsed}
    try:
        canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        canonical = repr(sorted(str(key) for key in parsed))
    return {
        "tool_call_id": call_id[:120],
        "argument_key_count": len(parsed),
        "arguments_hmac_sha256": hmac.new(
            _local_hmac_key(), canonical.encode("utf-8"), hashlib.sha256
        ).hexdigest(),
    }


class GrantApprovalRequired(Exception):
    """Checkpoint-mode signal carrying the structured Grant popup payload."""

    def __init__(self, *, request_id: str, approval_url: str, receipt_url: str = "", expires_at: str = "", signed_receipt: str = "") -> None:
        self.metadata = {
            "grant_approval_required": True,
            "request_id": request_id,
            "approval_url": approval_url,
            "receipt_url": receipt_url,
            "expires_at": expires_at,
            "signed_receipt": signed_receipt,
            "message": "Open the Grant checkpoint, approve, then retry the tool call.",
        }
        super().__init__(json.dumps(self.metadata, sort_keys=True))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.metadata)


class GrantDenied(Exception):
    """The human rejected, cancelled, or the receipt could not be verified."""


class GrantApprovalTimeout(Exception):
    """Blocking mode ran out of time before a human decided."""


class GrantGuardrail(CustomGuardrail):
    """Gate returned tool calls through a Grant human checkpoint.

    ``approval_mode="checkpoint"`` raises :class:`GrantApprovalRequired` with the
    approval URL so the caller can surface the Grant popup immediately.
    ``approval_mode="blocking"`` polls ``/v1/approvals/{request_id}`` until a
    human decides. An already-approved call adds one HTTP round trip only.

    Streaming is rejected in ``async_pre_call_hook`` unless ``allow_streaming``
    is explicitly enabled. When enabled, the streaming iterator buffers the
    whole response, gates reconstructed tool calls, and only then yields chunks.
    """

    use_native_lifecycle_hooks = True

    def __init__(
        self,
        guardrail_name: str = "grant-approval",
        *,
        protected_tools: Iterable[str] | None = None,
        grant_base_url: str | None = None,
        api_key: str | None = None,
        workspace: str = "public",
        approval_mode: str = CHECKPOINT,
        timeout: float = 2.0,
        poll_interval: float = 1.0,
        poll_timeout: float = 300.0,
        fail_open: bool = False,
        allow_streaming: bool = False,
        client: httpx.AsyncClient | None = None,
        **kwargs: Any,
    ) -> None:
        # Both hooks are required: pre_call rejects unbuffered streaming and
        # post_call gates the normalized tool calls in the response.
        kwargs["event_hook"] = ["pre_call", "post_call"]
        super().__init__(guardrail_name=guardrail_name, **kwargs)
        if approval_mode not in {BLOCKING, CHECKPOINT}:
            raise ValueError("approval_mode must be 'blocking' or 'checkpoint'")
        self.protected_tools = frozenset(
            str(tool).strip() for tool in (protected_tools if protected_tools is not None else DEFAULT_PROTECTED_TOOLS)
        )
        self.base_url = resolve_grant_base_url(grant_base_url)
        self.api_key = api_key or os.environ.get("GRANT_API_KEY") or ""
        self.workspace = workspace[:60] or "public"
        self.approval_mode = approval_mode
        self.timeout = timeout
        self.poll_interval = max(0.05, poll_interval)
        self.poll_timeout = max(1.0, poll_timeout)
        self.fail_open = fail_open
        self.allow_streaming = bool(allow_streaming)
        self.last_latency_ms: float | None = None
        self._client = client
        self._owns_client = client is None

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any = None,
        cache: Any = None,
        data: dict[str, Any] | None = None,
        call_type: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Reject unbuffered streaming before any model call is made."""
        request_data = kwargs.pop("data", data)
        if request_data is None:
            request_data = kwargs.pop("request_data", {}) or {}
        if isinstance(request_data, dict) and request_data.get("stream") is True and not self.allow_streaming:
            raise ValueError(STREAMING_REJECTION_MESSAGE)
        return request_data if isinstance(request_data, dict) else {}

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=0.5))
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _matches(self, tool_name: str) -> bool:
        if "*" in self.protected_tools:
            return True
        needle = tool_name.strip().casefold()
        return any(needle == tool.casefold() for tool in self.protected_tools)

    async def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        client = self._get_client()
        response = await client.request(
            method,
            f"{self.base_url}{path}",
            headers=self._headers(),
            json=payload if payload is not None else None,
        )
        try:
            body = response.json()
        except ValueError:
            body = {}
        return response.status_code, body if isinstance(body, dict) else {}

    async def _post_checkpoint(self, tool_name: str, arguments: Any, call_id: str) -> dict[str, Any]:
        try:
            status, body = await self._request(
                "POST",
                "/v1/approvals",
                {
                    "agent_name": "litellm-guardrail",
                    "tool_name": tool_name[:120],
                    "summary": f"Allow the assistant to call {tool_name[:100]}?",
                    "scope": _redacted_scope(arguments, call_id),
                    "workspace": self.workspace,
                    "channel": "litellm_guardrail",
                },
            )
        except httpx.HTTPError:
            if self.fail_open:
                return {"status": "approved", "signed_receipt": "", "failure": "grant_unavailable"}
            return {"status": "denied", "signed_receipt": "", "failure": "grant_unavailable"}
        if status >= 400:
            if self.fail_open:
                return {"status": "approved", "signed_receipt": "", "failure": f"http_{status}"}
            return {"status": "denied", "signed_receipt": "", "failure": f"http_{status}"}
        return body

    async def _poll(self, request_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.poll_timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(self.poll_interval)
            try:
                status, body = await self._request("GET", f"/v1/approvals/{request_id}")
            except httpx.HTTPError:
                continue
            if status >= 400:
                continue
            decision = str(body.get("decision") or "").strip().casefold()
            state = str(body.get("status") or "").strip().casefold()
            if decision in ALLOWING_DECISIONS or state == "approved":
                return body
            if decision in {"reject", "cancelled", "cancel", "denied"} or state in {"cancelled", "expired"}:
                return {**body, "status": "denied"}
        return {"status": "timeout", "request_id": request_id}

    @staticmethod
    def _approved(checkpoint: dict[str, Any]) -> bool:
        status = str(checkpoint.get("status") or "").strip().casefold()
        decision = str(checkpoint.get("decision_type") or checkpoint.get("decision") or "").strip().casefold()
        return status == "approved" or decision in {"continue_once", "always_allow", *ALLOWING_DECISIONS}

    async def _gate_calls(self, calls: Iterable[Any]) -> None:
        """Apply the identical Grant gate to calls from any response format."""
        started = time.perf_counter()
        try:
            for call in calls:
                tool_name, arguments, call_id = _tool_function(call)
                if not tool_name or not self._matches(tool_name):
                    continue
                checkpoint = await self._post_checkpoint(tool_name, arguments, call_id)
                if self._approved(checkpoint):
                    continue
                request_id = str(checkpoint.get("request_id") or checkpoint.get("id") or "")
                if str(checkpoint.get("status") or "").casefold() == "pending" or checkpoint.get("approval_url"):
                    if self.approval_mode == CHECKPOINT:
                        raise GrantApprovalRequired(
                            request_id=request_id,
                            approval_url=str(checkpoint.get("approval_url") or ""),
                            receipt_url=str(checkpoint.get("receipt_url") or ""),
                            expires_at=str(checkpoint.get("expires_at") or ""),
                            signed_receipt=str(checkpoint.get("signed_receipt") or ""),
                        )
                    checkpoint = await self._poll(request_id)
                    if self._approved(checkpoint):
                        continue
                    if str(checkpoint.get("status") or "").casefold() == "timeout":
                        raise GrantApprovalTimeout(f"Grant approval timed out for {request_id}")
                    raise GrantDenied(f"Grant did not approve {tool_name} ({request_id})")
                failure = str(checkpoint.get("failure") or checkpoint.get("status") or "unknown")
                raise GrantDenied(f"Grant blocked {tool_name}: {failure}")
        finally:
            self.last_latency_ms = (time.perf_counter() - started) * 1000

    async def async_post_call_success_hook(
        self,
        data: dict[str, Any] | None = None,
        user_api_key_dict: Any = None,
        response: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Inspect tool calls returned by the model and gate protected ones."""
        data = kwargs.pop("data", data)
        response = kwargs.pop("response", response)
        await self._gate_calls(_iter_tool_calls(response))
        return None

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: Any = None,
        response: AsyncIterable[Any] | None = None,
        request_data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Buffer a stream, gate reconstructed tool calls, then yield chunks."""
        response = kwargs.pop("response", response)
        request_data = kwargs.pop("request_data", request_data)
        if isinstance(request_data, dict) and request_data.get("stream") is True and not self.allow_streaming:
            raise ValueError(STREAMING_REJECTION_MESSAGE)
        if response is None:
            return
        buffered = [chunk async for chunk in response]
        await self._gate_calls(_iter_tool_calls_from_chunks(buffered))
        for chunk in buffered:
            yield chunk
