"""opencode adapter for managed agents v2.

Translates the sandbox-agnostic public API into opencode-specific HTTP
calls per contract §7. Stateless across calls — the handler passes
``sandbox_url`` and the opencode session id on every invocation.

Real-shape compatibility: this adapter is written against the actual
opencode 1.14.41 wire shapes documented in
``.claude/v2_opencode_real_responses.md``. Notable behaviors:

  - ``send_message`` splits our ``"<provider>/<model>"`` string into
    opencode's ``{providerID, modelID}`` object form.
  - ``stream_events`` maintains a per-stream ``partID -> type`` map
    populated from ``message.part.updated`` so streaming
    ``message.part.delta`` events can be routed to text/reasoning.
  - The stream loop also tracks the most recent assistant ``message_id``
    so ``session.idle`` (which doesn't carry one in opencode payloads)
    can be enriched before being yielded as ``message.completed``.
  - On ``permission.asked`` events for OUR session the adapter fires a
    fire-and-forget auto-grant request so tool calls don't hang waiting
    for approval. Permission events for other opencode sessions on the
    same server are dropped — auto-grant only runs after the
    ``event_matches_session`` filter so a caller streaming /events on
    session A cannot approve permissions for session B.
    This is MVP behavior — once we expose permission gating to v2
    callers we'll surface these as real events instead.

HTTP via ``httpx.AsyncClient``. SSE in ``stream_events`` parsed manually
from the chunked body — opencode emits standard ``data:`` lines + blank
separator over ``text/event-stream``.

All connection errors bubble up as ``SandboxUnreachableError``;
malformed upstream responses bubble up as ``SandboxBadGatewayError``.
The endpoint handlers translate these to 504 / 502 respectively.
"""

import asyncio
import json
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.managed_agents.adapters.base import (
    SandboxBadGatewayError,
    SandboxUnreachableError,
)
from litellm.managed_agents.adapters.normalization import (
    event_matches_session,
    normalize_opencode_event,
    normalize_opencode_message,
)
from litellm.managed_agents.types import MessageRow
from litellm.types.llms.custom_http import httpxSpecialProvider

# Reasonable default timeouts — these match the proxy's general HTTP defaults.
# The streaming endpoint uses an unbounded read because SSE is long-lived.
_DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)
_STREAM_TIMEOUT = httpx.Timeout(connect=5.0, read=None, write=10.0, pool=5.0)
_PERMISSION_TIMEOUT = httpx.Timeout(connect=5.0, read=5.0, write=5.0, pool=5.0)


def _get_async_client(timeout: httpx.Timeout) -> httpx.AsyncClient:
    """Return a cached ``httpx.AsyncClient`` from the proxy's shared pool.

    Creating a fresh ``httpx.AsyncClient`` per request adds ~500 ms of
    setup overhead and prevents connection reuse — the proxy's
    ``get_async_httpx_client`` caches one client per (provider, params)
    tuple so we share connections across calls. We pass the timeout into
    the cache key so the three distinct timeout profiles
    (default / stream / permission) each get their own pooled client.
    """
    handler = get_async_httpx_client(
        llm_provider=httpxSpecialProvider.ManagedAgents,
        params={"timeout": timeout},
    )
    return handler.client


class OpencodeAdapter:
    """Adapter for opencode HTTP servers.

    The class itself is stateless — every method takes the
    ``sandbox_url`` and ``opencode_session_id`` it needs. Per-stream
    state for ``stream_events`` (e.g. the ``partID -> type`` map) lives
    inside the generator's local frame, not on the class.
    """

    async def send_message(
        self,
        sandbox_url: str,
        opencode_session_id: str,
        content: str,
        model: Optional[str],
    ) -> None:
        """POST <sandbox_url>/session/<oc_sid>/prompt_async.

        opencode 1.14.41 validates the body's ``model`` field as an
        OBJECT, not a string. We split our ``"<provider>/<model>"`` form
        on the first ``/`` into ``{providerID, modelID}``. If the input
        has no ``/`` we omit the field entirely and let opencode fall
        back to its default agent/model.

        opencode returns 204 No Content on success — we do not parse a
        body. The caller (handler) synthesizes our ``msg_*`` id and
        writes the user MessageRow.
        """
        url = self._url(sandbox_url, f"/session/{opencode_session_id}/prompt_async")
        body: Dict[str, Any] = {
            "parts": [{"type": "text", "text": content}],
        }
        model_obj = self._build_model_object(model)
        if model_obj is not None:
            body["model"] = model_obj

        try:
            client = _get_async_client(_DEFAULT_TIMEOUT)
            response = await client.post(url, json=body)
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise SandboxUnreachableError(
                f"opencode unreachable at {sandbox_url}: {e}"
            ) from e

        # opencode contract: 204 No Content on success. Some versions may
        # return 200 with an empty body — accept either as success.
        if response.status_code not in (200, 204):
            raise SandboxBadGatewayError(
                f"opencode prompt_async returned {response.status_code}: "
                f"{response.text[:500]}"
            )

    async def list_messages(
        self,
        sandbox_url: str,
        opencode_session_id: str,
        our_session_id: str,
        limit: int = 50,
    ) -> List[MessageRow]:
        """GET <sandbox_url>/session/<oc_sid>/message → normalize → list.

        opencode returns a bare JSON array of ``{info, parts}`` envelopes.
        opencode does not support server-side ``limit`` / ``cursor`` query
        params on this endpoint, so we honor ``limit`` client-side after
        normalization.
        """
        url = self._url(sandbox_url, f"/session/{opencode_session_id}/message")

        try:
            client = _get_async_client(_DEFAULT_TIMEOUT)
            response = await client.get(url)
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise SandboxUnreachableError(
                f"opencode unreachable at {sandbox_url}: {e}"
            ) from e

        if response.status_code != 200:
            raise SandboxBadGatewayError(
                f"opencode list-messages returned {response.status_code}: "
                f"{response.text[:500]}"
            )

        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError) as e:
            raise SandboxBadGatewayError(
                f"opencode list-messages returned non-JSON: {e}"
            ) from e

        if not isinstance(payload, list):
            raise SandboxBadGatewayError(
                f"opencode list-messages expected list, got {type(payload).__name__}"
            )

        # Normalize, then truncate. We truncate AFTER normalizing so the
        # ordering matches what opencode returned.
        normalized = [
            normalize_opencode_message(item, our_session_id)
            for item in payload
            if isinstance(item, dict)
        ]
        if limit and limit > 0:
            normalized = normalized[:limit]
        return normalized

    async def stream_events(
        self,
        sandbox_url: str,
        opencode_session_id: str,
        our_session_id: str,
    ) -> AsyncIterator[Tuple[str, dict]]:
        """GET <sandbox_url>/event (SSE) → yield normalized events.

        First yield is the synthesized ``("connected", {"session_id": ...})``.
        After that, only events whose payload references
        ``opencode_session_id`` are yielded, normalized via
        ``normalize_opencode_event``.

        Per-stream state:
          - ``part_types``: maps each ``partID`` to its part type
            (``text`` / ``reasoning`` / ``tool`` / ...). Populated from
            ``message.part.updated`` events; consulted by
            ``message.part.delta`` events to route deltas correctly.
          - ``current_assistant_message_id``: the most recent assistant
            ``msg_*`` we've seen in a ``message.updated`` event. Used to
            attach a ``message_id`` to ``session.idle`` events that don't
            carry one (per real opencode payloads).

        Permission gating: ``permission.asked`` events trigger a
        fire-and-forget auto-grant via
        ``POST /session/:sid/permissions/:per_id`` with
        ``{"response": "once"}``. Without this, gated tool calls hang
        forever.
        """
        # Synthesized first event — see contract §6.6.
        yield ("connected", {"session_id": our_session_id})

        url = self._url(sandbox_url, "/event")

        # Per-stream state — scoped to one generator invocation so
        # multiple concurrent streams do not interfere.
        part_types: Dict[str, str] = {}
        current_assistant_message_id: Optional[str] = None

        try:
            client = _get_async_client(_STREAM_TIMEOUT)
            async with client.stream("GET", url) as response:
                if response.status_code != 200:
                    raise SandboxBadGatewayError(
                        f"opencode /event returned {response.status_code}"
                    )
                async for parsed in self._iter_sse(response):
                    # Drop events for other sessions FIRST. Without this,
                    # any caller streaming /events on session A could
                    # auto-grant permissions for session B on the same
                    # opencode server.
                    if not event_matches_session(parsed, opencode_session_id):
                        continue

                    # Auto-grant permission for OUR session only.
                    if parsed.get("type") == "permission.asked":
                        self._auto_grant_permission(sandbox_url, parsed)
                        continue

                    # Track the in-flight assistant message id BEFORE
                    # routing — so ``session.idle`` (which has no
                    # messageID) can be enriched.
                    current_assistant_message_id = self._track_assistant_message_id(
                        parsed, current_assistant_message_id
                    )

                    normalized = normalize_opencode_event(parsed, part_types)
                    if normalized is None:
                        continue

                    event_type, data = normalized
                    if (
                        event_type == "message.completed"
                        and not data.get("message_id")
                        and current_assistant_message_id
                    ):
                        data = {**data, "message_id": current_assistant_message_id}

                    yield (event_type, data)
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise SandboxUnreachableError(
                f"opencode unreachable at {sandbox_url}: {e}"
            ) from e

    async def delete(
        self,
        sandbox_url: str,
        opencode_session_id: str,
    ) -> None:
        """DELETE <sandbox_url>/session/<oc_sid>. Best-effort.

        Errors are logged at debug level and swallowed — the sandbox will
        also be torn down on idle/timeout regardless.
        """
        url = self._url(sandbox_url, f"/session/{opencode_session_id}")
        try:
            client = _get_async_client(_DEFAULT_TIMEOUT)
            response = await client.delete(url)
            if response.status_code not in (200, 202, 204, 404):
                verbose_logger.debug(
                    "opencode delete returned %d for %s/%s",
                    response.status_code,
                    sandbox_url,
                    opencode_session_id,
                )
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            # Best-effort — log and swallow.
            verbose_logger.debug(
                "opencode delete failed (non-fatal) for %s/%s: %s",
                sandbox_url,
                opencode_session_id,
                e,
            )

    async def abort(
        self,
        sandbox_url: str,
        opencode_session_id: str,
    ) -> None:
        """POST <sandbox_url>/session/<oc_sid>/abort. Best-effort.

        Aborts the in-flight turn for a session. Connection failures bubble
        up as ``SandboxUnreachableError`` so the handler can return 504.
        Non-2xx responses from opencode (404 if the session is already
        terminated, 4xx if there's no in-flight turn) are swallowed —
        aborting an idle session is not a meaningful error.
        """
        url = self._url(sandbox_url, f"/session/{opencode_session_id}/abort")
        try:
            client = _get_async_client(_DEFAULT_TIMEOUT)
            response = await client.post(url, json={})
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise SandboxUnreachableError(
                f"opencode unreachable at {sandbox_url}: {e}"
            ) from e

        if response.status_code not in (200, 202, 204):
            verbose_logger.debug(
                "opencode abort returned %d for %s/%s (best-effort, ignored)",
                response.status_code,
                sandbox_url,
                opencode_session_id,
            )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _url(sandbox_url: str, path: str) -> str:
        """Join ``sandbox_url`` and ``path`` into a full URL.

        We intentionally avoid ``urljoin`` here because opencode's base
        URL usually has no trailing slash and ``urljoin`` against an
        absolute path would drop unrelated path components. Simple concat
        is fine.
        """
        base = sandbox_url.rstrip("/")
        if not path.startswith("/"):
            path = "/" + path
        return base + path

    @staticmethod
    def _build_model_object(model: Optional[str]) -> Optional[Dict[str, str]]:
        """Convert our ``"<provider>/<model>"`` string into opencode's
        ``{providerID, modelID}`` object form.

        Returns None when ``model`` is falsy or has no ``/`` separator —
        the caller omits the field entirely so opencode falls back to its
        default agent/model.
        """
        if not model or not isinstance(model, str):
            return None
        if "/" not in model:
            return None
        provider_id, model_id = model.split("/", 1)
        if not provider_id or not model_id:
            return None
        return {"providerID": provider_id, "modelID": model_id}

    @staticmethod
    def _track_assistant_message_id(
        parsed: Dict[str, Any],
        current: Optional[str],
    ) -> Optional[str]:
        """Update the cached "current assistant message id" based on a
        ``message.updated`` event. Returns the new value (or the existing
        one if the event is for something else)."""
        if parsed.get("type") != "message.updated":
            return current
        props = parsed.get("properties") or {}
        if not isinstance(props, dict):
            return current
        info = props.get("info") or {}
        if not isinstance(info, dict):
            return current
        if info.get("role") != "assistant":
            return current
        msg_id = info.get("id")
        if isinstance(msg_id, str) and msg_id:
            return msg_id
        return current

    @staticmethod
    def _auto_grant_permission(
        sandbox_url: str,
        parsed: Dict[str, Any],
    ) -> None:
        """Fire-and-forget auto-grant for ``permission.asked`` events.

        opencode emits ``permission.asked`` for tool calls that need user
        approval (external directory reads, network calls, etc.). Until
        we expose permission gating to v2 callers, we auto-grant via
        ``POST /session/:sid/permissions/:per_id`` body
        ``{"response":"once"}`` so tool calls don't hang.

        Uses ``asyncio.create_task`` so the SSE consumer doesn't block on
        the round-trip. Any failures are logged at debug level only —
        opencode will time out the tool itself if approval never arrives.
        """
        props = parsed.get("properties") or {}
        if not isinstance(props, dict):
            return
        permission_id = props.get("id")
        session_id = props.get("sessionID")
        if not isinstance(permission_id, str) or not isinstance(session_id, str):
            return
        if not permission_id or not session_id:
            return

        url = OpencodeAdapter._url(
            sandbox_url, f"/session/{session_id}/permissions/{permission_id}"
        )

        async def _grant() -> None:
            try:
                client = _get_async_client(_PERMISSION_TIMEOUT)
                response = await client.post(url, json={"response": "once"})
                if response.status_code not in (200, 204):
                    verbose_logger.debug(
                        "opencode permission grant returned %d for %s",
                        response.status_code,
                        url,
                    )
            except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError) as e:
                verbose_logger.debug(
                    "opencode permission grant failed (non-fatal) for %s: %s",
                    url,
                    e,
                )

        try:
            asyncio.create_task(_grant())
        except RuntimeError:
            # No running loop — extremely unlikely in our async context,
            # but better to log than to crash the SSE pipeline.
            verbose_logger.debug(
                "opencode permission grant skipped (no running loop) for %s",
                url,
            )

    @staticmethod
    async def _iter_sse(
        response: httpx.Response,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Parse a server-sent-events stream from an httpx Response.

        opencode emits standard SSE: ``data:`` lines terminated by a
        blank line. (No ``event:`` line — the type lives inside the
        JSON.) We accumulate ``data:`` lines per frame, JSON-decode the
        payload, and merge ``event:`` (when present) into the dict under
        a ``type`` key as a backstop.
        """
        data_lines: List[str] = []
        sse_event_name: Optional[str] = None

        async for raw_line in response.aiter_lines():
            # SSE frames: ``event: <name>\n``, ``data: <payload>\n``, blank line.
            line = raw_line.rstrip("\r")

            if line == "":
                # Frame boundary — emit if we have data buffered.
                if data_lines:
                    payload = "\n".join(data_lines)
                    data_lines = []
                    parsed = OpencodeAdapter._parse_sse_data(payload, sse_event_name)
                    sse_event_name = None
                    if parsed is not None:
                        yield parsed
                continue

            if line.startswith(":"):
                # SSE comment / heartbeat — ignore.
                continue

            if line.startswith("event:"):
                sse_event_name = line[len("event:") :].strip() or None
                continue

            if line.startswith("data:"):
                data_lines.append(line[len("data:") :].lstrip())
                continue

            # Unknown SSE field (id:, retry:, etc.) — ignore.

        # Tail flush in case the stream ended without a trailing blank line.
        if data_lines:
            payload = "\n".join(data_lines)
            parsed = OpencodeAdapter._parse_sse_data(payload, sse_event_name)
            if parsed is not None:
                yield parsed

    @staticmethod
    def _parse_sse_data(
        payload: str,
        sse_event_name: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """JSON-decode an SSE data payload, merging the SSE event name in.

        Returns None on malformed JSON — we do not raise from inside the
        stream loop because a single bad frame should not kill the whole
        subscription. The caller filters/normalizes the dict afterwards.
        """
        if not payload:
            return None
        try:
            obj = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            verbose_logger.debug("Dropping malformed SSE frame: %r", payload[:200])
            return None
        if not isinstance(obj, dict):
            return None
        # opencode includes ``type`` inside the JSON payload itself, but if
        # a frame uses the SSE ``event:`` field instead, fall back to that.
        if "type" not in obj and sse_event_name:
            obj["type"] = sse_event_name
        return obj
