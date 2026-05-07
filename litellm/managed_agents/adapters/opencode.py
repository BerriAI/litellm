"""opencode adapter for managed agents v2.

Translates the sandbox-agnostic public API into opencode-specific HTTP
calls per contract §7. Stateless — the handler passes `sandbox_url` and
the opencode session id on every call.

HTTP via `httpx.AsyncClient`. SSE in `stream_events` parsed manually from
the chunked HTTP body — opencode emits standard `event:` / `data:` /
blank-line frames over `text/event-stream`.

All connection errors bubble up as `SandboxUnreachableError`; malformed
upstream responses bubble up as `SandboxBadGatewayError`. The endpoint
handlers translate these to 504 / 502 respectively.
"""

import json
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx

from litellm._logging import verbose_logger
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

# Reasonable default timeouts — these match the proxy's general HTTP defaults.
# The streaming endpoint uses an unbounded read because SSE is long-lived.
_DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)
_STREAM_TIMEOUT = httpx.Timeout(connect=5.0, read=None, write=10.0, pool=5.0)


class OpencodeAdapter:
    """Adapter for opencode HTTP servers.

    The class is stateless. We instantiate one per process (via
    `registry.get_adapter`) but it carries no per-call state — every
    method takes the `sandbox_url` and `opencode_session_id` it needs.
    """

    async def send_message(
        self,
        sandbox_url: str,
        opencode_session_id: str,
        content: str,
        model: Optional[str],
    ) -> None:
        """POST <sandbox_url>/session/<oc_sid>/prompt_async.

        opencode returns 204 — we do not parse a body. The caller (handler)
        synthesizes our `msg_*` id and writes the user MessageRow.
        """
        url = self._url(sandbox_url, f"/session/{opencode_session_id}/prompt_async")
        body: Dict[str, Any] = {
            "parts": [{"type": "text", "text": content}],
        }
        if model:
            body["model"] = model

        try:
            async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
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

        opencode returns a JSON array. The `limit` parameter is honored
        client-side; opencode's API does not currently support pagination
        on this endpoint (TODO: verify against preflight findings).
        """
        url = self._url(sandbox_url, f"/session/{opencode_session_id}/message")

        try:
            async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
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

        First yield is the synthesized `("connected", {"session_id": ...})`.
        After that, only events whose payload references
        `opencode_session_id` are yielded. Events not in the translation
        table (per contract §7) are dropped.
        """
        # Synthesized first event — see contract §6.6.
        yield ("connected", {"session_id": our_session_id})

        url = self._url(sandbox_url, "/event")

        try:
            async with httpx.AsyncClient(timeout=_STREAM_TIMEOUT) as client:
                async with client.stream("GET", url) as response:
                    if response.status_code != 200:
                        raise SandboxBadGatewayError(
                            f"opencode /event returned {response.status_code}"
                        )
                    async for parsed in self._iter_sse(response):
                        if not event_matches_session(parsed, opencode_session_id):
                            continue
                        normalized = normalize_opencode_event(parsed)
                        if normalized is None:
                            continue
                        yield normalized
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
            async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
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

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _url(sandbox_url: str, path: str) -> str:
        """Join `sandbox_url` and `path` into a full URL.

        We intentionally avoid `urljoin` here because opencode's base URL
        usually has no trailing slash and an `urljoin` against an absolute
        path would drop unrelated path components. Simple concat is fine.
        """
        base = sandbox_url.rstrip("/")
        if not path.startswith("/"):
            path = "/" + path
        return base + path

    @staticmethod
    async def _iter_sse(
        response: httpx.Response,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Parse a server-sent-events stream from an httpx Response.

        opencode emits standard SSE: `event:` / `data:` lines terminated by
        a blank line. We accumulate `data:` lines per frame, JSON-decode
        the payload, and merge `event:` (when present) into the dict under
        a `type` key — opencode's event payloads already carry their own
        `type`, so the merge is mostly a backstop for events that omit it.
        """
        data_lines: List[str] = []
        sse_event_name: Optional[str] = None

        async for raw_line in response.aiter_lines():
            # SSE frames: `event: <name>\n`, `data: <payload>\n`, blank line.
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
        # opencode includes `type` inside the JSON payload itself, but if a
        # frame uses the SSE `event:` field instead, fall it back to that.
        if "type" not in obj and sse_event_name:
            obj["type"] = sse_event_name
        return obj
