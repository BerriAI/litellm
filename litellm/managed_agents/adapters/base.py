"""Sandbox adapter protocol for managed agents v2.

Each sandbox provider (today: only opencode) implements this protocol. The
endpoint handlers in `litellm/proxy/managed_agents_endpoints/router.py` resolve an adapter
via `registry.get_adapter(sandbox_type)` and use it to translate sandbox-
agnostic public-API calls into provider-specific HTTP requests.

The Protocol exists for typing only — concrete adapters (e.g.
`OpencodeAdapter`) are regular classes whose method signatures match.

Adapters are stateless: every call carries the `sandbox_url` and provider
session id. The handler does all DB work; the adapter only translates HTTP.
"""

from typing import AsyncIterator, Optional, Protocol, Tuple

from litellm.managed_agents.types import MessageRow


class SandboxUnreachableError(Exception):
    """Raised by adapter HTTP calls on connect/timeout failures.

    Endpoint handlers translate this to a 504 response with body
    `{"detail":{"error":"Sandbox unreachable"}}` per contract §7.
    """


class SandboxBadGatewayError(Exception):
    """Raised when the sandbox responds but the body is malformed.

    Endpoint handlers translate this to a 502 response with body
    `{"detail":{"error":"Bad gateway"}}` per contract §7.
    """


class SandboxAdapter(Protocol):
    """Protocol every sandbox adapter must implement.

    Concrete adapters do NOT need to inherit from this class — duck typing
    is enough. The Protocol is here so callers can type-annotate adapter
    handles and so static type checkers can verify each concrete adapter
    matches the contract.
    """

    async def send_message(
        self,
        sandbox_url: str,
        opencode_session_id: str,
        content: str,
        model: Optional[str],
    ) -> None:
        """Forward a user message to the sandbox.

        Translates to `POST <sandbox_url>/session/<oc_sid>/prompt_async` for
        the opencode adapter. The sandbox returns 204; the caller is
        responsible for writing the user `MessageRow` separately. Raises
        `SandboxUnreachableError` on connect/timeout failures.
        """
        ...

    async def list_messages(
        self,
        sandbox_url: str,
        opencode_session_id: str,
        our_session_id: str,
        limit: int = 50,
    ) -> list[MessageRow]:
        """Fetch the message history for a session, normalized to our shape.

        Translates to `GET <sandbox_url>/session/<oc_sid>/message` for the
        opencode adapter. Each provider message is run through normalization
        before being returned. Raises `SandboxUnreachableError` on
        connect/timeout failures and `SandboxBadGatewayError` on malformed
        responses.
        """
        ...

    def stream_events(
        self,
        sandbox_url: str,
        opencode_session_id: str,
        our_session_id: str,
    ) -> AsyncIterator[Tuple[str, dict]]:
        """Subscribe to the sandbox event bus and yield normalized events.

        First yield is always `("connected", {"session_id": our_session_id})`
        (synthesized — not from the upstream bus). Subsequent yields are the
        normalized `(event_type, data_dict)` tuples per contract §7. Events
        not in the translation table are dropped. Raises
        `SandboxUnreachableError` on connect/timeout failures.

        NOTE: this is declared as a regular ``def`` returning an
        ``AsyncIterator`` because concrete adapters implement it as an
        ``async def ... yield`` (async generator). An ``async def f() ->
        AsyncIterator[T]: yield x`` is typed as ``AsyncIterator[T]``,
        not ``Coroutine[..., AsyncIterator[T]]`` — declaring the
        protocol method as ``async def`` would force callers to ``await``
        it before iterating, which is the wrong runtime shape.
        """
        ...

    async def delete(
        self,
        sandbox_url: str,
        opencode_session_id: str,
    ) -> None:
        """Best-effort delete of the upstream sandbox session.

        Translates to `DELETE <sandbox_url>/session/<oc_sid>` for the
        opencode adapter. Errors are logged and swallowed — sandbox
        teardown is also enforced by idle/timeout in the provider.
        """
        ...

    async def abort(
        self,
        sandbox_url: str,
        opencode_session_id: str,
    ) -> None:
        """Abort the in-flight turn for a session.

        Translates to `POST <sandbox_url>/session/<oc_sid>/abort` for the
        opencode adapter (empty body). Best-effort: any non-2xx response
        from the provider is swallowed (the session may already be done).
        Raises `SandboxUnreachableError` only on connect/timeout failure.
        """
        ...
