"""
The outside sources a Responses request reads: the stored turns a follow-up continues and the client that opens a
hosted MCP server. They sit in a context variable rather than a parameter, since the proxy hands every request body key
to ``litellm.aresponses()`` as a keyword argument, so a parameter of that name could be bound from a request. Only a
caller that swaps a source, such as a test, sets it. The entry point reads it once, and what one request reuses across
its rounds travels with the call as a ``TurnReplay``, which no request body can build.
"""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from litellm.responses.litellm_completion_transformation.session_handler import (
    ResponsesSessionHandler,
    StoredTurnsSource,
)

if TYPE_CHECKING:
    from litellm.responses.mcp.direct_url_claims import DirectClientFactory


@dataclass(frozen=True, slots=True)
class ResponsesDependencies:
    stored_turns_source: StoredTurnsSource = ResponsesSessionHandler.get_stored_turns_for_previous_response_id
    direct_client_factory: "DirectClientFactory | None" = None


DEFAULT_RESPONSES_DEPENDENCIES: Final = ResponsesDependencies()
_dependencies: Final[ContextVar[ResponsesDependencies]] = ContextVar(
    "responses_dependencies", default=DEFAULT_RESPONSES_DEPENDENCIES
)


def responses_dependencies() -> ResponsesDependencies:
    return _dependencies.get()


@contextmanager
def using_responses_dependencies(dependencies: ResponsesDependencies) -> Generator[None]:
    token: Final = _dependencies.set(dependencies)
    try:
        yield
    finally:
        _dependencies.reset(token)
