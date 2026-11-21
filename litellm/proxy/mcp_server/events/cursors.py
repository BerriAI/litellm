import hashlib
import json
from dataclasses import dataclass
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from litellm.proxy.mcp_server.events.errors import EventsFailure, internal_error, invalid_params
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok, Result
from litellm.proxy.mcp_server.state_tokens import StateTokenError, open_state, seal_state
from litellm.types.mcp_server.events import JsonObject

_PURPOSE: Final = "mcp-events-cursor"
_LIFETIME_SECONDS: Final = 400 * 24 * 60 * 60


def arguments_digest(arguments: JsonObject) -> str:
    return hashlib.sha256(
        json.dumps(arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class CursorBinding:
    server_id: str
    event: str
    args_hash: str
    caller: str


@dataclass(frozen=True, slots=True)
class OpenedCursor:
    upstream: str | None
    expired: bool = False


class _CursorState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    v: Literal[1] = 1
    server_id: str
    event: str
    args_hash: str
    caller: str
    upstream: str


def _token_failure(error: StateTokenError) -> EventsFailure:
    if error is StateTokenError.MISSING_KEY:
        return internal_error(error.value)
    return invalid_params("Invalid event cursor")


def seal_cursor(binding: CursorBinding, upstream: str | None, now: int) -> Result[str | None, EventsFailure]:
    if upstream is None:
        return Ok(None)
    state: Final = _CursorState(
        server_id=binding.server_id,
        event=binding.event,
        args_hash=binding.args_hash,
        caller=binding.caller,
        upstream=upstream,
    )
    sealed: Final = seal_state(
        state.model_dump(mode="json"), purpose=_PURPOSE, expires_at=now + _LIFETIME_SECONDS, now=now
    )
    if isinstance(sealed, Error):
        return Error(_token_failure(sealed.error))
    return Ok(sealed.ok)


def open_cursor(token: str | None, binding: CursorBinding, now: int) -> Result[OpenedCursor, EventsFailure]:
    if token is None:
        return Ok(OpenedCursor(upstream=None))
    opened: Final = open_state(token, purpose=_PURPOSE, now=now)
    if isinstance(opened, Error):
        if opened.error is StateTokenError.EXPIRED:
            return Ok(OpenedCursor(upstream=None, expired=True))
        return Error(_token_failure(opened.error))
    try:
        state: Final = _CursorState.model_validate(opened.ok)
    except ValidationError:
        return Error(invalid_params("Invalid event cursor"))
    if (state.server_id, state.event, state.args_hash, state.caller) != (
        binding.server_id,
        binding.event,
        binding.args_hash,
        binding.caller,
    ):
        return Error(invalid_params("Event cursor does not belong to this subscription"))
    return Ok(OpenedCursor(upstream=state.upstream))


def ensure_sealable(binding: CursorBinding, now: int) -> EventsFailure | None:
    """Fail before any upstream work when cursors cannot be sealed, so no event is consumed and then lost."""
    probe: Final = seal_cursor(binding, "probe", now)
    return probe.error if isinstance(probe, Error) else None
