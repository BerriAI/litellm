import pytest

from litellm._auth.failure import NativeAuthFailure, native_boundary
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.agent_endpoints.auth.agent_permission_handler import AgentAccess, AgentRequestHandler


@pytest.mark.parametrize("error", (TypeError("bad fact"), ValueError("bad wire"), RuntimeError("native error")))
def test_native_errors_keep_public_exception_type_and_are_marked_terminal(error: Exception) -> None:
    @native_boundary
    def entry(value: int) -> int:
        raise error

    with pytest.raises(type(error), match=str(error)) as raised:
        entry(1)
    assert isinstance(raised.value, NativeAuthFailure)
    assert raised.value.__cause__ is error


@pytest.mark.asyncio
async def test_native_failure_never_becomes_unrestricted_agent_access() -> None:
    @native_boundary
    def broken_entry() -> None:
        raise ValueError("invalid native policy answer")

    async def key_lookup(auth: UserAPIKeyAuth | None, *, strict: bool) -> AgentAccess:
        broken_entry()
        raise AssertionError("native failure returned")

    async def team_lookup(auth: UserAPIKeyAuth | None, *, strict: bool) -> AgentAccess:
        raise AssertionError("team lookup followed native failure")

    with pytest.raises(NativeAuthFailure, match="invalid native policy answer"):
        await AgentRequestHandler.resolve_key_team_agent_access(
            UserAPIKeyAuth(), lookup_key_access=key_lookup, lookup_team_access=team_lookup
        )
