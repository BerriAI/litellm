from typing import Final

import pytest

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.agent_endpoints.auth.agent_caller import agent_caller_auth, agent_caller_from_headers
from litellm.types.agents import AgentCaller

_AGENT_KEY: Final = UserAPIKeyAuth(api_key="agent-key", user_id="agent-owner", team_id="agent-team", agent_id="agent-1")


def test_agent_key_echoing_both_ids_acts_for_that_user_and_team() -> None:
    headers: Final = {"X-LiteLLM-User-Id": " alice ", "x-litellm-team-id": "callers"}

    assert agent_caller_from_headers(headers, _AGENT_KEY) == AgentCaller(user_id="alice", team_id="callers")


def test_agent_key_echoing_only_a_user_id_acts_for_a_teamless_user() -> None:
    assert agent_caller_from_headers({"x-litellm-user-id": "alice"}, _AGENT_KEY) == AgentCaller(user_id="alice")


@pytest.mark.parametrize("headers", [{}, {"x-litellm-user-id": "  ", "x-litellm-team-id": ""}])
def test_agent_key_echoing_no_caller_acts_for_itself(headers: dict[str, str]) -> None:
    assert agent_caller_from_headers(headers, _AGENT_KEY) is None


def test_caller_headers_on_a_key_without_an_agent_are_ignored() -> None:
    plain_key: Final = UserAPIKeyAuth(api_key="plain-key", user_id="bob")

    assert agent_caller_from_headers({"x-litellm-user-id": "alice", "x-litellm-team-id": "callers"}, plain_key) is None


def test_caller_auth_stands_for_the_invoking_user_not_the_agent() -> None:
    agent_key: Final = UserAPIKeyAuth(
        api_key="agent-key", user_id="agent-owner", team_id="agent-team", agent_id="agent-1"
    )
    agent_key.agent_caller = AgentCaller(user_id="alice", team_id="callers")

    caller_auth: Final = agent_caller_auth(agent_key)

    assert caller_auth is not None
    assert (caller_auth.user_id, caller_auth.team_id, caller_auth.agent_id, caller_auth.api_key) == (
        "alice",
        "callers",
        None,
        None,
    )
    assert agent_caller_auth(_AGENT_KEY) is None


def test_agent_caller_cannot_be_set_from_a_request_payload() -> None:
    forged: Final = UserAPIKeyAuth.model_validate(
        {"api_key": "agent-key", "agent_id": "agent-1", "agent_caller": {"user_id": "alice", "team_id": "callers"}}
    )

    assert forged.agent_caller is None
    assert "agent_caller" not in forged.model_dump()
