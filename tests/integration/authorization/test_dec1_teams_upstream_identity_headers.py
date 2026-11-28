"""Team identity in the `x-litellm-user_api_key_*` headers sent to the provider.

Contract, read from litellm/proxy/litellm_pre_call_utils.py (`add_litellm_data_for_backend_llm_call`,
`add_headers_to_llm_call`, `get_sanitized_user_information_from_key`): with
`general_settings.forward_client_headers_to_llm_api: true` and `litellm_settings.add_user_information_to_llm_headers:
true`, every non null field of the caller's identity is sent upstream as `x-litellm-<field>`. Strings are sent as is,
numbers with `str()`, objects and lists as JSON, and a None field is left out entirely.

For a team key that means `x-litellm-user_api_key_team_id`, `x-litellm-user_api_key_team_alias` and
`x-litellm-user_api_key_team_max_budget`, all read from the auth object the team lookup (#52176, `RustTeamStore`)
fills. A port that loses the team alias or sends `None` as a string would change what providers and their gateways
see. The key hash header and the credential properties are pinned in
`security/test_dec1_authsec_forwarded_credentials.py`; this file covers only the team fields.

One owned proxy is booted for the module through `owned_door` (security/dec1_authsec_support.py), in front of a
recording provider whose requests the tests read.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Final

import pytest

from tests.integration._support.wire import Request
from tests.integration.security.dec1_authsec_support import (
    CHAT,
    OWNED_MODEL,
    Door,
    assert_served,
    chat_body,
    owned_door,
)

_TEAM_ID_HEADER: Final = "x-litellm-user_api_key_team_id"
_TEAM_ALIAS_HEADER: Final = "x-litellm-user_api_key_team_alias"
_TEAM_BUDGET_HEADER: Final = "x-litellm-user_api_key_team_max_budget"


@pytest.fixture(scope="module")
def door(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Door]:
    with owned_door(
        tmp_path_factory.mktemp("dec1-teams-identity-headers"),
        general_settings={"forward_client_headers_to_llm_api": True},
        litellm_settings={"add_user_information_to_llm_headers": True},
    ) as owned:
        yield owned


def _one_call(door: Door, key: str) -> Request:
    door.calls()
    assert_served(door.gateway.request("POST", CHAT, chat_body(OWNED_MODEL), key=key))
    (call,) = door.calls()
    return call


def test_a_team_key_sends_the_team_id_and_alias_upstream(door: Door) -> None:
    """A team with alias `<alias>` has a key that calls chat.

    The provider receives `x-litellm-user_api_key_team_id: <team>` and `x-litellm-user_api_key_team_alias: <alias>`.
    """
    with door.gateway.scenario() as scenario:
        alias: Final = f"dec1-teams-alias-{uuid.uuid4().hex[:8]}"
        team: Final = scenario.team(team_alias=alias)
        call: Final = _one_call(door, scenario.key(team_id=team))
        assert call.headers.get(_TEAM_ID_HEADER) == team, call.headers
        assert call.headers.get(_TEAM_ALIAS_HEADER) == alias, call.headers


def test_a_key_without_a_team_sends_no_team_headers(door: Door) -> None:
    """A key with no team calls chat.

    None of the team headers is present at all, not even as the string `None`: `add_headers_to_llm_call` skips a
    field whose value is None.
    """
    with door.gateway.scenario() as scenario:
        call: Final = _one_call(door, scenario.key())
        for header in (_TEAM_ID_HEADER, _TEAM_ALIAS_HEADER, _TEAM_BUDGET_HEADER):
            assert header not in call.headers, call.headers


def test_a_team_max_budget_is_sent_as_its_decimal_string(door: Door) -> None:
    """A team with `max_budget: 5` has a key that calls chat.

    The provider receives `x-litellm-user_api_key_team_max_budget: 5.0`: the value is a float on the auth object
    and numbers are sent with `str()`.
    """
    with door.gateway.scenario() as scenario:
        team: Final = scenario.team(max_budget=5)
        call: Final = _one_call(door, scenario.key(team_id=team))
        assert call.headers.get(_TEAM_BUDGET_HEADER) == "5.0", call.headers


def test_a_team_without_a_budget_sends_no_team_budget_header(door: Door) -> None:
    """A team created without `max_budget` has a key that calls chat.

    The team id header is sent and the team budget header is absent.
    """
    with door.gateway.scenario() as scenario:
        team: Final = scenario.team()
        call: Final = _one_call(door, scenario.key(team_id=team))
        assert call.headers.get(_TEAM_ID_HEADER) == team, call.headers
        assert _TEAM_BUDGET_HEADER not in call.headers, call.headers


def test_a_key_moved_to_another_team_sends_the_new_team_id(door: Door) -> None:
    """A key is served once in team one and moved to team two with `/key/update`; it calls chat again.

    The second provider request carries `x-litellm-user_api_key_team_id: <team two>` and team two's alias: the move
    evicts the cached key object.
    """
    with door.gateway.scenario() as scenario:
        second_alias: Final = f"dec1-teams-second-{uuid.uuid4().hex[:8]}"
        first: Final = scenario.team()
        second: Final = scenario.team(team_alias=second_alias)
        key: Final = scenario.key(team_id=first)
        assert _one_call(door, key).headers.get(_TEAM_ID_HEADER) == first
        door.gateway.post("/key/update", {"key": key, "team_id": second})
        call: Final = _one_call(door, key)
        assert call.headers.get(_TEAM_ID_HEADER) == second, call.headers
        assert call.headers.get(_TEAM_ALIAS_HEADER) == second_alias, call.headers
