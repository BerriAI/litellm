"""How long a proxy keeps serving an auth object it cached, when the row changes behind its back.

Contract, read from litellm/proxy/proxy_server.py (startup, `user_api_key_cache_ttl`),
litellm/proxy/common_utils/user_api_key_cache.py (`get_management_object_ttl`, the memory tier built with
`DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL`) and litellm/constants.py:2071:

* Auth reads every object (key, team, user, end user and the end user registry, project, organization, tag,
  object permission) cache first and never checks freshness against the database. Only a management write's
  eviction, or the cache TTL, makes a change visible.
* The TTL is 60 seconds by default. `general_settings.user_api_key_cache_ttl: <seconds>` replaces it for the
  memory tier and for the Redis tier when `enable_redis_auth_cache` attaches Redis.
* The environment variable `DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL` sets the fallback memory TTL when
  `user_api_key_cache_ttl` is not configured.

The tests change rows with plain SQL (`write_rows`), which is exactly a change no eviction announces: an
operator's script, a migration, or another service sharing the database. The Rust stores in draft PRs
#52173, #52176, #52170 and #52178 own "the memory tier" per object and must keep these TTLs.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import pytest

from tests.integration._support.client import Gateway, gateway_from_environment, held
from tests.integration._support.database import write_rows
from tests.integration._support.process import OwnedState, owned_proxy
from tests.integration.authorization.dec1_dbstores_support import (
    DEFAULT_AUTH_CACHE_TTL_SECONDS,
    assert_budget_refused,
    assert_customer_model_denied,
    assert_ok,
    chat_as,
    config_with_settings,
    customer,
    settled,
)
from tests.integration.authorization.dec1_keys_support import BLOCKED_KEY_MESSAGE, assert_denied, digest

pytestmark: Final = pytest.mark.timeout(300)

_SHORT_TTL: Final = 2
_SLACK: Final = 8


# Both proxies run on the shared database: `write_rows` changes rows in the database `DATABASE_URL` names, and an
# owned proxy defaults to a private scratch database (tests/integration/_support/process.py:65-72), where the
# out-of-band change would never be seen.
@pytest.fixture(scope="module")
def short_ttl(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("dec1-dbstores-short-ttl")
    config: Final = config_with_settings(directory, general_settings={"user_api_key_cache_ttl": _SHORT_TTL})
    with (
        gateway_from_environment() as shared,
        owned_proxy(shared, directory / "proxy", {}, config=config, state=OwnedState.SHARED_WITH_WORKER) as proxy,
    ):
        yield proxy


@pytest.fixture(scope="module")
def short_management_ttl(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("dec1-dbstores-short-management-ttl")
    with (
        gateway_from_environment() as shared,
        owned_proxy(
            shared,
            directory / "proxy",
            {"DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL": str(_SHORT_TTL)},
            state=OwnedState.SHARED_WITH_WORKER,
        ) as proxy,
    ):
        yield proxy


def _team_blocked_message(team: str) -> str:
    return f"Authentication Error, Team={team} is blocked. Update via `/team/unblock` if you're an admin."


def _block_team_out_of_band(team: str) -> None:
    write_rows('UPDATE "LiteLLM_TeamTable" SET blocked = true WHERE team_id = %s', (team,))


def test_with_a_short_ttl_an_out_of_band_team_block_is_enforced_after_the_ttl(short_ttl: Gateway) -> None:
    """With `user_api_key_cache_ttl: 2`, a team key is served, then the team row gets `blocked = true` by SQL.

    Within 2 + 8 seconds the key must answer 401 with the exact team blocked message.
    """
    with short_ttl.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        key: Final = scenario.key(team_id=team)
        assert_ok(chat_as(short_ttl, key, model))
        _block_team_out_of_band(team)
        refused: Final = settled(lambda: chat_as(short_ttl, key, model), 401, _SHORT_TTL + _SLACK)
        assert_denied(refused, status=401, error_type="auth_error", message=_team_blocked_message(team))
        short_ttl.post("/team/unblock", {"team_id": team})


def test_with_a_short_ttl_an_out_of_band_key_block_is_enforced_after_the_ttl(short_ttl: Gateway) -> None:
    """With `user_api_key_cache_ttl: 2`, a key is served, then its `LiteLLM_VerificationToken` row gets
    `blocked = true` by SQL.

    Within 10 seconds the key must answer 401 with the exact key blocked message. The key row is matched by the
    SHA-256 of the key, which is how the key store looks it up.
    """
    with short_ttl.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        assert_ok(chat_as(short_ttl, key, model))
        write_rows('UPDATE "LiteLLM_VerificationToken" SET blocked = true WHERE token = %s', (digest(key),))
        refused: Final = settled(lambda: chat_as(short_ttl, key, model), 401, _SHORT_TTL + _SLACK)
        assert_denied(refused, status=401, error_type="auth_error", message=BLOCKED_KEY_MESSAGE)
        short_ttl.post("/key/unblock", {"key": key})


def test_with_a_short_ttl_an_out_of_band_project_block_is_enforced_after_the_ttl(short_ttl: Gateway) -> None:
    """With `user_api_key_cache_ttl: 2`, a project key is served, then the project row gets `blocked = true` by SQL.

    Within 10 seconds the key must answer 401 `auth_error` (project blocked). The project cache entry uses
    `_cache_management_object`, so it honours the same TTL.
    """
    with short_ttl.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        project: Final = scenario.project(team)
        key: Final = scenario.key(team_id=team, project_id=project)
        assert_ok(chat_as(short_ttl, key, model))
        write_rows('UPDATE "LiteLLM_ProjectTable" SET blocked = true WHERE project_id = %s', (project,))
        refused: Final = settled(lambda: chat_as(short_ttl, key, model), 401, _SHORT_TTL + _SLACK)
        assert f"Project={project} is blocked" in refused.text, refused.text
        short_ttl.post("/project/update", {"project_id": project, "blocked": False})


def test_with_a_short_ttl_an_out_of_band_customer_restriction_is_enforced_after_the_ttl(short_ttl: Gateway) -> None:
    """With `user_api_key_cache_ttl: 2`, an unrestricted customer is served model B, then its row gets
    `models = ARRAY[<A>]` by SQL.

    Within 10 seconds model B must be refused with 403 `customer_model_access_denied`. Both the end user row and
    the restricted registry (which decides whether the row is read at all) expire with the TTL.
    """
    with short_ttl.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = customer(scenario)
        assert_ok(chat_as(short_ttl, key, other, user=end_user))
        write_rows('UPDATE "LiteLLM_EndUserTable" SET models = ARRAY[%s] WHERE user_id = %s', (allowed, end_user))
        assert_customer_model_denied(
            settled(lambda: chat_as(short_ttl, key, other, user=end_user), 403, _SHORT_TTL + _SLACK)
        )


def test_with_a_short_ttl_an_out_of_band_user_budget_is_enforced_after_the_ttl(short_ttl: Gateway) -> None:
    """With `user_api_key_cache_ttl: 2`, a user's personal key is served, then the user row gets
    `max_budget = 0` by SQL.

    Within 10 seconds the key must be refused with 422 `budget_exceeded`.
    """
    with short_ttl.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = scenario.user()
        key: Final = scenario.key(user_id=user)
        assert_ok(chat_as(short_ttl, key, model))
        write_rows('UPDATE "LiteLLM_UserTable" SET max_budget = 0 WHERE user_id = %s', (user,))
        assert_budget_refused(settled(lambda: chat_as(short_ttl, key, model), 422, _SHORT_TTL + _SLACK))


def test_by_default_an_out_of_band_team_block_is_not_seen_while_the_team_is_cached(gateway: Gateway) -> None:
    """On the suite proxy (default 60 second TTL), a team key is served, then the team row gets `blocked = true`
    by SQL.

    For the next 10 seconds every request is still served with 200: the team is read from the cache, not from the
    database, on every request. Then, within the 60 second TTL plus 15 seconds, the block is enforced. A store
    that reads the row on every request passes the second half and fails the first; one that never expires
    fails the second.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        key: Final = scenario.key(team_id=team)
        assert_ok(chat_as(gateway, key, model))
        _block_team_out_of_band(team)
        held(lambda: chat_as(gateway, key, model).status_code, lambda status: status == 200, holding=10, seconds=12)
        settled(lambda: chat_as(gateway, key, model), 401, DEFAULT_AUTH_CACHE_TTL_SECONDS + 15)
        gateway.post("/team/unblock", {"team_id": team})


def test_by_default_an_out_of_band_customer_restriction_is_not_seen_while_cached(gateway: Gateway) -> None:
    """On the suite proxy, an unrestricted customer is served model B, then its row gets `models = ARRAY[<A>]`
    by SQL.

    For 10 seconds model B keeps being served (the registry says the id restricts nothing, so its row is not
    even read); within 75 seconds it is refused with 403 `customer_model_access_denied`.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key()
        end_user: Final = customer(scenario)
        assert_ok(chat_as(gateway, key, other, user=end_user))
        write_rows('UPDATE "LiteLLM_EndUserTable" SET models = ARRAY[%s] WHERE user_id = %s', (allowed, end_user))
        held(
            lambda: chat_as(gateway, key, other, user=end_user).status_code,
            lambda status: status == 200,
            holding=10,
            seconds=12,
        )
        assert_customer_model_denied(
            settled(lambda: chat_as(gateway, key, other, user=end_user), 403, DEFAULT_AUTH_CACHE_TTL_SECONDS + 15)
        )


def test_the_management_object_ttl_environment_variable_shortens_the_team_cache(short_management_ttl: Gateway) -> None:
    """A proxy started with `DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL=2` and no `user_api_key_cache_ttl`
    serves a team key, then the team row gets `blocked = true` by SQL.

    UNVERIFIED: within 2 + 8 seconds the team blocked 401 is returned. Assumes the suite config does not set
    `user_api_key_cache_ttl` and that a Redis auth tier, if enabled, also expires within that window
    (`get_management_object_ttl` falls back to the constant when no default TTL is configured).
    """
    with short_management_ttl.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        key: Final = scenario.key(team_id=team)
        assert_ok(chat_as(short_management_ttl, key, model))
        _block_team_out_of_band(team)
        refused: Final = settled(lambda: chat_as(short_management_ttl, key, model), 401, _SHORT_TTL + _SLACK)
        assert_denied(refused, status=401, error_type="auth_error", message=_team_blocked_message(team))
        short_management_ttl.post("/team/unblock", {"team_id": team})
