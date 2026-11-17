"""Which fields of the project, tag, user and object permission rows take effect on a request, and how fast a
management write reaches a proxy that already cached them.

Contract, read from litellm/proxy/auth/auth_checks.py (`_run_project_checks`, `get_project_object`,
`delete_cached_project_object`, `tag_max_budget_check_for_tags`, `get_tag_objects_batch`, `get_user_object`,
`_should_check_db`, `forget_missing_user`, `get_object_permission`) and
litellm/proxy/management_endpoints/tag_management_endpoints.py `_evict_tag_cache_keys`:

* A key with `project_id` reads the project row (joined to its budget) on every request, cache first under
  `project_id:<id>`. `blocked: true` refuses with 401 `auth_error` and the message
  "Authentication Error, Project=<id> is blocked. Update via `/project/update` if you're an admin."; a non-empty
  `models` list refuses other models with 403 `project_model_access_denied`. `/project/update` evicts and
  broadcasts (LIT-3803).
* Tags named by the request (body `tags`, `metadata.tags`) or by the key's `metadata.tags` are looked up in
  `LiteLLM_TagTable` through a cached name registry; a tag without a row is ignored. `/tag/new` and `/tag/delete`
  evict the tag and the registry, `/tag/update` evicts the tag, all broadcast.
* A user that does not exist is remembered as missing for `DEFAULT_IN_MEMORY_TTL` (5 seconds) per worker;
  `/user/new` forgets that on the worker that handled it. `/user/update` evicts and broadcasts the user row,
  which is cached under the bare `user_id`.
* Object permissions (`LiteLLM_ObjectPermissionTable`, here its `search_tools` column) are cached under
  `object_permission_id:<id>`; `/key/update` and `/team/update` with `object_permission` evict and broadcast.

Draft PR #52178 moves project, tag, access group and object permission reads to Rust stores; #52170 moves users.
Not duplicated: project and tag rate limits (`test_dec1_ratelim_projects.py`, `test_dec1_ratelim_tags.py`),
project, tag and user budget arithmetic (`test_dec1_budgets_boundaries.py`), search tool refusals
(`test_search_tool_allowlist.py`).
"""

from __future__ import annotations

import uuid
from typing import Final

import httpx

from tests.integration._support.client import Gateway, Scenario, eventually, string_value
from tests.integration.authorization.dec1_dbstores_support import (
    PEER_SECONDS,
    WRITER_SECONDS,
    assert_budget_refused,
    assert_ok,
    chat_as,
    settled,
    tag,
)
from tests.integration.authorization.dec1_keys_support import assert_denied, assert_model_denied


def _project_blocked_message(project_id: str) -> str:
    return f"Authentication Error, Project={project_id} is blocked. Update via `/project/update` if you're an admin."


def _overspent(gateway: Gateway, name: str) -> str:
    """Give the tag `spend: 1` through `/tag/update`, which also seeds its spend counter
    (tag_management_endpoints.py:466,478-479). The tag check refuses only when spend is above the budget
    (`spend <= max_budget` passes, litellm/rust/crates/callbacks-budgets/src/checks.rs:333), so a `max_budget: 0` tag
    nobody spent on is served."""
    gateway.post("/tag/update", {"name": name, "spend": 1})
    return name


def _tagged(gateway: Gateway, key: str, model: str, *tags: str) -> httpx.Response:
    return chat_as(gateway, key, model, metadata={"tags": list(tags)})


def test_a_blocked_project_refuses_its_key_with_the_exact_401(gateway: Gateway) -> None:
    """A project inside a team is created with `blocked: true` and a key bound to it with `project_id`.

    The key's chat request answers 401 `auth_error` with the exact project blocked message and `param: null`.
    The plain `Exception` raised by `_run_project_checks` is wrapped as "Authentication Error, " + message.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        project: Final = scenario.project(team, blocked=True)
        key: Final = scenario.key(team_id=team, project_id=project)
        assert_denied(
            chat_as(gateway, key, model), status=401, error_type="auth_error", message=_project_blocked_message(project)
        )


def test_a_projects_model_list_refuses_another_model_with_403(gateway: Gateway) -> None:
    """A project allows only model A; its key asks for model B (allowed by the key and the team).

    The answer is 403 `project_model_access_denied` with `param: "model"` and the generic model denial message;
    model A is served.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        team: Final = scenario.team()
        project: Final = scenario.project(team, models=[allowed])
        key: Final = scenario.key(team_id=team, project_id=project)
        assert_model_denied(chat_as(gateway, key, other), other, "project_model_access_denied")
        assert_ok(chat_as(gateway, key, allowed))


def test_a_project_with_an_empty_model_list_restricts_nothing(gateway: Gateway) -> None:
    """A project is created with the default `models: []`; its key asks for any model.

    The request is served: only a non-empty project list is enforced (`len(project_object.models) > 0`).
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        project: Final = scenario.project(team)
        assert_ok(chat_as(gateway, scenario.key(team_id=team, project_id=project), model))


def test_blocking_a_warmed_project_refuses_its_key_on_both_proxies(gateway: Gateway, peer: Gateway) -> None:
    """A project key is served on both proxies, then `/project/update` sets `blocked: true`.

    Both proxies must answer the exact project blocked 401 within 5 and 10 seconds; unblocking serves again.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        project: Final = scenario.project(team)
        key: Final = scenario.key(team_id=team, project_id=project)
        for proxy in (gateway, peer):
            assert_ok(chat_as(proxy, key, model))
        gateway.post("/project/update", {"project_id": project, "blocked": True})
        for proxy, seconds in ((gateway, WRITER_SECONDS), (peer, PEER_SECONDS)):
            refused = settled(lambda proxy=proxy: chat_as(proxy, key, model), 401, seconds)
            assert_denied(refused, status=401, error_type="auth_error", message=_project_blocked_message(project))
        gateway.post("/project/update", {"project_id": project, "blocked": False})
        assert_ok(settled(lambda: chat_as(peer, key, model), 200, PEER_SECONDS))


def test_adding_a_model_to_a_warmed_project_serves_it_on_both_proxies(gateway: Gateway, peer: Gateway) -> None:
    """A project allowing model A refuses model B on both proxies, then `/project/update` sets `models: [A, B]`.

    Model B must be served on the writer within 5 seconds and on the peer within 10.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        team: Final = scenario.team()
        project: Final = scenario.project(team, models=[first])
        key: Final = scenario.key(team_id=team, project_id=project)
        for proxy in (gateway, peer):
            assert_model_denied(chat_as(proxy, key, second), second, "project_model_access_denied")
        gateway.post("/project/update", {"project_id": project, "models": [first, second]})
        assert_ok(settled(lambda: chat_as(gateway, key, second), 200, WRITER_SECONDS))
        assert_ok(settled(lambda: chat_as(peer, key, second), 200, PEER_SECONDS))


def test_a_tag_with_a_zero_budget_refuses_requests_that_carry_it(gateway: Gateway) -> None:
    """A tag is created with `max_budget: 0` and `spend: 1`; a key sends `metadata.tags: [<tag>]`.

    The request is refused with 422 `budget_exceeded`; the same key without the tag is served.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        name: Final = _overspent(gateway, tag(scenario, max_budget=0))
        assert_budget_refused(_tagged(gateway, key, model, name))
        assert_ok(chat_as(gateway, key, model))


def test_a_tag_on_the_key_metadata_is_held_to_the_tag_budget(gateway: Gateway) -> None:
    """A key carries `metadata: {"tags": [<tag with max_budget 0 and spend 1>]}` and sends no tags itself.

    The request is refused with 422: the key's tags are merged into the request before the tag budget lookup
    (`apply_key_tags_pre_auth`).
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        name: Final = _overspent(gateway, tag(scenario, max_budget=0))
        key: Final = scenario.key(metadata={"tags": [name]})
        assert_budget_refused(chat_as(gateway, key, model))


def test_a_tag_without_a_row_is_ignored(gateway: Gateway) -> None:
    """A request carries `metadata.tags: [<name never created>]`.

    The request is served with 200: a tag outside the registry costs no lookup and restricts nothing.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        assert_ok(_tagged(gateway, key, model, f"dec1-dbstores-unknown-{uuid.uuid4().hex[:8]}"))


def test_a_tag_created_after_its_name_was_used_is_enforced_on_both_proxies(gateway: Gateway, peer: Gateway) -> None:
    """A tag name is sent on both proxies while no tag row exists (both cache the registry without it), then
    `/tag/new` creates it with `max_budget: 0`.

    The tagged request must be refused with 422 on the writer within 5 seconds and on the peer within 10. A port
    that keeps the stale name registry would ignore the new tag for the whole TTL.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        name: Final = f"dec1-dbstores-tag-{uuid.uuid4().hex[:12]}"
        for proxy in (gateway, peer):
            assert_ok(_tagged(proxy, key, model, name))
        gateway.post("/tag/new", {"name": name, "max_budget": 0})
        scenario.cleanups.callback(gateway.request, "POST", "/tag/delete", {"name": name})
        assert_budget_refused(settled(lambda: _tagged(gateway, key, model, name), 422, WRITER_SECONDS))
        assert_budget_refused(settled(lambda: _tagged(peer, key, model, name), 422, PEER_SECONDS))


def test_deleting_a_zero_budget_tag_lifts_the_refusal_on_both_proxies(gateway: Gateway, peer: Gateway) -> None:
    """A tag with `max_budget: 0` and `spend: 1` is refused on both proxies, then `/tag/delete` removes it.

    The tagged request must be served on the writer within 5 seconds and on the peer within 10.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        name: Final = f"dec1-dbstores-tag-{uuid.uuid4().hex[:12]}"
        gateway.post("/tag/new", {"name": name, "max_budget": 0})
        _overspent(gateway, name)
        for proxy in (gateway, peer):
            assert_budget_refused(_tagged(proxy, key, model, name))
        gateway.post("/tag/delete", {"name": name})
        assert_ok(settled(lambda: _tagged(gateway, key, model, name), 200, WRITER_SECONDS))
        assert_ok(settled(lambda: _tagged(peer, key, model, name), 200, PEER_SECONDS))


def test_raising_a_tag_budget_readmits_it_on_both_proxies(gateway: Gateway, peer: Gateway) -> None:
    """A tag with `max_budget: 0` and `spend: 1` is refused on both proxies, then `/tag/update` sets `max_budget: 100`.

    The tagged request must be served on the writer within 5 seconds and on the peer within 10. `/tag/update`
    writes through the linked budget row (`handle_budget_for_entity`), so the cached tag must be re-read with its
    joined budget.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        name: Final = _overspent(gateway, tag(scenario, max_budget=0))
        for proxy in (gateway, peer):
            assert_budget_refused(_tagged(proxy, key, model, name))
        gateway.post("/tag/update", {"name": name, "max_budget": 100})
        assert_ok(settled(lambda: _tagged(gateway, key, model, name), 200, WRITER_SECONDS))
        assert_ok(settled(lambda: _tagged(peer, key, model, name), 200, PEER_SECONDS))


def test_a_user_created_after_their_key_was_used_is_enforced_on_both_proxies(gateway: Gateway, peer: Gateway) -> None:
    """A key is generated for a `user_id` that has no user row and is served on both proxies (each remembers the
    user as missing). `/user/new` then creates that user with `max_budget: 0`.

    The writer must refuse with 422 within 5 seconds (`forget_missing_user` runs there) and the peer within
    10 seconds (its missing-user memory lasts `DEFAULT_IN_MEMORY_TTL`, 5 seconds). A store that caches "no
    such user" for the full 60 second TTL would let the user spend past a budget set at creation.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user_id: Final = f"dec1-dbstores-late-user-{uuid.uuid4().hex}"
        # Registered before the key so it runs after the key's own cleanup: /user/delete also deletes the user's
        # keys, and the later /key/delete would then answer 404 "No keys found".
        scenario.cleanups.callback(gateway.request, "POST", "/user/delete", {"user_ids": [user_id]})
        key: Final = scenario.key(user_id=user_id)
        for proxy in (gateway, peer):
            assert_ok(chat_as(proxy, key, model))
        gateway.post("/user/new", {"user_id": user_id, "max_budget": 0, "auto_create_key": False})
        assert_budget_refused(settled(lambda: chat_as(gateway, key, model), 422, WRITER_SECONDS))
        assert_budget_refused(settled(lambda: chat_as(peer, key, model), 422, PEER_SECONDS))


def test_lowering_a_users_budget_reaches_their_warmed_personal_key_on_both_proxies(
    gateway: Gateway, peer: Gateway
) -> None:
    """A user with `max_budget: 100` has a personal key (no team) served on both proxies, then `/user/update`
    sets `max_budget: 0`.

    Both proxies must refuse with 422 within 5 and 10 seconds, and `/user/update` back to 100 readmits.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = scenario.user(max_budget=100)
        key: Final = scenario.key(user_id=user)
        for proxy in (gateway, peer):
            assert_ok(chat_as(proxy, key, model))
        gateway.post("/user/update", {"user_id": user, "max_budget": 0})
        assert_budget_refused(settled(lambda: chat_as(gateway, key, model), 422, WRITER_SECONDS))
        assert_budget_refused(settled(lambda: chat_as(peer, key, model), 422, PEER_SECONDS))
        gateway.post("/user/update", {"user_id": user, "max_budget": 100})
        assert_ok(settled(lambda: chat_as(peer, key, model), 200, PEER_SECONDS))


def test_a_personal_key_of_a_user_with_a_zero_budget_is_refused_on_its_first_request(gateway: Gateway) -> None:
    """A user is created with `max_budget: 0` and a personal key that was never used.

    The first request is refused with 422: the cold user lookup returns `max_budget` and `spend`.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = scenario.user(max_budget=0)
        assert_budget_refused(chat_as(gateway, scenario.key(user_id=user), model))


def _search_tool(gateway: Gateway, scenario: Scenario) -> str:
    name: Final = f"dec1-dbstores-tool-{uuid.uuid4().hex[:10]}"
    created: Final = gateway.post(
        "/search_tools",
        {
            "search_tool": {
                "search_tool_name": name,
                "litellm_params": {
                    "search_provider": "exa_ai",
                    "api_key": "synthetic-dec1-dbstores-key",
                    "api_base": f"{gateway.upstream_url}/search",
                },
                "search_tool_info": {"description": "dec1 dbstores integration tool"},
            }
        },
    )
    scenario.cleanups.callback(gateway.request, "DELETE", f"/search_tools/{string_value(created['search_tool_id'])}")
    return name


def _visible_tools(proxy: Gateway, key: str, candidates: frozenset[str]) -> frozenset[str]:
    response: Final = proxy.request("GET", "/search_tools/list", key=key)
    assert response.status_code == 200, response.text
    return frozenset(entry["search_tool_name"] for entry in response.json()["search_tools"]) & candidates


def test_widening_a_warmed_keys_object_permission_reaches_both_proxies(gateway: Gateway, peer: Gateway) -> None:
    """A key whose `object_permission.search_tools` names tool A lists only A on both proxies, then `/key/update`
    sets `object_permission: {"search_tools": [A, B]}`.

    `/search_tools/list` for that key must show both A and B on the writer within 5 seconds and on the peer
    within 10. The permission row is cached apart from the key (`object_permission_id:<id>`).
    """
    with gateway.scenario() as scenario:
        first: Final = _search_tool(gateway, scenario)
        second: Final = _search_tool(gateway, scenario)
        both: Final = frozenset({first, second})
        key: Final = scenario.key(user_id=scenario.user(), object_permission={"search_tools": [first]})
        for proxy in (gateway, peer):
            assert _visible_tools(proxy, key, both) == {first}
        gateway.post("/key/update", {"key": key, "object_permission": {"search_tools": [first, second]}})
        eventually(lambda: _visible_tools(gateway, key, both), lambda seen: seen == both, seconds=WRITER_SECONDS)
        eventually(lambda: _visible_tools(peer, key, both), lambda seen: seen == both, seconds=PEER_SECONDS)


def test_narrowing_a_teams_object_permission_reaches_a_warmed_team_key_on_both_proxies(
    gateway: Gateway, peer: Gateway
) -> None:
    """A team whose `object_permission.search_tools` names A and B has a member key that lists both on both
    proxies; `/team/update` then sets `object_permission: {"search_tools": [A]}`.

    The listing must shrink to A on the writer within 5 seconds and on the peer within 10. The team row and its
    permission row are separate cache entries; both must be refreshed.
    """
    with gateway.scenario() as scenario:
        first: Final = _search_tool(gateway, scenario)
        second: Final = _search_tool(gateway, scenario)
        both: Final = frozenset({first, second})
        team: Final = scenario.team(object_permission={"search_tools": [first, second]})
        key: Final = scenario.key(team_id=team, user_id=scenario.member(team))
        for proxy in (gateway, peer):
            assert _visible_tools(proxy, key, both) == both
        gateway.post("/team/update", {"team_id": team, "object_permission": {"search_tools": [first]}})
        eventually(lambda: _visible_tools(gateway, key, both), lambda seen: seen == {first}, seconds=WRITER_SECONDS)
        eventually(lambda: _visible_tools(peer, key, both), lambda seen: seen == {first}, seconds=PEER_SECONDS)


def test_a_key_created_with_an_object_permission_applies_it_on_the_first_request(gateway: Gateway) -> None:
    """A key is created with `object_permission: {"search_tools": [A]}` while B also exists; it was never used.

    Its first `/search_tools/list` shows only A: the cold key lookup must follow `object_permission_id` to the
    permission row (the key view does not carry the permission columns).
    """
    with gateway.scenario() as scenario:
        first: Final = _search_tool(gateway, scenario)
        second: Final = _search_tool(gateway, scenario)
        key: Final = scenario.key(user_id=scenario.user(), object_permission={"search_tools": [first]})
        assert _visible_tools(gateway, key, frozenset({first, second})) == {first}
