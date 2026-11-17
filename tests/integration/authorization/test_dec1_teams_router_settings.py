"""A team's `router_settings` as a per request override, and team attribution of requests served by a fallback.

Contract, read from litellm/proxy/proxy_server.py (`_get_hierarchical_router_settings`,
`_parse_router_settings_value`), litellm/proxy/common_request_processing.py (`hierarchical_router_settings_updates`,
`_resolve_per_request_model_group_alias`) and litellm/proxy/route_llm_request.py (`_merge_router_settings_override`,
`_PER_REQUEST_ROUTER_SETTINGS`):

* The key's own non empty `router_settings` win; otherwise the team's are read through the cached `get_team_object`
  lookup that #52176 moves to `RustTeamStore`. Global router settings stay on the Router.
* Only `fallbacks`, `context_window_fallbacks`, `content_policy_fallbacks`, `num_retries`, `timeout`,
  `model_group_retry_policy`, `routing_strategy` and `enable_tag_filtering` are copied into the request, and only
  when the request body does not already set that field: a caller's own `num_retries` wins over the team's.
* `model_group_alias` from the team rewrites the requested model before the call, after checking the key may call
  the target.
* A request served by a fallback is still the team key's request: the success spend row names the team, and the
  team's `spend` grows by that row's cost.

Uses the suite `gateway`. Failures are scripted per deployment through the upstream's `/__scripts/<model>` control.
Already pinned and not repeated: team fallbacks for embeddings and images (`routing/`), a key's settings shadowing the
team's (`routing/`), a fallback outside the team model list being skipped (`test_dec1_modelaccess_fallbacks.py`).
"""

from __future__ import annotations

import json
import uuid
from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.database import write_rows
from tests.integration.authorization.dec1_teams_support import (
    ATTEMPTED_FALLBACKS_HEADER,
    UPSTREAM_CONTENT,
    WRITER_SECONDS,
    answer_text,
    chat_as,
    marker,
    row_metadata,
    scripted_model,
    settled,
    spend_rows_of,
    team_row,
    upstream_bodies,
)


def _success_rows(gateway: Gateway, key: str) -> list[dict[str, object]]:
    rows: Final = eventually(
        lambda: [row for row in spend_rows_of(gateway, key) if row["status"] == "success"],
        lambda found: len(found) >= 1,
        seconds=70,
    )
    return [dict(row) for row in rows]


def test_a_team_num_retries_of_zero_stops_the_default_retry(gateway: Gateway) -> None:
    """A team has `router_settings: {"num_retries": 0}`; its key calls a deployment scripted to answer 500 then 200.

    The caller sees the 500 and the upstream received exactly one request for the prompt: the team's value replaces
    the Router's default of 2 retries for this request. A port that ignores team router settings would retry and
    answer 200. UNVERIFIED: assumes the suite Router keeps the documented default `num_retries` of 2
    (litellm/router.py:144) since `proxy_config.yaml` sets none.
    """
    text: Final = marker()
    with gateway.scenario() as scenario:
        model: Final = scripted_model(scenario, [500, 200])
        team: Final = scenario.team(router_settings={"num_retries": 0})
        key: Final = scenario.key(team_id=team)
        response: Final = chat_as(gateway, key, model, text=text)
        assert response.status_code == 500, response.text
        assert len(upstream_bodies(gateway, text)) == 1


def test_a_requests_num_retries_wins_over_the_teams(gateway: Gateway) -> None:
    """The same team (`num_retries: 0`) and a deployment scripted 500 then 200; the request body sends
    `num_retries: 1`.

    The request is served with 200 after two upstream calls: `_merge_router_settings_override` copies a team field
    only when the body does not already carry it.
    """
    text: Final = marker()
    with gateway.scenario() as scenario:
        model: Final = scripted_model(scenario, [500, 200])
        team: Final = scenario.team(router_settings={"num_retries": 0})
        key: Final = scenario.key(team_id=team)
        assert answer_text(chat_as(gateway, key, model, text=text, num_retries=1)) == UPSTREAM_CONTENT
        assert len(upstream_bodies(gateway, text)) == 2


def test_a_key_of_another_team_keeps_the_default_retries(gateway: Gateway) -> None:
    """Team one has `num_retries: 0`; a key of team two (no router settings) calls a deployment scripted 500 then 200.

    The request is served with 200 after a retry: a team's router settings are per request, never written onto the
    shared Router. Same default retry assumption as the first test.
    """
    text: Final = marker()
    with gateway.scenario() as scenario:
        model: Final = scripted_model(scenario, [500, 200])
        scenario.team(router_settings={"num_retries": 0})
        other: Final = scenario.team()
        key: Final = scenario.key(team_id=other)
        assert answer_text(chat_as(gateway, key, model, text=text)) == UPSTREAM_CONTENT
        assert len(upstream_bodies(gateway, text)) == 2


def test_team_fallbacks_serve_a_failing_chat_model_and_report_one_attempted_fallback(gateway: Gateway) -> None:
    """A team has `router_settings: {"fallbacks": [{<primary>: [<backup>]}], "num_retries": 0}`; the primary answers
    500 and the backup is healthy.

    The team key's chat is served with the upstream's content and `x-litellm-attempted-fallbacks: 1`.
    """
    with gateway.scenario() as scenario:
        primary: Final = scripted_model(scenario, [500, 500, 500])
        backup: Final = scenario.model()
        team: Final = scenario.team(router_settings={"fallbacks": [{primary: [backup]}], "num_retries": 0})
        key: Final = scenario.key(team_id=team)
        response: Final = chat_as(gateway, key, primary)
        assert answer_text(response) == UPSTREAM_CONTENT
        assert response.headers.get(ATTEMPTED_FALLBACKS_HEADER) == "1", dict(response.headers)


def test_a_request_served_by_a_team_fallback_is_billed_to_the_team(gateway: Gateway) -> None:
    """The same team fallback setup; the key makes one call that the backup serves.

    Exactly one success spend row exists for the key; it names the team in `team_id`, carries the team alias in
    `metadata.user_api_key_team_alias`, and its `model_group` is the backup's group. UNVERIFIED: whether the failed
    primary attempt also writes a `failure` row; only success rows are counted.
    """
    alias: Final = f"alias-{marker()}"
    with gateway.scenario() as scenario:
        primary: Final = scripted_model(scenario, [500, 500, 500])
        backup: Final = scenario.model()
        team: Final = scenario.team(
            team_alias=alias, router_settings={"fallbacks": [{primary: [backup]}], "num_retries": 0}
        )
        key: Final = scenario.key(team_id=team)
        answer_text(chat_as(gateway, key, primary))
        rows: Final = _success_rows(gateway, key)
        assert len(rows) == 1, rows
        assert rows[0]["team_id"] == team, rows
        assert rows[0]["model_group"] == backup, rows
        assert row_metadata(rows[0]).get("user_api_key_team_alias") == alias, rows


def test_team_spend_grows_by_the_cost_of_the_fallback_row(gateway: Gateway) -> None:
    """The same team fallback setup; after one call served by the backup, the team row's `spend` equals the sum of
    the key's spend rows.

    A port that charges the primary group, or charges the failed attempt as well, would make the two disagree.
    """
    with gateway.scenario() as scenario:
        primary: Final = scripted_model(scenario, [500, 500, 500])
        backup: Final = scenario.model()
        team: Final = scenario.team(router_settings={"fallbacks": [{primary: [backup]}], "num_retries": 0})
        key: Final = scenario.key(team_id=team)
        answer_text(chat_as(gateway, key, primary))
        rows: Final = _success_rows(gateway, key)
        expected: Final = sum(float(str(row["spend"])) for row in spend_rows_of(gateway, key))
        assert expected > 0, rows
        eventually(
            lambda: float(str(team_row(gateway, team)["spend"])),
            lambda spend: spend == pytest.approx(expected),
            seconds=70,
        )


def test_team_fallbacks_added_by_team_update_reach_a_warmed_key(gateway: Gateway) -> None:
    """A team key is refused with the primary's 500 (team `num_retries: 0`, no fallbacks); `/team/update` then sets
    `router_settings: {"fallbacks": [{<primary>: [<backup>]}], "num_retries": 0}`.

    Within 5 seconds the same key is served by the backup. The team object is evicted on update and re-read by
    `_get_hierarchical_router_settings`; a store that caches team router settings separately would keep failing.
    """
    with gateway.scenario() as scenario:
        primary: Final = scripted_model(scenario, [500] * 64)
        backup: Final = scenario.model()
        team: Final = scenario.team(router_settings={"num_retries": 0})
        key: Final = scenario.key(team_id=team)
        assert chat_as(gateway, key, primary).status_code == 500
        gateway.post(
            "/team/update",
            {"team_id": team, "router_settings": {"fallbacks": [{primary: [backup]}], "num_retries": 0}},
        )
        settled(lambda: chat_as(gateway, key, primary), 200, WRITER_SECONDS)


def test_a_team_model_group_alias_routes_the_request_to_its_target(gateway: Gateway) -> None:
    """A team has `router_settings: {"model_group_alias": {<alias>: <target>}}` and no model list; its key calls
    `<alias>`, a name no deployment serves.

    The request is served with the upstream's content and the spend row's `model_group` is `<target>`
    (`_resolve_per_request_model_group_alias`). A key outside the team calling `<alias>` is the contrast in the next
    test.
    """
    alias: Final = f"dec1-teams-alias-{uuid.uuid4().hex[:8]}"
    with gateway.scenario() as scenario:
        target: Final = scenario.model()
        team: Final = scenario.team(router_settings={"model_group_alias": {alias: target}})
        key: Final = scenario.key(team_id=team)
        assert answer_text(chat_as(gateway, key, alias)) == UPSTREAM_CONTENT
        assert spend_rows_of(gateway, key)[0]["model_group"] == target


def test_a_team_model_group_alias_is_invisible_to_a_key_of_another_team(gateway: Gateway) -> None:
    """Team one maps `<alias>` to `<target>`; a key of team two (no router settings) calls `<alias>`.

    The request is not served (4xx, the "model not found" or invalid model answer) and the upstream receives
    nothing: team aliases are resolved per request for that team only. UNVERIFIED: the exact status, 400 from the
    router's invalid model group versus 404; only `400 <= status < 500` is asserted.
    """
    alias: Final = f"dec1-teams-alias-{uuid.uuid4().hex[:8]}"
    text: Final = marker()
    with gateway.scenario() as scenario:
        target: Final = scenario.model()
        scenario.team(router_settings={"model_group_alias": {alias: target}})
        other: Final = scenario.team()
        key: Final = scenario.key(team_id=other)
        response: Final = chat_as(gateway, key, alias, text=text)
        assert 400 <= response.status_code < 500, response.text
        assert upstream_bodies(gateway, text) == []


def test_team_router_settings_stored_as_json_text_still_apply(gateway: Gateway) -> None:
    """The team's `router_settings` column is overwritten with SQL to hold the JSON text `"{\\"num_retries\\": 0}"`
    (a JSON string, as older writers stored it); its key calls a deployment scripted 500 then 200.

    The caller sees the 500 and only one upstream call: `_parse_router_settings_value` parses a string as YAML or
    JSON. A Rust store that only accepts a JSON object column would silently fall back to the default retries.
    UNVERIFIED: the team object is read fresh because the key was never used; the cache is not involved.
    """
    text: Final = marker()
    with gateway.scenario() as scenario:
        model: Final = scripted_model(scenario, [500, 200])
        team: Final = scenario.team()
        write_rows(
            'UPDATE "LiteLLM_TeamTable" SET router_settings = %s::jsonb WHERE team_id = %s',
            (json.dumps(json.dumps({"num_retries": 0})), team),
        )
        key: Final = scenario.key(team_id=team)
        response: Final = chat_as(gateway, key, model, text=text)
        assert response.status_code == 500, response.text
        assert len(upstream_bodies(gateway, text)) == 1
