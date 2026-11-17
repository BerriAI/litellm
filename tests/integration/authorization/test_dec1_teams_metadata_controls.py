"""Which fields of a team's `metadata` take effect on a request made with one of its keys.

Contract, read from litellm/proxy/litellm_pre_call_utils.py (`add_litellm_data_to_request`,
`add_key_level_controls`, `add_team_and_project_level_controls`, `_merge_spend_logs_metadata`, `merge_tags`,
`get_inherited_tags`, `_key_or_team_metadata_flag_is_true`, `_get_enforced_params`) and
litellm/proxy/spend_tracking/spend_tracking_utils.py (`get_logging_payload`):

* The team row's `metadata` reaches the request through the team lookup (`valid_token.team_metadata`), which
  draft PR #52176 moves to `RustTeamStore`. A store that drops or retypes a JSON field silently changes billing
  labels, tags or opt ins, with no error anywhere.
* `spend_logs_metadata`: the request's own values win, then the key's, then the team's (team values are defaults).
  A non object value on the team is ignored.
* `tags`: request, key, team and project tags are merged without repeats; a team `tags` value that is not a list
  is ignored.
* Three opt ins are honoured on the key OR the team, and only when the stored value is the JSON boolean `true`:
  `allow_client_mock_response`, `allow_client_pricing_override`, `allow_client_message_redaction_opt_out`.
* Some key metadata controls have no team equivalent: `enforced_params` and `disable_fallbacks` are read only from
  the key's metadata, so the same field on the team is ignored. A port that "unifies" key and team metadata
  handling would change these.
* The spend row carries `team_id` as a column and `user_api_key_team_alias` / `user_api_key_team_id` in its
  `metadata`.

Every test uses the suite `gateway` fixture and models from `scenario.model()` served by the scripted upstream.
Already pinned elsewhere and not repeated: team tags next to key tags (`test_dec1_keys_request_shaping.py`), team
guardrails and `opted_out_global_guardrails` (`observability/test_dec1_guardrails_attachment.py`), team logging
callbacks (`observability/`), `require_trace_id` (`test_dec1_dbstores_team_org_and_key_lookups.py`).
"""

from __future__ import annotations

from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, list_value
from tests.integration.authorization.dec1_teams_support import (
    ATTEMPTED_FALLBACKS_HEADER,
    UPSTREAM_COMPLETION_TOKENS,
    UPSTREAM_CONTENT,
    UPSTREAM_PROMPT_TOKENS,
    answer_text,
    chat_as,
    marker,
    row_metadata,
    scripted_model,
    spend_rows_of,
    upstream_bodies,
)

_INPUT_PRICE: Final = 0.001
_OUTPUT_PRICE: Final = 0.002
_OVERRIDE_SPEND: Final = UPSTREAM_PROMPT_TOKENS * _INPUT_PRICE + UPSTREAM_COMPLETION_TOKENS * _OUTPUT_PRICE


def _spend_logs_metadata(gateway: Gateway, key: str) -> JsonValue:
    return row_metadata(spend_rows_of(gateway, key)[0]).get("spend_logs_metadata")


def _request_tags(gateway: Gateway, key: str) -> list[JsonValue]:
    return list(list_value(spend_rows_of(gateway, key)[0]["request_tags"]))


def test_team_spend_logs_metadata_lands_on_the_spend_row_of_a_team_key(gateway: Gateway) -> None:
    """A team has `metadata.spend_logs_metadata: {"cost_center": "team-cc"}`; its key (no metadata) calls chat.

    The spend row's `metadata.spend_logs_metadata` must be exactly `{"cost_center": "team-cc"}`. Read from
    `add_team_and_project_level_controls`, which merges team values into the request metadata on every request.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(metadata={"spend_logs_metadata": {"cost_center": "team-cc"}})
        key: Final = scenario.key(team_id=team)
        answer_text(chat_as(gateway, key, model))
        assert _spend_logs_metadata(gateway, key) == {"cost_center": "team-cc"}


def test_a_requests_spend_logs_metadata_wins_over_the_teams_on_the_same_field(gateway: Gateway) -> None:
    """The team has `spend_logs_metadata: {"cost_center": "team", "owner": "team"}`; the request sends
    `metadata.spend_logs_metadata: {"cost_center": "request"}`.

    The spend row must hold `{"cost_center": "request", "owner": "team"}`: team values are defaults only
    (`_merge_spend_logs_metadata` returns `{**team, **request}`).
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(metadata={"spend_logs_metadata": {"cost_center": "team", "owner": "team"}})
        key: Final = scenario.key(team_id=team)
        answer_text(chat_as(gateway, key, model, metadata={"spend_logs_metadata": {"cost_center": "request"}}))
        assert _spend_logs_metadata(gateway, key) == {"cost_center": "request", "owner": "team"}


def test_a_keys_spend_logs_metadata_wins_over_its_teams_on_the_same_field(gateway: Gateway) -> None:
    """The team has `spend_logs_metadata: {"cost_center": "team", "owner": "team"}` and the key has
    `metadata.spend_logs_metadata: {"cost_center": "key"}`; the request sends none.

    The spend row must hold `{"cost_center": "key", "owner": "team"}`. Key controls run first
    (`add_key_level_controls`) and the team merge then treats everything already present as the request's value.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(metadata={"spend_logs_metadata": {"cost_center": "team", "owner": "team"}})
        key: Final = scenario.key(team_id=team, metadata={"spend_logs_metadata": {"cost_center": "key"}})
        answer_text(chat_as(gateway, key, model))
        assert _spend_logs_metadata(gateway, key) == {"cost_center": "key", "owner": "team"}


def test_request_key_and_team_spend_logs_metadata_resolve_in_that_order(gateway: Gateway) -> None:
    """The team sets fields a, b and c, the key sets b and c, the request sets c only, each to its own source name.

    The spend row must hold `{"a": "team", "b": "key", "c": "request"}`: one field per level proves the full
    precedence request > key > team in a single row.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(metadata={"spend_logs_metadata": {"a": "team", "b": "team", "c": "team"}})
        key: Final = scenario.key(team_id=team, metadata={"spend_logs_metadata": {"b": "key", "c": "key"}})
        answer_text(chat_as(gateway, key, model, metadata={"spend_logs_metadata": {"c": "request"}}))
        assert _spend_logs_metadata(gateway, key) == {"a": "team", "b": "key", "c": "request"}


def test_a_team_spend_logs_metadata_that_is_not_an_object_is_ignored(gateway: Gateway) -> None:
    """The team stores `spend_logs_metadata: "team-cc"` (a string); its key calls chat with no metadata.

    The request is served and the spend row's `metadata.spend_logs_metadata` is null, because
    `_merge_spend_logs_metadata` only reads dicts and returns None when neither side is one. A store that parses
    JSON strings inside the metadata column would invent an object here.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(metadata={"spend_logs_metadata": "team-cc"})
        key: Final = scenario.key(team_id=team)
        answer_text(chat_as(gateway, key, model))
        assert _spend_logs_metadata(gateway, key) is None


def test_request_key_team_and_project_tags_all_land_once_on_the_spend_row(gateway: Gateway) -> None:
    """A project inside a team; request, key, team and project each carry one own tag and all share one tag.

    The spend row's `request_tags` must contain the four own tags and the shared tag exactly once
    (`merge_tags` skips repeats; `get_inherited_tags` uses `dict.fromkeys`). The expected order is request, key,
    team, project, but only membership and uniqueness are asserted because baggage and header tags may be merged
    in a different pass.
    """
    shared: Final = f"shared-{marker()}"
    own: Final = {source: f"{source}-{marker()}" for source in ("request", "key", "team", "project")}
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(metadata={"tags": [own["team"], shared]})
        project: Final = scenario.project(team, metadata={"tags": [own["project"], shared]})
        key: Final = scenario.key(team_id=team, project_id=project, metadata={"tags": [own["key"], shared]})
        answer_text(chat_as(gateway, key, model, metadata={"tags": [own["request"], shared]}))
        tags: Final = _request_tags(gateway, key)
        assert set(own.values()) | {shared} <= set(tags), tags
        assert tags.count(shared) == 1, tags


def test_a_team_tags_value_that_is_not_a_list_adds_no_tag(gateway: Gateway) -> None:
    """The team stores `metadata.tags: "<tag>"` (a string, not a list); its key calls chat.

    The request is served and the spend row's `request_tags` must not contain `<tag>`, nor any of its
    characters as separate tags: `add_team_and_project_level_controls` passes the value on only when it is a list.
    """
    tag: Final = f"team-{marker()}"
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(metadata={"tags": tag})
        key: Final = scenario.key(team_id=team)
        answer_text(chat_as(gateway, key, model))
        tags: Final = _request_tags(gateway, key)
        assert tag not in tags and "t" not in tags, tags


def test_a_team_tag_reaches_the_spend_row_of_every_key_of_the_team(gateway: Gateway) -> None:
    """A team has `metadata.tags: [<tag>]` and two keys with no metadata of their own; each key calls chat once.

    Both spend rows must contain `<tag>` in `request_tags`, because the tag comes from the team row, not from a copy
    on the key.
    """
    tag: Final = f"team-{marker()}"
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(metadata={"tags": [tag]})
        keys: Final = (scenario.key(team_id=team), scenario.key(team_id=team))
        for key in keys:
            answer_text(chat_as(gateway, key, model))
        for key in keys:
            assert tag in _request_tags(gateway, key)


def test_a_team_tag_added_by_team_update_reaches_the_next_request(gateway: Gateway) -> None:
    """A team key is served once with no team tags, then `/team/update` sets `metadata: {"tags": [<tag>]}`.

    The next request's spend row must contain `<tag>`; the team update evicts the cached team object so the warmed
    key reads the new metadata. Only the last spend row is checked.
    """
    tag: Final = f"team-{marker()}"
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        key: Final = scenario.key(team_id=team)
        answer_text(chat_as(gateway, key, model))
        gateway.post("/team/update", {"team_id": team, "metadata": {"tags": [tag]}})
        answer_text(chat_as(gateway, key, model))
        rows: Final = spend_rows_of(gateway, key, at_least=2)
        assert tag in list_value(rows[-1]["request_tags"]), rows[-1]


def test_team_metadata_enforced_params_are_not_enforced(gateway: Gateway) -> None:
    """The team stores `metadata.enforced_params: ["user"]`; its key (no metadata) sends chat without `user`.

    The request is served with 200: `_get_enforced_params` reads `general_settings.enforced_params`, service
    account settings and the KEY's `metadata.enforced_params` only. Contrast `test_dec1_keys_request_shaping.py`,
    where the same list on the key refuses with 400. A port that reads team metadata here would start refusing.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(metadata={"enforced_params": ["user"]})
        key: Final = scenario.key(team_id=team)
        assert answer_text(chat_as(gateway, key, model)) == UPSTREAM_CONTENT


def test_a_team_allow_client_mock_response_lets_its_keys_mock(gateway: Gateway) -> None:
    """The team stores `metadata.allow_client_mock_response: true`; its key has no metadata of its own and sends
    `mock_response: "<text>"`.

    The answer's content must be `<text>` and the scripted upstream must receive no request carrying the prompt:
    `_key_or_team_metadata_flag_is_true` checks the key's metadata and then the team's.
    """
    text: Final = marker()
    mocked: Final = f"mocked-{marker()}"
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(metadata={"allow_client_mock_response": True})
        key: Final = scenario.key(team_id=team)
        assert answer_text(chat_as(gateway, key, model, text=text, mock_response=mocked)) == mocked
        assert upstream_bodies(gateway, text) == []


def test_a_string_true_allow_client_mock_response_on_the_team_does_not_let_keys_mock(gateway: Gateway) -> None:
    """The team stores `metadata.allow_client_mock_response: "true"` (a string); its key sends `mock_response`.

    `mock_response` is stripped (the flag must be `is True`), so the answer is the upstream's canned content and the
    upstream receives exactly one request with the prompt, which does not carry `mock_response`.
    """
    text: Final = marker()
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(metadata={"allow_client_mock_response": "true"})
        key: Final = scenario.key(team_id=team)
        assert answer_text(chat_as(gateway, key, model, text=text, mock_response="never")) == UPSTREAM_CONTENT
        bodies: Final = upstream_bodies(gateway, text)
        assert len(bodies) == 1 and "mock_response" not in bodies[0], bodies


def test_a_key_flag_set_to_false_does_not_cancel_its_teams_mock_opt_in(gateway: Gateway) -> None:
    """The team stores `allow_client_mock_response: true` and its key stores `allow_client_mock_response: false`.

    The key's `mock_response` is still honoured: the flag is an OR over key and team metadata, not a key override.
    A port that resolves the flag with key over team precedence would answer from the upstream instead.
    """
    text: Final = marker()
    mocked: Final = f"mocked-{marker()}"
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(metadata={"allow_client_mock_response": True})
        key: Final = scenario.key(team_id=team, metadata={"allow_client_mock_response": False})
        assert answer_text(chat_as(gateway, key, model, text=text, mock_response=mocked)) == mocked
        assert upstream_bodies(gateway, text) == []


def test_a_team_allow_client_pricing_override_prices_its_keys_requests_with_the_body_prices(gateway: Gateway) -> None:
    """The team stores `metadata.allow_client_pricing_override: true`; its key sends `input_cost_per_token: 0.001`
    and `output_cost_per_token: 0.002` in the body.

    The spend row must cost exactly 20 * 0.001 + 20 * 0.002 = 0.06 (the scripted upstream reports 20 prompt and 20
    completion tokens). `_strip_client_pricing_overrides` is skipped when the key OR its team opts in.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(metadata={"allow_client_pricing_override": True})
        key: Final = scenario.key(team_id=team)
        answer_text(
            chat_as(gateway, key, model, input_cost_per_token=_INPUT_PRICE, output_cost_per_token=_OUTPUT_PRICE)
        )
        spend: Final = spend_rows_of(gateway, key)[0]["spend"]
        assert isinstance(spend, (int, float)) and spend == pytest.approx(_OVERRIDE_SPEND), spend


def test_without_the_team_opt_in_body_prices_do_not_change_spend(gateway: Gateway) -> None:
    """The same body prices from a key whose team has no `allow_client_pricing_override`.

    The prices are stripped before the call, so the spend row must not cost 0.06; it is priced from the deployment
    (`openai/gpt-4o-mini` in the cost map), far below 0.06 for 40 tokens.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        key: Final = scenario.key(team_id=team)
        answer_text(
            chat_as(gateway, key, model, input_cost_per_token=_INPUT_PRICE, output_cost_per_token=_OUTPUT_PRICE)
        )
        spend: Final = spend_rows_of(gateway, key)[0]["spend"]
        assert isinstance(spend, (int, float)) and spend < _OVERRIDE_SPEND / 10, spend


def test_team_metadata_disable_fallbacks_does_not_switch_off_request_fallbacks(gateway: Gateway) -> None:
    """The team stores `metadata.disable_fallbacks: true`; the primary deployment answers 500 and the request names
    a healthy fallback in `fallbacks`.

    The request is served with 200 and `x-litellm-attempted-fallbacks: 1`, because `disable_fallbacks` is read only
    from the KEY's metadata (`add_key_level_controls`). The next test shows the key form refusing.
    """
    with gateway.scenario() as scenario:
        primary: Final = scripted_model(scenario, [500, 500, 500], num_retries=0)
        backup: Final = scenario.model()
        team: Final = scenario.team(metadata={"disable_fallbacks": True})
        key: Final = scenario.key(team_id=team)
        response: Final = chat_as(gateway, key, primary, fallbacks=[backup])
        assert answer_text(response) == UPSTREAM_CONTENT
        assert response.headers.get(ATTEMPTED_FALLBACKS_HEADER) == "1", dict(response.headers)


def test_key_metadata_disable_fallbacks_switches_off_request_fallbacks(gateway: Gateway) -> None:
    """The key itself stores `metadata.disable_fallbacks: true`, in a team without it; the primary answers 500.

    `data["disable_fallbacks"] = True` stops the router from trying the named fallback, so the caller sees the
    primary's 500 and the fallback deployment is never asked. This is the contrast for the team test above.
    """
    text: Final = marker()
    with gateway.scenario() as scenario:
        primary: Final = scripted_model(scenario, [500, 500, 500], num_retries=0)
        backup: Final = scenario.model()
        team: Final = scenario.team()
        key: Final = scenario.key(team_id=team, metadata={"disable_fallbacks": True})
        response: Final = chat_as(gateway, key, primary, text=text, fallbacks=[backup])
        assert response.status_code == 500, response.text
        assert len(upstream_bodies(gateway, text)) == 1


def test_a_team_key_spend_row_names_the_team_id_and_alias(gateway: Gateway) -> None:
    """A team with alias `<alias>` has a key that calls chat once.

    The spend row's `team_id` column must equal the team id and its `metadata` must carry
    `user_api_key_team_id: <team>` and `user_api_key_team_alias: <alias>` (`_get_spend_logs_metadata`).
    """
    alias: Final = f"alias-{marker()}"
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(team_alias=alias)
        key: Final = scenario.key(team_id=team)
        answer_text(chat_as(gateway, key, model))
        row: Final = spend_rows_of(gateway, key)[0]
        metadata: Final = row_metadata(row)
        assert row["team_id"] == team, row
        assert metadata.get("user_api_key_team_id") == team, metadata
        assert metadata.get("user_api_key_team_alias") == alias, metadata


def test_a_key_without_a_team_writes_an_empty_team_id_and_no_team_alias(gateway: Gateway) -> None:
    """A key with no team calls chat once.

    The spend row's `team_id` column is the empty string (`team_id=metadata.get("user_api_key_team_id", "") or ""`
    in `get_logging_payload`) and `metadata.user_api_key_team_alias` is null. UNVERIFIED: a database writer could
    turn the empty string into NULL; check `LiteLLM_SpendLogs.team_id` rows of personal keys if this fails.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        answer_text(chat_as(gateway, key, model))
        row: Final = spend_rows_of(gateway, key)[0]
        assert row["team_id"] == "", row
        assert row_metadata(row).get("user_api_key_team_alias") is None, row


def test_a_renamed_team_alias_reaches_the_next_spend_row(gateway: Gateway) -> None:
    """A team key is served once under alias `<old>`, then `/team/update` sets `team_alias: <new>`.

    A request made after the update must write `metadata.user_api_key_team_alias: <new>`. UNVERIFIED: the key's
    cached auth object carries `team_alias` from the key/team join, and `user_api_key_auth.py:3510` resyncs only
    `team_metadata` from the fresh team lookup (the project branch below it does resync `project_alias`). If
    `/team/update` does not also evict the team's key cache entries, the old alias is written for up to the 60 second
    auth cache TTL, which would be a stale attribution bug worth a `pytest.skip("BUG: ...")`.
    """
    old: Final = f"old-{marker()}"
    new: Final = f"new-{marker()}"
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(team_alias=old)
        key: Final = scenario.key(team_id=team)
        answer_text(chat_as(gateway, key, model))
        gateway.post("/team/update", {"team_id": team, "team_alias": new})
        answer_text(chat_as(gateway, key, model))
        rows: Final = spend_rows_of(gateway, key, at_least=2)
        assert row_metadata(rows[0]).get("user_api_key_team_alias") == old, rows[0]
        assert row_metadata(rows[-1]).get("user_api_key_team_alias") == new, rows[-1]
