"""Contracts for which guardrails run for a request: default_on, key, team and project assignment, and precedence.

The proxy builds the request's guardrail list in `move_guardrails_to_metadata` (litellm_pre_call_utils.py): the
key's `metadata.guardrails` come first, then the team's, then the project's, merged without duplicates, and the list
is written over whatever `metadata.guardrails` the caller sent. The body `guardrails` field is then appended, so a
caller can add guardrails but never remove an assigned one. A `default_on: true` guardrail runs on every request
unless the key's or team's admin metadata opts out (`disable_global_guardrails`, `opted_out_global_guardrails`);
the same flags sent by the caller are ignored. A team whose metadata says `{"guardrails": {"modify_guardrails":
false}}` refuses requests that try to name guardrails with 403. Key and team writes refuse unknown guardrail names.

These pins matter because auth objects (key, team, project) are moving to Rust stores: the metadata that carries
assigned guardrails and the admin opt-out flags must still reach the Python guardrail hooks with the same shape.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import JSON_OBJECT, object_value
from integration.observability.dec1_guardrails_support import (
    APPLIED_HEADER,
    PLAIN_ANSWER,
    Rig,
    boot_rig,
    chat,
    chat_text,
    error_message,
    generic,
    marker,
    messages,
    rails_called,
    refused_input,
    responses,
)
from pydantic import JsonValue

ALWAYS: Final = "dec1-at-always"
ALWAYS_TWO: Final = "dec1-at-always-two"
KEY_RAIL: Final = "dec1-at-key"
TEAM_RAIL: Final = "dec1-at-team"
PROJECT_RAIL: Final = "dec1-at-project"
EXTRA_RAIL: Final = "dec1-at-extra"
UNKNOWN: Final = "dec1-at-does-not-exist"
DEFAULT_ON: Final = frozenset({ALWAYS, ALWAYS_TWO})


def _guardrails(policy_url: str) -> list[dict[str, JsonValue]]:
    always: Final = [generic(name, "pre_call", policy_url, default_on=True) for name in (ALWAYS, ALWAYS_TWO)]
    return [
        *always,
        *(generic(name, "pre_call", policy_url) for name in (KEY_RAIL, TEAM_RAIL, PROJECT_RAIL, EXTRA_RAIL)),
    ]


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with boot_rig(tmp_path_factory.mktemp("dec1-guardrails-attachment"), _guardrails) as value:
        yield value


@pytest.fixture(autouse=True)
def _drained(rig: Rig) -> None:
    rig.drain()


def _rails(rig: Rig) -> list[str]:
    return sorted(rail for rail, _ in rails_called(rig.policy_calls()))


def _assigned(rig: Rig) -> list[str]:
    return [rail for rail in _rails(rig) if rail not in DEFAULT_ON]


def test_default_on_guardrails_run_on_a_request_that_names_none(rig: Rig) -> None:
    """Two guardrails are configured with default_on true. A plain master-key chat names none, yet the peer must
    be called once by each, and the applied header must name both."""
    response: Final = chat(rig, f"plain {marker()}")
    assert chat_text(response) == PLAIN_ANSWER
    assert _rails(rig) == sorted(DEFAULT_ON)
    assert sorted(response.headers[APPLIED_HEADER].split(",")) == sorted(DEFAULT_ON)


def test_a_default_on_guardrail_refuses_a_request_that_named_nothing(rig: Rig) -> None:
    """A plain chat carries BLOCK-IN. The first default_on guardrail refuses it: the caller gets 400 with
    "dec1 refused input by dec1-at-always" and the provider gets nothing."""
    response: Final = chat(rig, f"BLOCK-IN {marker()}")
    assert response.status_code == 400, response.text
    assert error_message(response) == refused_input(ALWAYS)
    assert rig.provider_calls() == ()


@pytest.mark.parametrize("surface", ["messages", "responses"])
def test_default_on_guardrails_run_on_messages_and_responses_too(rig: Rig, surface: str) -> None:
    """A plain request on /v1/messages or /v1/responses names no guardrail. Both default_on guardrails must still be
    called, so the Rust-served /v1/messages path is not a way around global guardrails."""
    response: Final = (
        messages(rig, f"plain {marker()}") if surface == "messages" else responses(rig, f"plain {marker()}")
    )
    assert response.status_code == 200, response.text
    assert _rails(rig) == sorted(DEFAULT_ON)


def test_a_requested_guardrail_runs_in_addition_to_the_default_on_ones(rig: Rig) -> None:
    """A chat names one non-default guardrail. The peer is called by all three and the header names all three."""
    response: Final = chat(rig, f"plus one {marker()}", extra={"guardrails": [EXTRA_RAIL]})
    assert response.status_code == 200, response.text
    assert _rails(rig) == sorted([*DEFAULT_ON, EXTRA_RAIL])
    assert sorted(response.headers[APPLIED_HEADER].split(",")) == sorted([*DEFAULT_ON, EXTRA_RAIL])


def test_a_team_admin_opt_out_skips_only_the_named_default_on_guardrail(rig: Rig) -> None:
    """The admin creates a team with metadata {"opted_out_global_guardrails": ["dec1-at-always-two"]}. A key of that
    team chats: only dec1-at-always runs, and a BLOCK-IN prompt is still refused by it."""
    team: Final = rig.scenario.team(metadata={"opted_out_global_guardrails": [ALWAYS_TWO]})
    key: Final = rig.scenario.key(team_id=team)
    assert chat(rig, f"opted out {marker()}", key=key).status_code == 200
    assert _rails(rig) == [ALWAYS]
    refused: Final = chat(rig, f"BLOCK-IN {marker()}", key=key)
    assert refused.status_code == 400, refused.text
    assert error_message(refused) == refused_input(ALWAYS)


def test_a_key_admin_disable_global_guardrails_skips_every_default_on_guardrail(rig: Rig) -> None:
    """The admin generates a key with metadata {"disable_global_guardrails": true}. A BLOCK-IN chat on that key is
    served with 200 and the peer receives nothing, because no default_on guardrail runs."""
    key: Final = rig.scenario.key(metadata={"disable_global_guardrails": True})
    response: Final = chat(rig, f"BLOCK-IN {marker()}", key=key)
    assert chat_text(response) == PLAIN_ANSWER
    assert _rails(rig) == []


@pytest.mark.parametrize(
    "flag",
    [{"disable_global_guardrails": True}, {"opted_out_global_guardrails": [ALWAYS, ALWAYS_TWO]}],
    ids=["disable_global_guardrails", "opted_out_global_guardrails"],
)
def test_a_caller_cannot_switch_off_default_on_guardrails_through_request_metadata(
    rig: Rig, flag: dict[str, JsonValue]
) -> None:
    """A plain key sends the opt-out flag in its own request `metadata`. The flags are read only from admin key and
    team metadata, so both default_on guardrails still run and a BLOCK-IN prompt is refused with 400 (read from code:
    custom_guardrail.py _get_admin_metadata)."""
    key: Final = rig.scenario.key()
    response: Final = chat(rig, f"BLOCK-IN {marker()}", key=key, extra={"metadata": flag})
    assert response.status_code == 400, response.text
    assert error_message(response) == refused_input(ALWAYS)


def test_a_key_guardrail_runs_on_that_keys_requests_only(rig: Rig) -> None:
    """One key is generated with guardrails [dec1-at-key] and another without. Only the first key's chat makes
    the peer see dec1-at-key."""
    guarded: Final = rig.scenario.key(guardrails=[KEY_RAIL])
    plain: Final = rig.scenario.key()
    assert chat(rig, f"plain key {marker()}", key=plain).status_code == 200
    assert _assigned(rig) == []
    assert chat(rig, f"guarded key {marker()}", key=guarded).status_code == 200
    assert _assigned(rig) == [KEY_RAIL]


def test_an_empty_body_list_cannot_remove_a_key_guardrail(rig: Rig) -> None:
    """A key carries guardrails [dec1-at-key] and the caller sends `guardrails: []`. The body list is appended to
    the key's list, so dec1-at-key still runs."""
    key: Final = rig.scenario.key(guardrails=[KEY_RAIL])
    assert chat(rig, f"empty list {marker()}", key=key, extra={"guardrails": []}).status_code == 200
    assert _assigned(rig) == [KEY_RAIL]


def test_a_body_guardrail_is_added_to_the_key_guardrails(rig: Rig) -> None:
    """A key carries [dec1-at-key] and the caller names dec1-at-extra in the body. Both run once each."""
    key: Final = rig.scenario.key(guardrails=[KEY_RAIL])
    assert chat(rig, f"add one {marker()}", key=key, extra={"guardrails": [EXTRA_RAIL]}).status_code == 200
    assert _assigned(rig) == sorted([KEY_RAIL, EXTRA_RAIL])


def test_a_callers_metadata_guardrails_are_replaced_by_the_key_guardrails(rig: Rig) -> None:
    """A key carries [dec1-at-key] and the caller sends `metadata.guardrails: [dec1-at-extra]`. The key's list is
    written over the caller's metadata list, so only dec1-at-key runs (read from code: litellm_pre_call_utils.py
    add_guardrails_from_key_or_team_metadata assigns, it does not extend). A Rust port that merges instead would
    change which guardrails run."""
    key: Final = rig.scenario.key(guardrails=[KEY_RAIL])
    response: Final = chat(rig, f"replaced {marker()}", key=key, extra={"metadata": {"guardrails": [EXTRA_RAIL]}})
    assert response.status_code == 200, response.text
    assert _assigned(rig) == [KEY_RAIL]


def test_the_same_guardrail_on_the_key_and_in_the_body_runs_once(rig: Rig) -> None:
    """A key carries [dec1-at-key] and the body names dec1-at-key again. The peer must be called by it once, not
    twice, because each callback runs once per hook regardless of how often it is named."""
    key: Final = rig.scenario.key(guardrails=[KEY_RAIL])
    assert chat(rig, f"twice {marker()}", key=key, extra={"guardrails": [KEY_RAIL]}).status_code == 200
    assert _assigned(rig) == [KEY_RAIL]


def test_a_team_guardrail_runs_for_its_keys(rig: Rig) -> None:
    """A team is created with guardrails [dec1-at-team]. A key of that team chats and the peer sees dec1-at-team."""
    team: Final = rig.scenario.team(guardrails=[TEAM_RAIL])
    key: Final = rig.scenario.key(team_id=team)
    assert chat(rig, f"team {marker()}", key=key).status_code == 200
    assert _assigned(rig) == [TEAM_RAIL]


def test_key_and_team_guardrails_both_run(rig: Rig) -> None:
    """A team carries [dec1-at-team] and its key carries [dec1-at-key]. Both run: the key's list does not shadow the
    team's (they are merged, key first)."""
    team: Final = rig.scenario.team(guardrails=[TEAM_RAIL])
    key: Final = rig.scenario.key(team_id=team, guardrails=[KEY_RAIL])
    assert chat(rig, f"key and team {marker()}", key=key).status_code == 200
    assert _assigned(rig) == sorted([KEY_RAIL, TEAM_RAIL])


def test_a_guardrail_named_on_key_and_team_runs_once(rig: Rig) -> None:
    """Both the team and its key carry [dec1-at-key]. The merged list names it once and the peer is called once."""
    team: Final = rig.scenario.team(guardrails=[KEY_RAIL])
    key: Final = rig.scenario.key(team_id=team, guardrails=[KEY_RAIL])
    assert chat(rig, f"dedupe {marker()}", key=key).status_code == 200
    assert _assigned(rig) == [KEY_RAIL]


def test_a_project_guardrail_runs_for_keys_in_the_project(rig: Rig) -> None:
    """A project inside a team is created with guardrails [dec1-at-project]. A key bound to that project chats and the
    peer sees dec1-at-project. UNVERIFIED: that `/project/new` stores `guardrails` in project metadata (read from
    code: _types.py NewProjectRequest moves the field into metadata) and that the key's auth carries
    project_metadata."""
    team: Final = rig.scenario.team()
    project: Final = rig.scenario.project(team, guardrails=[PROJECT_RAIL])
    key: Final = rig.scenario.key(team_id=team, project_id=project)
    assert chat(rig, f"project {marker()}", key=key).status_code == 200
    assert _assigned(rig) == [PROJECT_RAIL]


def test_key_team_and_project_guardrails_all_run(rig: Rig) -> None:
    """A team with [dec1-at-team], a project in it with [dec1-at-project], and a key in both with [dec1-at-key]. All
    three assigned guardrails run once each."""
    team: Final = rig.scenario.team(guardrails=[TEAM_RAIL])
    project: Final = rig.scenario.project(team, guardrails=[PROJECT_RAIL])
    key: Final = rig.scenario.key(team_id=team, project_id=project, guardrails=[KEY_RAIL])
    assert chat(rig, f"all three {marker()}", key=key).status_code == 200
    assert _assigned(rig) == sorted([KEY_RAIL, TEAM_RAIL, PROJECT_RAIL])


def test_a_key_guardrail_added_by_key_update_runs_on_the_next_request(rig: Rig) -> None:
    """A plain key chats (no assigned guardrail runs), then /key/update sets guardrails [dec1-at-key]. The next chat
    on the same key must run dec1-at-key, so the auth cache is invalidated on update."""
    key: Final = rig.scenario.key()
    assert chat(rig, f"before {marker()}", key=key).status_code == 200
    assert _assigned(rig) == []
    rig.gateway.post("/key/update", {"key": key, "guardrails": [KEY_RAIL]})
    assert chat(rig, f"after {marker()}", key=key).status_code == 200
    assert _assigned(rig) == [KEY_RAIL]


def test_key_generate_refuses_an_unknown_guardrail_name(rig: Rig) -> None:
    """/key/generate with guardrails naming a guardrail that is neither in config nor in the database answers 400
    with "Unknown guardrail(s): dec1-at-does-not-exist. Create the guardrail first, or check the name against
    /v2/guardrails/list." (read from code: guardrail_helpers.py validate_guardrail_names)."""
    response: Final = rig.gateway.request("POST", "/key/generate", {"guardrails": [UNKNOWN]})
    assert response.status_code == 400, response.text
    assert (
        f"Unknown guardrail(s): {UNKNOWN}. Create the guardrail first, or check the name against /v2/guardrails/list."
        in response.text
    )


def test_team_new_refuses_an_unknown_guardrail_name(rig: Rig) -> None:
    """/team/new with guardrails naming an unknown guardrail answers 400 with the same "Unknown guardrail(s)" text."""
    response: Final = rig.gateway.request(
        "POST", "/team/new", {"team_alias": f"dec1-{marker()}", "guardrails": [UNKNOWN]}
    )
    assert response.status_code == 400, response.text
    assert f"Unknown guardrail(s): {UNKNOWN}." in response.text


def test_key_generate_refuses_a_guardrails_value_that_is_not_a_list_of_names(rig: Rig) -> None:
    """/key/generate with `metadata.guardrails` set to a number answers 400 whose message starts with
    "Invalid guardrail detected - 5." and asks for a list of guardrail names (read from code: guardrail_helpers.py
    _malformed_guardrails_message)."""
    response: Final = rig.gateway.request("POST", "/key/generate", {"metadata": {"guardrails": 5}})
    assert response.status_code == 400, response.text
    assert "Invalid guardrail detected - 5. Please provide a valid list of guardrail names." in response.text


def test_a_team_without_modify_permission_refuses_requests_that_name_guardrails(rig: Rig) -> None:
    """A team is created with metadata {"guardrails": {"modify_guardrails": false}}. A key of that team gets 403
    "Your team does not have permission to modify guardrails." when it names guardrails in the body, and the
    provider is never called (read from code: auth_checks.py require_guardrail_modification_permission)."""
    team: Final = rig.scenario.team(metadata={"guardrails": {"modify_guardrails": False}})
    key: Final = rig.scenario.key(team_id=team)
    response: Final = chat(rig, f"modify {marker()}", key=key, extra={"guardrails": [EXTRA_RAIL]})
    assert response.status_code == 403, response.text
    assert "Your team does not have permission to modify guardrails." in response.text
    assert rig.provider_calls() == ()


@pytest.mark.parametrize(
    "extra",
    [
        {"metadata": {"guardrails": [EXTRA_RAIL]}},
        {"metadata": {"disable_global_guardrails": True}},
        {"litellm_metadata": {"opted_out_global_guardrails": [ALWAYS]}},
        {"metadata": '{"disable_global_guardrails": true}'},
    ],
    ids=["metadata_guardrails", "metadata_disable_flag", "litellm_metadata_opt_out", "metadata_as_json_string"],
)
def test_a_team_without_modify_permission_refuses_every_spelling_of_a_guardrail_change(
    rig: Rig, extra: dict[str, JsonValue]
) -> None:
    """The same restricted team's key tries to change guardrails through metadata, litellm_metadata, or metadata
    sent as a JSON string. Each spelling must be refused with 403 (read from code: auth_checks.py
    requests_guardrail_modification parses string containers)."""
    team: Final = rig.scenario.team(metadata={"guardrails": {"modify_guardrails": False}})
    key: Final = rig.scenario.key(team_id=team)
    response: Final = chat(rig, f"spelling {marker()}", key=key, extra=extra)
    assert response.status_code == 403, response.text


def test_a_team_without_modify_permission_still_serves_plain_requests_with_default_on_guardrails(rig: Rig) -> None:
    """The restricted team's key sends a plain chat that names no guardrail. It is served with 200 and both default_on
    guardrails still run."""
    team: Final = rig.scenario.team(metadata={"guardrails": {"modify_guardrails": False}})
    key: Final = rig.scenario.key(team_id=team)
    response: Final = chat(rig, f"plain restricted {marker()}", key=key)
    assert chat_text(response) == PLAIN_ANSWER
    assert _rails(rig) == sorted(DEFAULT_ON)


def test_a_key_info_read_returns_the_assigned_guardrails_in_metadata(rig: Rig) -> None:
    """A key generated with guardrails [dec1-at-key] is read back with /key/info. Its info.metadata.guardrails must be
    ["dec1-at-key"], the stored shape every guardrail resolver reads."""
    key: Final = rig.scenario.key(guardrails=[KEY_RAIL])
    response: Final = rig.gateway.request("GET", "/key/info", params={"key": key})
    assert response.status_code == 200, response.text
    info: Final = object_value(JSON_OBJECT.validate_json(response.content)["info"])
    assert object_value(info["metadata"])["guardrails"] == [KEY_RAIL], info
