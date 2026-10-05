"""Renaming a database model carries every ``models`` allowlist that named it over to the new name.

Keys, teams, organizations, projects and users store public model names, and access groups store them as
``access_model_names``. A caller restricted through any one of them must reach the renamed model under its
new name, lose the old one, and see only the new name in ``/v1/models``.
"""

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final

import httpx
from pydantic import BaseModel

from integration._support.client import Gateway, Scenario, object_value, string_value
from integration._support.database import read_rows


@dataclass(frozen=True, slots=True)
class _Holder:
    kind: str
    table: str
    id_column: str
    object_id: str
    key: str
    listing_is_scoped: bool = True


class _AccessGroup(BaseModel):
    access_group_id: str
    access_model_names: list[str]


class _Usage(BaseModel):
    total_tokens: int


class _Completion(BaseModel):
    model: str
    usage: _Usage


def _model_id(gateway: Gateway, name: str) -> str:
    entries: Final = gateway.get("/model/info")["data"]
    assert isinstance(entries, list)
    return next(
        string_value(object_value(entry["model_info"])["id"])
        for entry in map(object_value, entries)
        if entry["model_name"] == name
    )


@contextmanager
def _access_group(gateway: Gateway, team_id: str, model: str) -> Iterator[str]:
    created: Final = gateway.request(
        "POST",
        "/v1/access_group",
        {
            "access_group_name": f"integration-{uuid.uuid4().hex}",
            "access_model_names": [model],
            "assigned_team_ids": [team_id],
        },
    )
    assert created.status_code == 201, created.text
    identity: Final = string_value(created.json()["access_group_id"])
    try:
        yield identity
    finally:
        deleted: Final = gateway.request("DELETE", f"/v1/access_group/{identity}")
        assert deleted.status_code == 204, deleted.text


def _holders(scenario: Scenario, old: str) -> tuple[_Holder, ...]:
    team: Final = scenario.team(models=[old])
    organization: Final = scenario.organization(models=[old])
    org_team: Final = scenario.team(organization_id=organization, models=[old])
    project_team: Final = scenario.team()
    project: Final = scenario.project(project_team, models=[old])
    user: Final = scenario.user(user_role="internal_user", models=[old])
    key: Final = scenario.key(models=[old])
    return (
        _Holder("key", "LiteLLM_VerificationToken", "key_alias", "", key),
        _Holder("team", "LiteLLM_TeamTable", "team_id", team, scenario.key(team_id=team)),
        _Holder(
            "organization",
            "LiteLLM_OrganizationTable",
            "organization_id",
            organization,
            scenario.key(team_id=org_team, organization_id=organization),
        ),
        _Holder(
            "project",
            "LiteLLM_ProjectTable",
            "project_id",
            project,
            scenario.key(team_id=project_team, project_id=project),
            listing_is_scoped=False,
        ),
        _Holder("user", "LiteLLM_UserTable", "user_id", user, scenario.key(user_id=user), listing_is_scoped=False),
    )


def _allowlist(holder: _Holder) -> list[dict[str, object]]:
    if holder.kind == "key":
        return read_rows(
            'SELECT models FROM "LiteLLM_VerificationToken" WHERE token = encode(sha256(%s::bytea), \'hex\')',
            (holder.key,),
        )
    return read_rows(
        f'SELECT models FROM "{holder.table}" WHERE "{holder.id_column}" = %s',  # noqa: S608
        (holder.object_id,),
    )


def _chat(gateway: Gateway, model: str, key: str) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": "after rename"}]},
        key=key,
    )


def _listed(gateway: Gateway, key: str) -> list[str]:
    response: Final = gateway.request("GET", "/v1/models", key=key)
    assert response.status_code == 200, response.text
    entries: Final = response.json()["data"]
    assert isinstance(entries, list), response.text
    return sorted(string_value(object_value(entry)["id"]) for entry in entries)


def _assert_listing(gateway: Gateway, caller: _Holder, *, served: str, gone: str | None) -> None:
    listed: Final = _listed(gateway, caller.key)
    if caller.listing_is_scoped:
        assert listed == [served], caller.kind
        return
    # Project and user allowlists gate calls but not /v1/models, which then lists every proxy model.
    assert served in listed and gone not in listed, (caller.kind, listed)


def test_renamed_model_is_reachable_by_its_new_name_through_every_allowlist_that_named_it(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        old: Final = scenario.model()
        model_id: Final = _model_id(gateway, old)
        holders: Final = _holders(scenario, old)
        access_team: Final = scenario.team(models=["no-default-models"])
        access_key: Final = scenario.key(team_id=access_team)
        with _access_group(gateway, access_team, old) as access_group_id:
            callers: Final = (*holders, _Holder("access_group", "", "", access_group_id, access_key))
            for caller in callers:
                assert _chat(gateway, old, caller.key).status_code == 200, caller.kind
                _assert_listing(gateway, caller, served=old, gone=None)

            new: Final = f"integration-renamed-{uuid.uuid4().hex}"
            renamed: Final = gateway.request("PATCH", f"/model/{model_id}/update", {"model_name": new})
            assert renamed.status_code == 200, renamed.text
            assert read_rows(
                'SELECT model_name FROM "LiteLLM_ProxyModelTable" WHERE model_id = %s', (model_id,)
            ) == [{"model_name": new}]

            for holder in holders:
                assert _allowlist(holder) == [{"models": [new]}], holder.kind
            group: Final = gateway.request("GET", f"/v1/access_group/{access_group_id}")
            assert group.status_code == 200, group.text
            assert _AccessGroup.model_validate_json(group.content) == _AccessGroup(
                access_group_id=access_group_id, access_model_names=[new]
            ), group.text

            for caller in callers:
                served = _chat(gateway, new, caller.key)
                assert served.status_code == 200, f"{caller.kind}: {served.text}"
                assert _Completion.model_validate_json(served.content) == _Completion(
                    model=new, usage=_Usage(total_tokens=40)
                ), f"{caller.kind}: {served.text}"
                refused = _chat(gateway, old, caller.key)
                assert refused.status_code in (401, 403), f"{caller.kind}: {refused.text}"
                _assert_listing(gateway, caller, served=new, gone=old)
