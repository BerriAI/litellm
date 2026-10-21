"""Renaming a database model carries every ``models`` allowlist that named it over to the new name.

Keys, teams, organizations, projects and users store public model names, and access groups store them as
``access_model_names``. A caller restricted through any one of them must reach the renamed model under its
new name, lose the old one, and see only the new name in ``/v1/models``.
"""

import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
from typing import Final

import httpx
import pytest
from pydantic import BaseModel, JsonValue, TypeAdapter

from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, Wire, wire_server

_PROVIDER_MODEL: Final = "gpt-4o-mini"
_PROVIDER_API_KEY: Final = "integration-provider-key"
_CALLER_KINDS: Final = ("key", "team", "organization", "project", "user", "access_group")
_JSON_BODY: Final = TypeAdapter(dict[str, JsonValue])


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


class _Error(BaseModel):
    message: str
    type: str
    param: str | None
    code: str | None


@dataclass(frozen=True, slots=True)
class _RenameCall:
    kind: str
    status_code: int
    error: _Error
    requests: tuple[Request, ...]


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
            "SELECT models FROM \"LiteLLM_VerificationToken\" WHERE token = encode(sha256(%s::bytea), 'hex')",
            (holder.key,),
        )
    return read_rows(
        f'SELECT models FROM "{holder.table}" WHERE "{holder.id_column}" = %s',  # noqa: S608  # table and column come from the fixed _holders tuple, never from input
        (holder.object_id,),
    )


def _chat(gateway: Gateway, model: str, caller: _Holder) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": f"after rename {caller.kind}"}]},
        key=caller.key,
    )


def _expected_provider_body(text: str) -> dict[str, JsonValue]:
    return {"model": _PROVIDER_MODEL, "messages": [{"role": "user", "content": text}]}


def _is_model_discovery(request: Request) -> bool:
    return (request.method, request.target) == ("GET", "/v1/models")


def _provider_calls(wire: Wire) -> tuple[Request, ...]:
    return tuple(entry for entry in wire.drain() if not _is_model_discovery(entry))


def _rename_reply(request: Request) -> Reply:
    if _is_model_discovery(request):
        return Reply(body=json.dumps({"object": "list", "data": []}).encode())
    assert (request.method, request.target) == ("POST", "/v1/chat/completions"), request
    assert request.headers["authorization"] == f"Bearer {_PROVIDER_API_KEY}", request
    assert _JSON_BODY.validate_json(request.body) in tuple(
        _expected_provider_body(f"after rename {kind}") for kind in _CALLER_KINDS
    ), request.body
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-rename",
                "object": "chat.completion",
                "created": 1,
                "model": _PROVIDER_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "renamed"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 20, "total_tokens": 40},
            }
        ).encode()
    )


def _listed(gateway: Gateway, key: str) -> list[str]:
    response: Final = gateway.request("GET", "/v1/models", key=key)
    assert response.status_code == 200, response.text
    entries: Final = response.json()["data"]
    assert isinstance(entries, list), response.text
    return sorted(string_value(object_value(entry)["id"]) for entry in entries)


def _assert_listing(gateway: Gateway, caller: _Holder, *, served: str, gone: str | None) -> None:
    def converged(listed: list[str]) -> bool:
        if caller.listing_is_scoped:
            return listed == [served]
        # Project and user allowlists gate calls but not /v1/models, which then lists every proxy model.
        return served in listed and gone not in listed

    eventually(lambda: _listed(gateway, caller.key), converged)


def _served(gateway: Gateway, model: str, caller: _Holder) -> httpx.Response:
    served: Final = eventually(
        lambda: _chat(gateway, model, caller), lambda r: r.status_code == 200, return_last_on_timeout=True
    )
    assert served.status_code == 200, f"{caller.kind}: {served.text}"
    return served


def _baseline_call(gateway: Gateway, model: str, caller: _Holder, wire: Wire) -> tuple[Request, ...]:
    served: Final = _served(gateway, model, caller)
    assert _Completion.model_validate_json(served.content) == _Completion(model=model, usage=_Usage(total_tokens=40)), (
        f"{caller.kind}: {served.text}"
    )
    _assert_listing(gateway, caller, served=model, gone=None)
    requests: Final = _provider_calls(wire)
    assert len(requests) == 1, f"{caller.kind}: expected one provider request, received {requests!r}"
    return requests


def _renamed_call(
    gateway: Gateway,
    old: str,
    new: str,
    caller: _Holder,
    wire: Wire,
) -> _RenameCall:
    served: Final = _served(gateway, new, caller)
    assert _Completion.model_validate_json(served.content) == _Completion(model=new, usage=_Usage(total_tokens=40)), (
        f"{caller.kind}: {served.text}"
    )
    requests: Final = _provider_calls(wire)
    assert len(requests) == 1, f"{caller.kind}: expected one provider request, received {requests!r}"
    refused: Final = _chat(gateway, old, caller)
    assert refused.status_code == 403, f"{caller.kind}: {refused.text}"
    error: Final = _Error.model_validate(object_value(refused.json())["error"])
    assert _provider_calls(wire) == (), f"{caller.kind}: the old name reached the provider"
    _assert_listing(gateway, caller, served=new, gone=old)
    return _RenameCall(caller.kind, refused.status_code, error, requests)


@pytest.mark.parametrize("rename_style", ("patch", "legacy_post"))
def test_renamed_model_is_reachable_by_its_new_name_through_every_allowlist_that_named_it(
    gateway: Gateway, rename_style: str
) -> None:
    with wire_server(_rename_reply) as wire, gateway.scenario() as scenario:
        old: Final = scenario.model(api_base=f"{wire.url}/v1", api_key=_PROVIDER_API_KEY)
        model_id: Final = _model_id(gateway, old)
        holders: Final = _holders(scenario, old)
        access_team: Final = scenario.team(models=["no-default-models"])
        access_key: Final = scenario.key(team_id=access_team)
        with _access_group(gateway, access_team, old) as access_group_id:
            callers: Final = (*holders, _Holder("access_group", "", "", access_group_id, access_key))
            baseline_requests: Final = tuple(_baseline_call(gateway, old, caller, wire) for caller in callers)

            new: Final = f"integration-renamed-{uuid.uuid4().hex}"
            renamed: Final = (
                gateway.request("PATCH", f"/model/{model_id}/update", {"model_name": new})
                if rename_style == "patch"
                else gateway.request(
                    "POST",
                    "/model/update",
                    {"model_name": new, "model_info": {"id": model_id}, "litellm_params": {}},
                )
            )
            assert renamed.status_code == 200, renamed.text
            assert read_rows('SELECT model_name FROM "LiteLLM_ProxyModelTable" WHERE model_id = %s', (model_id,)) == [
                {"model_name": new}
            ]

            results: Final = tuple(_renamed_call(gateway, old, new, caller, wire) for caller in callers)
            for holder in holders:
                assert _allowlist(holder) == [{"models": [new]}], holder.kind
            group: Final = gateway.request("GET", f"/v1/access_group/{access_group_id}")
            assert group.status_code == 200, group.text
            assert _AccessGroup.model_validate_json(group.content) == _AccessGroup(
                access_group_id=access_group_id, access_model_names=[new]
            ), group.text

            denied_message: Final = (
                f"The requested model '{old}' is not available for this API key, or the model name is invalid. "
                "Check the models available to you and try again."
            )
            expected_denials: Final = {
                "key": _Error(message=denied_message, type="key_model_access_denied", param="model", code="403"),
                "team": _Error(message=denied_message, type="team_model_access_denied", param="model", code="403"),
                "organization": _Error(
                    message=denied_message, type="team_model_access_denied", param="model", code="403"
                ),
                "project": _Error(message=denied_message, type="project_model_access_denied", param="model", code="403"),
                "user": _Error(message=denied_message, type="user_model_access_denied", param="model", code="403"),
                "access_group": _Error(
                    message=denied_message, type="team_model_access_denied", param="model", code="403"
                ),
            }
            assert tuple(result.status_code for result in results) == (403,) * len(callers)
            assert {result.kind: result.error for result in results} == expected_denials, json.dumps(
                {result.kind: result.error.model_dump() for result in results}
            )
            provider_requests: Final = tuple(
                chain.from_iterable(baseline_requests)
            ) + tuple(chain.from_iterable(result.requests for result in results))
            expected_request: Final = (
                "POST",
                "/v1/chat/completions",
                f"Bearer {_PROVIDER_API_KEY}",
            )
            expected_requests: Final = tuple(
                (*expected_request, _expected_provider_body(f"after rename {caller.kind}"))
                for caller in callers
            ) * 2
            assert tuple(
                (
                    request.method,
                    request.target,
                    request.headers["authorization"],
                    _JSON_BODY.validate_json(request.body),
                )
                for request in provider_requests
            ) == expected_requests
