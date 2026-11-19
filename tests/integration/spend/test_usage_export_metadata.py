import csv
import io
import json
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import pytest
from integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows, write_rows
from integration._support.process import owned_proxy_process
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.shares_worker_state

_ROWS: Final = TypeAdapter(list[dict[str, JsonValue]])
_LONG: Final = "x" * 5120
_REDACTED: Final = "REDACTED"
_KEY_METADATA: Final[Mapping[str, JsonValue]] = {
    "cost_center": "cc-7",
    "labels": ["blue", "gold"],
    "max_tokens": 5,
    "author": "ana",
    "keyboard": "qwerty",
    "notes": _LONG,
    "webhook_token": "whsec_1",
    "signing-secret": "sig_1",
    "apiKey": "camel_1",
    "OpenAIAPIKey": "acronym_1",
    "apikey": "unseparated_1",
    "clientSecret": "client_1",
    "db_password": 7,
    "auth_token": ["a", "b"],
    "private_key": {"pem": "p"},
    "session_token": "",
    "refresh_token": None,
    "access_token": _LONG,
    "langfuse": {
        "host": "https://langfuse.invalid",
        "apiKey": "lf_1",
        "logging": [{"callback_name": "langfuse"}],
        "scim_roles": ["r"],
    },
    "routes": [{"name": "a", "api_key": "route_1", "callback_settings": {"otel": {"headers": "h"}}}],
    "secret_manager_settings": {"vault": "v"},
    "scim_enterprise": {"employeeNumber": "E-1"},
    "apiKeys": "k_1",
    "tokens": "t_1",
    "database_url": "db.invalid",
    "note": "sk-proj-1234567890abcdef",
    "owners": ["Bearer abc.def.ghi", "ops"],
}
_KEY_METADATA_EXPORTED: Final[Mapping[str, JsonValue]] = {
    "cost_center": "cc-7",
    "labels": ["blue", "gold"],
    "max_tokens": 5,
    "author": "ana",
    "keyboard": "qwerty",
    "notes": _LONG,
    "webhook_token": _REDACTED,
    "signing-secret": _REDACTED,
    "apiKey": _REDACTED,
    "OpenAIAPIKey": _REDACTED,
    "apikey": _REDACTED,
    "clientSecret": _REDACTED,
    "db_password": _REDACTED,
    "auth_token": _REDACTED,
    "private_key": _REDACTED,
    "session_token": _REDACTED,
    "refresh_token": None,
    "access_token": _REDACTED,
    "langfuse": {"host": "https://langfuse.invalid", "apiKey": _REDACTED},
    "routes": [{"name": "a", "api_key": _REDACTED}],
    "apiKeys": _REDACTED,
    "tokens": _REDACTED,
    "database_url": _REDACTED,
    "note": _REDACTED,
    "owners": [_REDACTED, "ops"],
}
_USER_METADATA: Final[Mapping[str, JsonValue]] = {
    "department": "research",
    "ldap_password": "pw_1",
    "APIKEY": "upper_1",
    "profile": {"sessionToken": "st_1", "scim_entitlements": ["e"], "level": 3},
    "logging": [{"callback_name": "langfuse"}],
    "callback_settings": {"otel": {"headers": "h"}},
    "scim_enterprise": {"employeeNumber": "E-1"},
    "scim_entitlements": ["ent"],
    "scim_roles": ["role"],
    "slackWebhookUrl": "https://hooks.slack.invalid/T/B/X",
    "bio": "postgres://admin:pw@db.invalid:5432/app",
}
_USER_METADATA_EXPORTED: Final[Mapping[str, JsonValue]] = {
    "department": "research",
    "slackWebhookUrl": _REDACTED,
    "bio": _REDACTED,
    "ldap_password": _REDACTED,
    "APIKEY": _REDACTED,
    "profile": {"sessionToken": _REDACTED, "level": 3},
}
_BASE_COLUMNS: Final = (
    "date",
    "entity_id",
    "entity_alias",
    "api_key",
    "key_alias",
    "user_id",
    "user_email",
    "model",
    "spend",
    "flat_cost",
    "prompt_tokens",
    "completion_tokens",
    "api_requests",
    "successful_requests",
    "failed_requests",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)
_SECRET_VALUES: Final = (
    "whsec_1",
    "sig_1",
    "camel_1",
    "acronym_1",
    "unseparated_1",
    "client_1",
    "lf_1",
    "route_1",
    "pw_1",
    "upper_1",
    "st_1",
    "E-1",
    "langfuse.invalid/otel",
    "callback_name",
)


@dataclass(frozen=True, slots=True)
class _Usage:
    model: str
    user_id: str
    team_id: str
    organization_id: str
    tag: str
    end_user_id: str
    agent_id: str
    key: str
    digest: str
    response_id: str
    dates: Mapping[str, str]

    def routes(self) -> tuple[tuple[str, str, str], ...]:
        return (
            ("/user", "user_id", self.user_id),
            ("/team", "team_ids", self.team_id),
            ("/tag", "tags", self.tag),
            ("/organization", "organization_ids", self.organization_id),
            ("/customer", "end_user_ids", self.end_user_id),
            ("/agent", "agent_ids", self.agent_id),
        )


def _post_ok(gateway: Gateway, method: str, path: str, body: Mapping[str, JsonValue]) -> None:
    response: Final = gateway.request(method, path, body)
    assert response.status_code == 200, response.text


def _set_metadata(table: str, column: str, identity: str, metadata: JsonValue) -> None:
    assert table in ("LiteLLM_VerificationToken", "LiteLLM_UserTable")
    if table == "LiteLLM_VerificationToken":
        write_rows(
            'UPDATE "LiteLLM_VerificationToken" SET metadata = %s::jsonb WHERE token = %s',
            (json.dumps(metadata), identity),
        )
    else:
        write_rows(
            'UPDATE "LiteLLM_UserTable" SET metadata = %s::jsonb WHERE user_id = %s', (json.dumps(metadata), identity)
        )
    stored: Final = read_rows(f'SELECT metadata FROM "{table}" WHERE {column} = %s', (identity,))
    assert [row["metadata"] for row in stored] == [metadata]


def _chat(gateway: Gateway, usage_model: str, key: str, tag: str, end_user_id: str, *, stream: bool = False) -> str:
    marker: Final = f"usage export {uuid.uuid4().hex}"
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        JSON_OBJECT.validate_python(
            {
                "model": usage_model,
                "messages": [{"role": "user", "content": marker}],
                "metadata": {"tags": [tag]},
                "user": end_user_id,
                "stream": stream,
            }
        ),
        key=key,
    )
    assert response.status_code == 200, response.text
    observed: Final = httpx.get(f"{gateway.upstream_url}/__observations", timeout=5, trust_env=False).text
    assert observed.count(marker) == 1, "the upstream did not receive this request exactly once"
    if not stream:
        return string_value(object_value(response.json())["id"])
    chunks: Final = tuple(
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    )
    assert chunks, response.text
    return string_value(chunks[0]["id"])


def _spend_rows(response_ids: tuple[str, ...]) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT request_id, api_key FROM "LiteLLM_SpendLogs" WHERE request_id = ANY(%s) ORDER BY request_id',
        (list(response_ids),),  # pyright: ignore[reportArgumentType]  # psycopg adapts the list to a text array
    )


def _seed(
    gateway: Gateway,
    scenario: Scenario,
    *,
    key_metadata: JsonValue = None,
    user_metadata: JsonValue = None,
    with_user: bool = True,
    key_outlives_test: bool = True,
) -> _Usage:
    model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
    organization: Final = gateway.post("/organization/new", {"organization_alias": f"integration-{uuid.uuid4().hex}"})
    organization_id: Final = string_value(organization["organization_id"])
    scenario.cleanups.callback(
        _post_ok, gateway, "DELETE", "/organization/delete", {"organization_ids": [organization_id]}
    )
    team_id: Final = scenario.team(organization_id=organization_id)
    user_id: Final = scenario.user(user_role="internal_user")
    tag: Final = f"integration-{uuid.uuid4().hex}"
    gateway.post("/tag/new", {"name": tag})
    scenario.cleanups.callback(_post_ok, gateway, "POST", "/tag/delete", {"name": tag})
    end_user_id: Final = f"integration-{uuid.uuid4().hex}"
    gateway.post("/end_user/new", {"user_id": end_user_id})
    scenario.cleanups.callback(_post_ok, gateway, "POST", "/end_user/delete", {"user_ids": [end_user_id]})
    agent: Final = gateway.post(
        "/v1/agents",
        {
            "agent_name": f"integration-{uuid.uuid4().hex}",
            "agent_card_params": {
                "protocolVersion": "0.3",
                "name": "integration",
                "description": "integration agent",
                "url": "http://127.0.0.1:1/agent",
                "version": "1",
                "capabilities": {},
                "defaultInputModes": ["text"],
                "defaultOutputModes": ["text"],
                "skills": [],
            },
        },
    )
    agent_id: Final = string_value(agent["agent_id"])
    scenario.cleanups.callback(lambda: gateway.request("DELETE", f"/v1/agents/{agent_id}").raise_for_status())
    owner: Final[dict[str, JsonValue]] = {"user_id": user_id} if with_user else {}
    key_fields: Final[dict[str, JsonValue]] = {
        "models": [model],
        "team_id": team_id,
        "organization_id": organization_id,
        "agent_id": agent_id,
        **owner,
    }
    key: Final = (
        scenario.key(**key_fields)
        if key_outlives_test
        else string_value(gateway.post("/key/generate", key_fields)["key"])
    )
    digest: Final = sha256(key.encode()).hexdigest()
    if key_metadata is not None:
        _set_metadata("LiteLLM_VerificationToken", "token", digest, key_metadata)
    if user_metadata is not None:
        _set_metadata("LiteLLM_UserTable", "user_id", user_id, user_metadata)
    response_id: Final = _chat(gateway, model, key, tag, end_user_id)
    today: Final = datetime.now(timezone.utc).date()
    dates: Final = {
        "start_date": (today - timedelta(days=1)).isoformat(),
        "end_date": (today + timedelta(days=1)).isoformat(),
        "timezone": "0",
    }
    usage: Final = _Usage(
        model, user_id, team_id, organization_id, tag, end_user_id, agent_id, key, digest, response_id, dates
    )
    _await_usage(usage)
    return usage


def _await_usage(usage: _Usage) -> None:
    tables: Final = (
        ("LiteLLM_DailyUserSpend", "user_id", usage.user_id),
        ("LiteLLM_DailyTeamSpend", "team_id", usage.team_id),
        ("LiteLLM_DailyTagSpend", "tag", usage.tag),
        ("LiteLLM_DailyOrganizationSpend", "organization_id", usage.organization_id),
        ("LiteLLM_DailyEndUserSpend", "end_user_id", usage.end_user_id),
        ("LiteLLM_DailyAgentSpend", "agent_id", usage.agent_id),
    )
    for table, column, identity in tables:
        if table == "LiteLLM_DailyUserSpend" and not read_rows(
            'SELECT user_id FROM "LiteLLM_VerificationToken" WHERE token = %s AND user_id = %s',
            (usage.digest, usage.user_id),
        ):
            continue
        eventually(
            lambda table=table, column=column, identity=identity: read_rows(
                f'SELECT api_requests FROM "{table}" WHERE "{column}" = %s AND api_key = %s', (identity, usage.digest)
            ),
            lambda rows: [row["api_requests"] for row in rows] == [1],
            seconds=70,
        )
    assert _spend_rows((usage.response_id,)) == [{"request_id": usage.response_id, "api_key": usage.digest}]


def _export(
    gateway: Gateway,
    usage: _Usage,
    route: tuple[str, str, str],
    *,
    key: str | None = None,
    export_type: str = "daily_with_keys",
    file_format: str = "json",
    include_metadata: bool = True,
) -> httpx.Response:
    prefix, parameter, identity = route
    return gateway.request(
        "GET",
        f"{prefix}/daily/activity/export",
        key=key,
        params={
            **usage.dates,
            parameter: identity,
            "export_type": export_type,
            "format": file_format,
            "include_metadata": "true" if include_metadata else "false",
        },
    )


def _json_rows(response: httpx.Response) -> list[dict[str, JsonValue]]:
    assert response.status_code == 200, response.text
    return _ROWS.validate_json(response.content)


def _metadata_of(rows: list[dict[str, JsonValue]]) -> list[tuple[JsonValue, JsonValue]]:
    return [(row.get("key_metadata", "absent"), row.get("user_metadata", "absent")) for row in rows]


def _assert_no_secret(text: str) -> None:
    assert [value for value in _SECRET_VALUES if value in text] == [], text[:2000]


@pytest.mark.parametrize("route_index", range(6), ids=("user", "team", "tag", "organization", "customer", "agent"))
def test_admin_export_carries_redacted_key_and_user_metadata_as_json_and_csv(
    gateway: Gateway, scenario: Scenario, route_index: int
) -> None:
    usage: Final = _seed(gateway, scenario, key_metadata=dict(_KEY_METADATA), user_metadata=dict(_USER_METADATA))
    route: Final = usage.routes()[route_index]

    as_json: Final = _export(gateway, usage, route)
    rows: Final = _json_rows(as_json)
    assert [(row["entity_id"], row["api_key"], row["user_id"], row["api_requests"]) for row in rows] == [
        (route[2], usage.digest, usage.user_id, 1)
    ], as_json.text
    assert tuple(rows[0])[: len(_BASE_COLUMNS)] == _BASE_COLUMNS, as_json.text
    assert _metadata_of(rows) == [(dict(_KEY_METADATA_EXPORTED), dict(_USER_METADATA_EXPORTED))], as_json.text
    _assert_no_secret(as_json.text)

    as_csv: Final = _export(gateway, usage, route, file_format="csv")
    assert as_csv.status_code == 200, as_csv.text
    header, *lines = tuple(csv.reader(io.StringIO(as_csv.text)))
    assert tuple(header) == (*_BASE_COLUMNS, "key_metadata", "user_metadata"), as_csv.text
    assert [(line[1], line[3], line[5], line[12]) for line in lines] == [
        (route[2], usage.digest, usage.user_id, "1")
    ], as_csv.text
    assert [(json.loads(line[-2]), json.loads(line[-1])) for line in lines] == [
        (dict(_KEY_METADATA_EXPORTED), dict(_USER_METADATA_EXPORTED))
    ], as_csv.text
    assert lines[0][-2] == json.dumps(json.loads(lines[0][-2]), separators=(",", ":")), as_csv.text
    _assert_no_secret(as_csv.text)


def test_admin_viewer_sees_metadata_and_other_exports_keep_the_default_shape(
    gateway: Gateway, scenario: Scenario
) -> None:
    usage: Final = _seed(gateway, scenario, key_metadata=dict(_KEY_METADATA), user_metadata=dict(_USER_METADATA))
    viewer_key: Final = scenario.key(user_id=scenario.user(user_role="proxy_admin_viewer"))
    team_admin: Final = scenario.member(usage.team_id, role="admin")
    team_admin_key: Final = scenario.key(user_id=team_admin)
    user_route, team_route, *_ = usage.routes()

    viewer: Final = _export(gateway, usage, user_route, key=viewer_key)
    assert _metadata_of(_json_rows(viewer)) == [(dict(_KEY_METADATA_EXPORTED), dict(_USER_METADATA_EXPORTED))], (
        viewer.text
    )

    not_opted_in: Final = _export(gateway, usage, user_route, include_metadata=False)
    assert _metadata_of(_json_rows(not_opted_in)) == [("absent", "absent")], not_opted_in.text
    _assert_no_secret(not_opted_in.text)

    own: Final = _export(gateway, usage, user_route, key=usage.key)
    own_rows: Final = _json_rows(own)
    assert [row["api_key"] for row in own_rows] == [usage.digest], own.text
    assert _metadata_of(own_rows) == [("absent", "absent")], own.text

    own_csv: Final = _export(gateway, usage, user_route, key=usage.key, file_format="csv")
    assert own_csv.status_code == 200, own_csv.text
    header, line = tuple(csv.reader(io.StringIO(own_csv.text)))
    assert (tuple(header), len(line)) == (_BASE_COLUMNS, len(_BASE_COLUMNS)), own_csv.text

    team: Final = _export(gateway, usage, team_route, key=team_admin_key)
    team_rows: Final = _json_rows(team)
    assert [row["api_key"] for row in team_rows] == [usage.digest], team.text
    assert _metadata_of(team_rows) == [("absent", "absent")], team.text
    for response in (own, own_csv, team):
        _assert_no_secret(response.text)
        assert "cc-7" not in response.text and "research" not in response.text, response.text


def test_export_types_populate_only_the_metadata_their_grouping_joins(gateway: Gateway, scenario: Scenario) -> None:
    usage: Final = _seed(gateway, scenario, key_metadata=dict(_KEY_METADATA), user_metadata=dict(_USER_METADATA))
    route: Final = usage.routes()[1]
    observed: Final = {
        export_type: _metadata_of(_json_rows(_export(gateway, usage, route, export_type=export_type)))
        for export_type in ("daily", "daily_with_models", "daily_with_users", "daily_with_keys")
    }
    assert observed == {
        "daily": [({}, {})],
        "daily_with_models": [({}, {})],
        "daily_with_users": [({}, dict(_USER_METADATA_EXPORTED))],
        "daily_with_keys": [(dict(_KEY_METADATA_EXPORTED), dict(_USER_METADATA_EXPORTED))],
    }


def test_deleted_key_metadata_is_read_from_the_deleted_key_record(gateway: Gateway, scenario: Scenario) -> None:
    usage: Final = _seed(
        gateway,
        scenario,
        key_metadata=dict(_KEY_METADATA),
        user_metadata=dict(_USER_METADATA),
        key_outlives_test=False,
    )
    scenario.delete_key(usage.key)
    archived: Final = read_rows(
        'SELECT metadata FROM "LiteLLM_DeletedVerificationToken" WHERE token = %s', (usage.digest,)
    )
    assert [row["metadata"] for row in archived] == [dict(_KEY_METADATA)]
    for route in usage.routes():
        response = _export(gateway, usage, route)
        rows = _json_rows(response)
        assert [(row["api_key"], row["user_id"]) for row in rows] == [(usage.digest, usage.user_id)], response.text
        assert _metadata_of(rows) == [(dict(_KEY_METADATA_EXPORTED), dict(_USER_METADATA_EXPORTED))], response.text
        _assert_no_secret(response.text)


def test_unauthenticated_and_invalid_export_requests_are_rejected_without_metadata(
    gateway: Gateway, scenario: Scenario
) -> None:
    usage: Final = _seed(gateway, scenario, key_metadata=dict(_KEY_METADATA), user_metadata=dict(_USER_METADATA))
    outsider_key: Final = scenario.key(user_id=scenario.user(user_role="internal_user"))
    for route in usage.routes():
        prefix, parameter, identity = route
        anonymous = gateway.client.get(
            f"{prefix}/daily/activity/export",
            params={
                **usage.dates,
                parameter: identity,
                "export_type": "daily_with_keys",
                "format": "json",
                "include_metadata": "true",
            },
        )
        unknown = _export(gateway, usage, route, key="sk-not-a-key")
        assert (anonymous.status_code, unknown.status_code) == (401, 401), (anonymous.text, unknown.text)
        outsider = _export(gateway, usage, route, key=outsider_key)
        assert outsider.status_code in (200, 401, 403, 404), outsider.text
        for response in (anonymous, unknown, outsider):
            _assert_no_secret(response.text)
            assert usage.digest not in response.text and "cc-7" not in response.text, response.text
    user_route: Final = usage.routes()[0]
    bad_format: Final = _export(gateway, usage, user_route, file_format="xml")
    bad_type: Final = _export(gateway, usage, user_route, export_type="daily_with_secrets")
    assert (bad_format.status_code, bad_type.status_code) == (422, 422), (bad_format.text, bad_type.text)
    still_served: Final = _export(gateway, usage, user_route)
    assert _metadata_of(_json_rows(still_served)) == [(dict(_KEY_METADATA_EXPORTED), dict(_USER_METADATA_EXPORTED))], (
        still_served.text
    )


@pytest.mark.parametrize(
    ("stored_key", "stored_user", "with_user", "expected"),
    (
        pytest.param({}, {}, True, ({}, {}), id="empty-objects"),
        pytest.param({"plain": "v"}, None, False, ({"plain": "v"}, {}), id="key-without-user"),
        pytest.param(["a", {"api_key": "list_1"}], "text", True, ({}, {}), id="array-and-string"),
        pytest.param(None, 7, True, ({}, {}), id="null-and-number"),
        pytest.param(
            {"logging": [], "callback_settings": None, "scim_roles": ""},
            {"secret_manager_settings": {}, "scim_enterprise": None},
            True,
            ({}, {}),
            id="only-dropped-slots",
        ),
    ),
)
def test_missing_empty_and_non_object_metadata_export_as_empty_objects(
    gateway: Gateway,
    scenario: Scenario,
    stored_key: JsonValue,
    stored_user: JsonValue,
    with_user: bool,
    expected: tuple[JsonValue, JsonValue],
) -> None:
    usage: Final = _seed(gateway, scenario, with_user=with_user)
    scenario.cleanups.callback(
        write_rows, """UPDATE "LiteLLM_UserTable" SET metadata = '{}'::jsonb WHERE user_id = %s""", (usage.user_id,)
    )
    scenario.cleanups.callback(
        write_rows,
        """UPDATE "LiteLLM_VerificationToken" SET metadata = '{}'::jsonb WHERE token = %s""",
        (usage.digest,),
    )
    write_rows(
        'UPDATE "LiteLLM_VerificationToken" SET metadata = %s::jsonb WHERE token = %s',
        (json.dumps(stored_key), usage.digest),
    )
    if with_user:
        write_rows(
            'UPDATE "LiteLLM_UserTable" SET metadata = %s::jsonb WHERE user_id = %s',
            (json.dumps(stored_user), usage.user_id),
        )
    response: Final = _export(gateway, usage, usage.routes()[1])
    rows: Final = _json_rows(response)
    assert [(row["api_key"], row["user_id"]) for row in rows] == [
        (usage.digest, usage.user_id if with_user else None)
    ], response.text
    assert _metadata_of(rows) == [expected], response.text
    assert "list_1" not in response.text, response.text
    as_csv: Final = _export(gateway, usage, usage.routes()[1], file_format="csv")
    assert as_csv.status_code == 200, as_csv.text
    _, line = tuple(csv.reader(io.StringIO(as_csv.text)))
    assert (json.loads(line[-2]), json.loads(line[-1])) == expected, as_csv.text


def test_metadata_updates_through_the_management_api_show_in_the_next_export(
    gateway: Gateway, scenario: Scenario
) -> None:
    usage: Final = _seed(gateway, scenario)
    route: Final = usage.routes()[0]
    first: Final = _export(gateway, usage, route)
    again: Final = _export(gateway, usage, route)
    assert first.status_code == 200 and first.content == again.content, (first.text, again.text)

    gateway.post("/key/update", {"key": usage.key, "metadata": {"owner": "ana", "webhook_secret": "wh_2"}})
    gateway.post("/user/update", {"user_id": usage.user_id, "metadata": {"site": "lab", "accessKey": "ak_2"}})
    updated: Final = _export(gateway, usage, route)
    assert _metadata_of(_json_rows(updated)) == [
        ({"owner": "ana", "webhook_secret": _REDACTED}, {"site": "lab", "accessKey": _REDACTED})
    ], updated.text
    assert "wh_2" not in updated.text and "ak_2" not in updated.text, updated.text
    info: Final = object_value(gateway.get("/key/info", {"key": usage.key})["info"])
    assert info["metadata"] == {"owner": "ana", "webhook_secret": "wh_2"}


def test_credential_name_segmentation_keeps_plain_names_and_redacts_every_spelling(
    gateway: Gateway, scenario: Scenario
) -> None:
    kept: Final[dict[str, JsonValue]] = {
        name: "v" for name in ("max_tokens", "keyboard", "author", "monkey", "secretary", "tokenizer")
    }
    redacted: Final = (
        "api_key",
        "api-key",
        "apiKey",
        "APIKey",
        "AWSSecret",
        "x-api-key",
        "apikey",
        "APITOKEN",
        "accesstoken",
        "clientsecret",
        "privatekey",
        "passwd",
        "pwd",
        "webhooksecret",
        "input_cost_per_token",
        "token_count",
        "Authorization",
        "vertex_credentials",
    )
    stored: Final[dict[str, JsonValue]] = {**kept, **{name: "s" for name in redacted}}
    usage: Final = _seed(
        gateway, scenario, key_metadata=stored, user_metadata={"deep": [[{"a": {"b": [{"apiKey": "s"}]}}]]}
    )
    response: Final = _export(gateway, usage, usage.routes()[0])
    assert _metadata_of(_json_rows(response)) == [
        (
            {**kept, **{name: _REDACTED for name in redacted}},
            {"deep": [[{"a": {"b": [{"apiKey": _REDACTED}]}}]]},
        )
    ], response.text


def test_model_info_redaction_keeps_the_shared_default_masker_behaviour(gateway: Gateway, scenario: Scenario) -> None:
    model: Final = scenario.model(
        input_cost_per_token=0.001,
        extra_headers={"apiKey": "camel-header-value", "x-api-key": "kebab-header-value", "plain": "p"},
    )
    listed: Final = gateway.request("GET", "/model/info")
    assert listed.status_code == 200, listed.text
    params: Final = [
        object_value(entry["litellm_params"])
        for entry in _ROWS.validate_python(object_value(listed.json())["data"])
        if entry["model_name"] == model
    ]
    assert [(entry["input_cost_per_token"], entry["extra_headers"], "api_key" in entry) for entry in params] == [
        (0.001, {"apiKey": "camel-header-value", "x-api-key": "keba**********alue", "plain": "p"}, False)
    ], listed.text
    assert "integration-provider-key" not in listed.text and "kebab-header-value" not in listed.text, listed.text


_BURST: Final = 30


def _burst_request(gateway: Gateway, usage: _Usage, index: int) -> tuple[str, int, str]:
    if index % 3 == 0:
        exported: Final = _export(gateway, usage, usage.routes()[index % 6])
        return ("admin", exported.status_code, exported.text)
    if index % 3 == 1:
        scoped: Final = _export(gateway, usage, usage.routes()[0], key=usage.key)
        return ("owner", scoped.status_code, scoped.text)
    return (
        "chat",
        200,
        _chat(gateway, usage.model, usage.key, usage.tag, usage.end_user_id, stream=index % 2 == 0),
    )


def _burst_call(gateway: Gateway, usage: _Usage, index: int) -> tuple[str, int, str]:
    try:
        return _burst_request(gateway, usage, index)
    except httpx.TransportError as error:
        return ("transport", 0, type(error).__name__)


def test_export_burst_survives_a_worker_kill_and_each_chat_lands_once(gateway: Gateway, tmp_path: Path) -> None:
    with owned_proxy_process(gateway, tmp_path, {}, workers=2) as owned, owned.gateway.scenario() as scenario:
        proxy: Final = owned.gateway
        usage: Final = _seed(proxy, scenario, key_metadata=dict(_KEY_METADATA), user_metadata=dict(_USER_METADATA))
        workers: Final = owned.worker_pids()
        assert len(workers) == 2, workers
        with ThreadPoolExecutor(max_workers=_BURST) as pool, owned.replacing_worker(min(workers)) as kill:
            futures: Final = tuple(pool.submit(_burst_call, proxy, usage, index) for index in range(_BURST))
            kill()
            during_kill: Final = tuple(future.result() for future in futures)

        dropped: Final = tuple(index for index, outcome in enumerate(during_kill) if outcome[0] == "transport")
        assert len(dropped) < _BURST, during_kill
        retried: Final = {index: _burst_request(proxy, usage, index) for index in dropped}
        assert [outcome[:2] for outcome in retried.values() if outcome[1] != 200] == [], retried
        outcomes: Final = tuple(retried.get(index, outcome) for index, outcome in enumerate(during_kill))
        assert len(outcomes) == _BURST
        admin: Final = tuple(outcome for outcome in outcomes if outcome[0] == "admin")
        owner: Final = tuple(outcome for outcome in outcomes if outcome[0] == "owner")
        chats: Final = tuple(outcome[2] for outcome in outcomes if outcome[0] == "chat")
        assert admin and owner and chats, [outcome[:2] for outcome in outcomes]
        assert [outcome[:2] for outcome in outcomes if outcome[0] in ("admin", "owner") and outcome[1] != 200] == []
        for _, _, body in admin:
            assert {
                (json.dumps(key, sort_keys=True), json.dumps(user, sort_keys=True))
                for key, user in _metadata_of(_ROWS.validate_json(body))
            } == {
                (
                    json.dumps(dict(_KEY_METADATA_EXPORTED), sort_keys=True),
                    json.dumps(dict(_USER_METADATA_EXPORTED), sort_keys=True),
                )
            }, body[:2000]
            _assert_no_secret(body)
        for _, _, body in owner:
            assert _metadata_of(_ROWS.validate_json(body)) == [("absent", "absent")], body[:2000]
            _assert_no_secret(body)

        assert len(set(chats)) == len(chats), "duplicate response id in burst"
        landed: Final = eventually(lambda: _spend_rows(chats), lambda rows: len(rows) >= len(chats), seconds=90)
        assert [row["request_id"] for row in landed] == sorted(chats), "a burst chat did not land exactly once"
        assert len(owned.worker_pids()) == 2
        after: Final = _export(proxy, usage, usage.routes()[0])
        assert _metadata_of(_json_rows(after)) == [(dict(_KEY_METADATA_EXPORTED), dict(_USER_METADATA_EXPORTED))], (
            after.text
        )
        total: Final = eventually(
            lambda: _json_rows(_export(proxy, usage, usage.routes()[0])),
            lambda rows: [row["api_requests"] for row in rows] == [1 + len(chats)],
            seconds=90,
        )
        assert [row["api_requests"] for row in total] == [1 + len(chats)]
