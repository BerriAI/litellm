import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, gateway_from_environment, object_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.shares_worker_state

SCIM_HEADERS: Final = {"Content-Type": "application/scim+json"}
SCIM_CORE_GROUP_SCHEMA: Final = "urn:ietf:params:scim:schemas:core:2.0:Group"
SCIM_PATCH_SCHEMA: Final = "urn:ietf:params:scim:api:messages:2.0:PatchOp"
SCIM_CONFIG: Final = TypeAdapter(dict[str, JsonValue])
ORGANIZATION_ATTRIBUTE: Final = "org_id"


@dataclass(frozen=True, slots=True)
class OrganizationMappingRig:
    gateway: Gateway
    engineering_alias: str
    engineering_id: str
    finance_alias: str
    finance_id: str
    member: str


def _owned_scim_config(path: Path, settings: Mapping[str, JsonValue]) -> Path:
    config: Final = object_value(
        SCIM_CONFIG.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    )
    litellm_settings: Final = object_value(config["litellm_settings"])
    path.write_text(yaml.safe_dump({**config, "litellm_settings": {**litellm_settings, **settings}}))
    return path


def _delete_group_if_present(gateway: Gateway, group_id: str) -> None:
    if read_rows('SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_id = %s', (group_id,)):
        deleted: Final = gateway.request("DELETE", f"/scim/v2/Groups/{group_id}", headers=SCIM_HEADERS)
        assert deleted.status_code == 204, deleted.text
    assert read_rows('SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_id = %s', (group_id,)) == []


def _team_organization(team_id: str) -> str | None:
    rows: Final = read_rows('SELECT organization_id FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team_id,))
    assert len(rows) == 1, rows
    organization_id: Final = rows[0]["organization_id"]
    assert organization_id is None or isinstance(organization_id, str), rows
    return organization_id


def _organization_memberships(user_id: str) -> set[str]:
    rows: Final = read_rows(
        'SELECT organization_id FROM "LiteLLM_OrganizationMembership" WHERE user_id = %s', (user_id,)
    )
    return {str(row["organization_id"]) for row in rows}


def _group_body(
    group_id: str, display_name: str, members: tuple[str, ...], **attributes: JsonValue
) -> dict[str, JsonValue]:
    return {
        "schemas": [SCIM_CORE_GROUP_SCHEMA],
        "id": group_id,
        "displayName": display_name,
        "members": [{"value": member} for member in members],
        **attributes,
    }


def _patch_body(*operations: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {"schemas": [SCIM_PATCH_SCHEMA], "Operations": list(operations)}


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[OrganizationMappingRig]:
    directory: Final = tmp_path_factory.mktemp("scim-group-organization")
    config: Final = _owned_scim_config(
        directory / "config.yaml", {"scim_settings": {"organization_id_attribute": ORGANIZATION_ATTRIBUTE}}
    )
    with (
        gateway_from_environment() as gateway,
        owned_proxy_process(gateway, directory, {}, config=config) as owned,
        owned.gateway.scenario() as scenario,
    ):
        suffix: Final = uuid.uuid4().hex[:8]
        engineering_alias: Final = f"engineering-{suffix}"
        finance_alias: Final = f"finance-{suffix}"
        yield OrganizationMappingRig(
            gateway=owned.gateway,
            engineering_alias=engineering_alias,
            engineering_id=scenario.organization(organization_alias=engineering_alias),
            finance_alias=finance_alias,
            finance_id=scenario.organization(organization_alias=finance_alias),
            member=scenario.user(user_role="internal_user"),
        )


def _create_group(rig: OrganizationMappingRig, group_id: str, **attributes: JsonValue) -> httpx.Response:
    return rig.gateway.request(
        "POST",
        "/scim/v2/Groups",
        _group_body(group_id, f"Infrastructure {group_id[:8]}", (rig.member,), **attributes),
        headers=SCIM_HEADERS,
    )


@pytest.mark.timeout(300)
def test_scim_group_post_places_the_team_in_the_organization_the_attribute_names(rig: OrganizationMappingRig) -> None:
    group_id: Final = str(uuid.uuid4())
    try:
        created: Final = _create_group(rig, group_id, org_id=rig.engineering_alias)
        assert created.status_code == 201, created.text
        assert _team_organization(group_id) == rig.engineering_id
        assert rig.engineering_id in _organization_memberships(rig.member)
        info: Final = rig.gateway.request("GET", f"/team/info?team_id={group_id}")
        assert info.status_code == 200, info.text
        assert info.json()["team_info"]["organization_id"] == rig.engineering_id, info.text
    finally:
        _delete_group_if_present(rig.gateway, group_id)


@pytest.mark.timeout(300)
def test_scim_group_post_without_the_attribute_leaves_the_team_outside_any_organization(
    rig: OrganizationMappingRig,
) -> None:
    group_id: Final = str(uuid.uuid4())
    try:
        created: Final = _create_group(rig, group_id, department=rig.engineering_alias)
        assert created.status_code == 201, created.text
        assert _team_organization(group_id) is None
    finally:
        _delete_group_if_present(rig.gateway, group_id)


@pytest.mark.timeout(300)
def test_scim_group_post_with_an_unknown_organization_alias_is_a_404_and_creates_no_team(
    rig: OrganizationMappingRig,
) -> None:
    group_id: Final = str(uuid.uuid4())
    missing_alias: Final = f"missing-{uuid.uuid4().hex[:8]}"
    created: Final = _create_group(rig, group_id, org_id=missing_alias)
    assert created.status_code == 404, created.text
    assert missing_alias in created.text
    assert read_rows('SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_id = %s', (group_id,)) == []
    assert (
        read_rows(
            'SELECT organization_id FROM "LiteLLM_OrganizationTable" WHERE organization_alias = %s', (missing_alias,)
        )
        == []
    )


@pytest.mark.timeout(300)
def test_scim_group_put_and_patch_move_the_team_between_organizations(rig: OrganizationMappingRig) -> None:
    group_id: Final = str(uuid.uuid4())
    try:
        created: Final = _create_group(rig, group_id, org_id=rig.engineering_alias)
        assert created.status_code == 201, created.text

        replaced: Final = rig.gateway.request(
            "PUT",
            f"/scim/v2/Groups/{group_id}",
            _group_body(group_id, "Infrastructure renamed", (rig.member,), org_id=rig.finance_alias),
            headers=SCIM_HEADERS,
        )
        assert replaced.status_code == 200, replaced.text
        assert _team_organization(group_id) == rig.finance_id
        assert rig.finance_id in _organization_memberships(rig.member)

        kept: Final = rig.gateway.request(
            "PUT",
            f"/scim/v2/Groups/{group_id}",
            _group_body(group_id, "Infrastructure renamed", (rig.member,)),
            headers=SCIM_HEADERS,
        )
        assert kept.status_code == 200, kept.text
        assert _team_organization(group_id) == rig.finance_id, "a PUT without the attribute leaves the organization"

        patched: Final = rig.gateway.request(
            "PATCH",
            f"/scim/v2/Groups/{group_id}",
            _patch_body({"op": "replace", "path": ORGANIZATION_ATTRIBUTE, "value": rig.engineering_alias}),
            headers=SCIM_HEADERS,
        )
        assert patched.status_code == 200, patched.text
        assert _team_organization(group_id) == rig.engineering_id

        unknown: Final = rig.gateway.request(
            "PATCH",
            f"/scim/v2/Groups/{group_id}",
            _patch_body({"op": "replace", "path": ORGANIZATION_ATTRIBUTE, "value": "missing-org"}),
            headers=SCIM_HEADERS,
        )
        assert unknown.status_code == 404, unknown.text
        assert _team_organization(group_id) == rig.engineering_id

        removed: Final = rig.gateway.request(
            "PATCH",
            f"/scim/v2/Groups/{group_id}",
            _patch_body({"op": "remove", "path": ORGANIZATION_ATTRIBUTE}),
            headers=SCIM_HEADERS,
        )
        assert removed.status_code == 200, removed.text
        assert _team_organization(group_id) is None
    finally:
        _delete_group_if_present(rig.gateway, group_id)
