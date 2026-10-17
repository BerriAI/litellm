import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Final

import pytest

from tests.integration._support.client import Gateway, object_value, string_value
from tests.integration._support.database import write_rows

FALLBACK_EMAIL: Final = f"fallback-{uuid.uuid4().hex}@example.com"


@contextmanager
def _access_group(gateway: Gateway, body: dict[str, object], *, key: str) -> Iterator[str]:
    created: Final = gateway.request("POST", "/v1/access_group", body, key=key)
    assert created.status_code == 201, created.text
    identity: Final = string_value(created.json()["access_group_id"])
    try:
        yield identity
    finally:
        deleted: Final = gateway.request("DELETE", f"/v1/access_group/{identity}")
        assert deleted.status_code == 204, deleted.text


def test_access_group_detail_names_the_user_who_created_and_updated_it(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team_id: Final = scenario.team(team_alias="Review Team")
        user_id: Final = scenario.user(
            user_alias="Ada Reviewer",
            user_email=f"ada-{uuid.uuid4().hex}@example.com",
            user_role="proxy_admin",
        )
        key: Final = scenario.key(user_id=user_id)
        with _access_group(
            gateway,
            {
                "access_group_name": f"integration-{uuid.uuid4().hex}",
                "access_model_names": [],
                "assigned_team_ids": [team_id],
            },
            key=key,
        ) as group_id:
            updated: Final = gateway.request(
                "PUT",
                f"/v1/access_group/{group_id}",
                {"description": "updated description"},
                key=key,
            )
            assert updated.status_code == 200, updated.text

            detail: Final = gateway.request("GET", f"/v1/access_group/{group_id}")
            assert detail.status_code == 200, detail.text
            group: Final = object_value(detail.json())
            assert group["created_by_user"] == {"id": user_id, "name": "Ada Reviewer"}, detail.text
            assert group["updated_by_user"] == {"id": user_id, "name": "Ada Reviewer"}, detail.text
            assert group["assigned_teams"] == [{"id": team_id, "name": "Review Team"}], detail.text

            listing: Final = gateway.request("GET", "/v1/access_group")
            assert listing.status_code == 200, listing.text
            entries: Final = listing.json()
            assert isinstance(entries, list), listing.text
            listed: Final = object_value(next(entry for entry in entries if entry["access_group_id"] == group_id))
            assert listed["created_by_user"] == {"id": user_id, "name": "Ada Reviewer"}, listing.text


@pytest.mark.parametrize(
    ("user_fields", "expected_name"),
    (
        pytest.param({"user_email": FALLBACK_EMAIL}, FALLBACK_EMAIL, id="email"),
        pytest.param({}, None, id="neither"),
    ),
)
def test_access_group_creator_without_alias_falls_back_to_email_then_none(
    gateway: Gateway, user_fields: dict[str, str], expected_name: str | None
) -> None:
    with gateway.scenario() as scenario:
        user_id: Final = scenario.user(user_role="proxy_admin", **user_fields)
        key: Final = scenario.key(user_id=user_id)
        with _access_group(
            gateway,
            {"access_group_name": f"integration-{uuid.uuid4().hex}", "access_model_names": []},
            key=key,
        ) as group_id:
            detail: Final = gateway.request("GET", f"/v1/access_group/{group_id}")
            assert detail.status_code == 200, detail.text
            assert object_value(detail.json())["created_by_user"] == {
                "id": user_id,
                "name": expected_name,
            }, detail.text


def test_access_group_created_by_master_key_names_the_default_admin(gateway: Gateway) -> None:
    with _access_group(
        gateway,
        {"access_group_name": f"integration-{uuid.uuid4().hex}", "access_model_names": []},
        key=gateway.key,
    ) as group_id:
        detail: Final = gateway.request("GET", f"/v1/access_group/{group_id}")
        assert detail.status_code == 200, detail.text
        assert object_value(detail.json())["created_by_user"] == {
            "id": "default_user_id",
            "name": None,
        }, detail.text


def test_access_group_creator_with_blank_alias_falls_back_to_email(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        email: Final = f"blank-alias-{uuid.uuid4().hex}@example.com"
        user_id: Final = scenario.user(user_role="proxy_admin", user_alias="", user_email=email)
        key: Final = scenario.key(user_id=user_id)
        with _access_group(
            gateway,
            {"access_group_name": f"integration-{uuid.uuid4().hex}", "access_model_names": []},
            key=key,
        ) as group_id:
            detail: Final = gateway.request("GET", f"/v1/access_group/{group_id}")
            assert detail.status_code == 200, detail.text
            assert object_value(detail.json())["created_by_user"] == {"id": user_id, "name": email}, detail.text


def test_access_group_update_by_another_user_keeps_creator_and_names_updater(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        creator_id: Final = scenario.user(user_alias="Creator One", user_role="proxy_admin")
        creator_key: Final = scenario.key(user_id=creator_id)
        updater_id: Final = scenario.user(
            user_alias="Updater Two", user_email=f"updater-{uuid.uuid4().hex}@example.com", user_role="proxy_admin"
        )
        updater_key: Final = scenario.key(user_id=updater_id)
        with _access_group(
            gateway,
            {"access_group_name": f"integration-{uuid.uuid4().hex}", "access_model_names": []},
            key=creator_key,
        ) as group_id:
            updated: Final = gateway.request(
                "PUT", f"/v1/access_group/{group_id}", {"description": "by another"}, key=updater_key
            )
            assert updated.status_code == 200, updated.text
            detail: Final = gateway.request("GET", f"/v1/access_group/{group_id}")
            assert detail.status_code == 200, detail.text
            group: Final = object_value(detail.json())
            assert group["created_by_user"] == {"id": creator_id, "name": "Creator One"}, detail.text
            assert group["updated_by_user"] == {"id": updater_id, "name": "Updater Two"}, detail.text


def test_access_group_creator_deleted_afterwards_resolves_name_to_none(gateway: Gateway) -> None:
    user_id: Final = f"del-{uuid.uuid4().hex}"
    created_user: Final = gateway.request(
        "POST", "/user/new", {"user_id": user_id, "user_alias": "Soon Deleted", "user_role": "proxy_admin"}
    )
    assert created_user.status_code == 200, created_user.text
    minted: Final = gateway.request("POST", "/key/generate", {"user_id": user_id})
    assert minted.status_code == 200, minted.text
    key: Final = string_value(minted.json()["key"])
    with _access_group(
        gateway,
        {"access_group_name": f"integration-{uuid.uuid4().hex}", "access_model_names": []},
        key=key,
    ) as group_id:
        deleted: Final = gateway.request("POST", "/user/delete", {"user_ids": [user_id]})
        assert deleted.status_code == 200 and deleted.json() == 1, deleted.text
        detail: Final = gateway.request("GET", f"/v1/access_group/{group_id}")
        assert detail.status_code == 200, detail.text
        assert object_value(detail.json())["created_by_user"] == {
            "id": user_id,
            "name": None,
        }, detail.text


def test_access_group_name_reflects_the_current_alias_after_a_rename(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        user_id: Final = scenario.user(user_alias="Old Alias", user_role="proxy_admin")
        key: Final = scenario.key(user_id=user_id)
        with _access_group(
            gateway,
            {"access_group_name": f"integration-{uuid.uuid4().hex}", "access_model_names": []},
            key=key,
        ) as group_id:
            renamed: Final = gateway.request("POST", "/user/update", {"user_id": user_id, "user_alias": "New Alias"})
            assert renamed.status_code == 200, renamed.text
            detail: Final = gateway.request("GET", f"/v1/access_group/{group_id}")
            assert detail.status_code == 200, detail.text
            assert object_value(detail.json())["created_by_user"] == {
                "id": user_id,
                "name": "New Alias",
            }, detail.text


def test_access_group_list_resolves_names_past_the_in_chunk_size(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    chunk_size: Final = 5000
    write_rows(
        f'INSERT INTO "LiteLLM_UserTable" (user_id, user_alias) '
        f"SELECT 'chunk-{marker}-' || i, 'ChunkUser-' || i FROM generate_series(1, {chunk_size + 1}) AS i",
        (),
    )
    write_rows(
        f'INSERT INTO "LiteLLM_AccessGroupTable" (access_group_id, access_group_name, created_by) '
        f"SELECT gen_random_uuid(), 'chunkgrp-{marker}-' || i, 'chunk-{marker}-' || i "
        f"FROM generate_series(1, {chunk_size + 1}) AS i",
        (),
    )
    try:
        listing: Final = gateway.request("GET", "/v1/access_group")
        assert listing.status_code == 200, listing.text
        entries: Final = listing.json()
        assert isinstance(entries, list), listing.text
        seeded: Final = [e for e in entries if str(e.get("access_group_name", "")).startswith(f"chunkgrp-{marker}-")]
        assert len(seeded) == chunk_size + 1
        for entry in seeded:
            assert entry["created_by_user"]["name"] == str(entry["created_by_user"]["id"]).replace(
                f"chunk-{marker}-", "ChunkUser-"
            ), entry
    finally:
        write_rows('DELETE FROM "LiteLLM_AccessGroupTable" WHERE access_group_name LIKE %s', (f"chunkgrp-{marker}-%",))
        write_rows('DELETE FROM "LiteLLM_UserTable" WHERE user_id LIKE %s', (f"chunk-{marker}-%",))


def test_access_group_endpoints_reject_a_non_admin_caller(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        user_id: Final = scenario.user(user_role="internal_user")
        key: Final = scenario.key(user_id=user_id)
        listing: Final = gateway.request("GET", "/v1/access_group", key=key)
        assert listing.status_code == 401, listing.text


def test_access_group_get_unknown_id_returns_404(gateway: Gateway) -> None:
    response: Final = gateway.request("GET", f"/v1/access_group/{uuid.uuid4()}")
    assert response.status_code == 404, response.text


def test_access_group_put_with_malformed_body_returns_422(gateway: Gateway) -> None:
    response: Final = gateway.request("PUT", f"/v1/access_group/{uuid.uuid4()}", {"access_model_names": "not-a-list"})
    assert response.status_code == 422, response.text


def test_access_group_post_with_malformed_body_returns_422(gateway: Gateway) -> None:
    response: Final = gateway.request("POST", "/v1/access_group", {"access_group_name": 123})
    assert response.status_code == 422, response.text


def test_access_group_concurrent_updates_while_listing_stay_consistent(gateway: Gateway) -> None:
    with _access_group(
        gateway,
        {"access_group_name": f"integration-{uuid.uuid4().hex}", "access_model_names": []},
        key=gateway.key,
    ) as group_id:

        def call(index: int) -> int:
            if index % 2 == 0:
                response = gateway.request("PUT", f"/v1/access_group/{group_id}", {"description": f"desc-{index}"})
            else:
                response = gateway.request("GET", "/v1/access_group")
            return response.status_code

        with ThreadPoolExecutor(max_workers=8) as pool:
            statuses: Final = list(pool.map(call, range(20)))
        assert all(status == 200 for status in statuses), statuses
        detail: Final = gateway.request("GET", f"/v1/access_group/{group_id}")
        assert detail.status_code == 200, detail.text
