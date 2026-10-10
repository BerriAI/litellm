import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final

import pytest

from tests.integration._support.client import Gateway, object_value, string_value

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
