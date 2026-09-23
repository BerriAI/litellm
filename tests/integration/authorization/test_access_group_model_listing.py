import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final

import httpx
import pytest

from tests.integration._support.client import Gateway, eventually, object_value, string_value


@contextmanager
def _team_access_group(gateway: Gateway, team_id: str, model: str) -> Iterator[str]:
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


def _listed_model_ids(response: httpx.Response) -> tuple[str, ...]:
    entries: Final = response.json()["data"]
    assert isinstance(entries, list), response.text
    return tuple(string_value(object_value(entry)["id"]) for entry in entries)


@pytest.mark.covers("authorization.access_groups.team_key_lists_models_granted_through_team_access_group")
def test_team_key_lists_models_granted_through_team_access_group(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_id: Final = scenario.team(models=["no-default-models"])
        with _team_access_group(gateway, team_id, model):
            key: Final = scenario.key(team_id=team_id)
            response: Final = eventually(
                lambda: gateway.request("GET", "/v1/models", key=key),
                lambda value: value.status_code == 200 and _listed_model_ids(value) == (model,),
                return_last_on_timeout=True,
            )
            assert response.status_code == 200, response.text
            assert _listed_model_ids(response) == (model,), response.text
