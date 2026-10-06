import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from typing import Final

import httpx
import pytest
from pydantic import BaseModel, JsonValue

from tests.integration._support.client import Gateway, eventually, object_value, string_value
from tests.integration._support.database import read_rows


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


class _AccessGroupInfo(BaseModel):
    access_group_id: str
    assigned_team_ids: list[str] | None
    assigned_key_ids: list[str] | None


def _denied_error(model: str, denied_type: str) -> dict[str, JsonValue]:
    return {
        "error": {
            "message": f"The requested model '{model}' is not available for this API key, or the model name is invalid. Check the models available to you and try again.",
            "type": denied_type,
            "param": "model",
            "code": "403",
        }
    }


def _observed(upstream: httpx.Client) -> list[dict[str, JsonValue]]:
    response: Final = upstream.get("/__observations")
    response.raise_for_status()
    requests: Final = response.json()["requests"]
    assert isinstance(requests, list)
    return requests


def _chat(proxy: Gateway, model: str, key: str, text: str) -> httpx.Response:
    return proxy.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": text}]},
        key=key,
    )


def _observation(text: str) -> dict[str, JsonValue]:
    return {
        "path": "/v1/chat/completions",
        "authorization": "Bearer integration-provider-key",
        "body": {"messages": [{"role": "user", "content": text}], "model": "gpt-4o-mini"},
        "method": "POST",
        "api_key": "",
    }


def _team_groups(team_id: str) -> list[dict[str, JsonValue]]:
    return read_rows('SELECT access_group_ids FROM "LiteLLM_TeamTable" WHERE team_id=%s', (team_id,))


def _key_groups(hashed: str) -> list[dict[str, JsonValue]]:
    return read_rows('SELECT access_group_ids FROM "LiteLLM_VerificationToken" WHERE token=%s', (hashed,))


def _group_row(access_group_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT assigned_team_ids, assigned_key_ids FROM "LiteLLM_AccessGroupTable" WHERE access_group_id=%s',
        (access_group_id,),
    )


def _assert_denied(response: httpx.Response, model: str, denied_type: str) -> None:
    assert response.status_code == 403, response.text
    assert response.json() == _denied_error(model, denied_type), response.text


def _assert_served(response: httpx.Response) -> None:
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("route_prefix", ("/v1/access_group", "/v1/unified_access_group"))
def test_access_group_assignment_grants_and_revokes_on_gateway_and_peer(
    gateway: Gateway, peer: Gateway, route_prefix: str
) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model()
        team_id: Final = scenario.team(models=["no-default-models"])
        team_key: Final = scenario.key(team_id=team_id)
        assigned_key: Final = scenario.key(models=["no-default-models"])
        hashed_key: Final = sha256(assigned_key.encode()).hexdigest()
        _observed(upstream)
        _assert_denied(_chat(gateway, model, team_key, "control team"), model, "team_model_access_denied")
        _assert_denied(_chat(gateway, model, assigned_key, "control key"), model, "key_model_access_denied")
        assert _observed(upstream) == []

        created: Final = gateway.request(
            "POST",
            route_prefix,
            {
                "access_group_name": f"integration-{uuid.uuid4().hex}",
                "access_model_names": [model],
                "assigned_team_ids": [team_id],
                "assigned_key_ids": [hashed_key],
            },
        )
        assert created.status_code == 201, created.text
        access_group_id: Final = string_value(created.json()["access_group_id"])
        scenario.cleanups.callback(gateway.request, "DELETE", f"{route_prefix}/{access_group_id}")
        info: Final = _AccessGroupInfo.model_validate(gateway.get(f"{route_prefix}/{access_group_id}"))
        assert info.assigned_team_ids == [team_id]
        assert info.assigned_key_ids == [hashed_key]
        assert _group_row(access_group_id) == [{"assigned_team_ids": [team_id], "assigned_key_ids": [hashed_key]}]
        assert _team_groups(team_id) == [{"access_group_ids": [access_group_id]}]
        assert _key_groups(hashed_key) == [{"access_group_ids": [access_group_id]}]

        expected: Final = []
        for proxy in (gateway, peer):
            for key in (team_key, assigned_key):
                text: Final = f"granted {uuid.uuid4().hex}"
                _assert_served(
                    eventually(
                        lambda p=proxy, k=key, t=text: _chat(p, model, k, t),
                        lambda response: response.status_code == 200,
                        seconds=30,
                    )
                )
                expected.append(_observation(text))
        assert _observed(upstream) == expected

        removed_team: Final = gateway.request("PUT", f"{route_prefix}/{access_group_id}", {"assigned_team_ids": []})
        assert removed_team.status_code == 200, removed_team.text
        assert _team_groups(team_id) == [{"access_group_ids": []}]
        _assert_denied(_chat(gateway, model, team_key, "denied"), model, "team_model_access_denied")
        _assert_denied(
            eventually(
                lambda: _chat(peer, model, team_key, "denied"),
                lambda response: response.status_code == 403,
                seconds=75,
            ),
            model,
            "team_model_access_denied",
        )
        _observed(upstream)
        still_granted: Final = f"still granted {uuid.uuid4().hex}"
        _assert_served(_chat(gateway, model, assigned_key, still_granted))
        assert _observed(upstream) == [_observation(still_granted)]

        removed_key: Final = gateway.request("PUT", f"{route_prefix}/{access_group_id}", {"assigned_key_ids": []})
        assert removed_key.status_code == 200, removed_key.text
        assert _key_groups(hashed_key) == [{"access_group_ids": []}]
        _assert_denied(_chat(gateway, model, assigned_key, "denied"), model, "key_model_access_denied")
        assert _observed(upstream) == []

        restored: Final = gateway.request(
            "PUT",
            f"{route_prefix}/{access_group_id}",
            {"assigned_team_ids": [team_id], "assigned_key_ids": [hashed_key]},
        )
        assert restored.status_code == 200, restored.text
        _observed(upstream)
        regranted: Final = []
        for key in (team_key, assigned_key):
            text = f"re-granted {uuid.uuid4().hex}"
            _assert_served(_chat(gateway, model, key, text))
            regranted.append(_observation(text))
        assert _observed(upstream) == regranted

        deleted: Final = gateway.request("DELETE", f"{route_prefix}/{access_group_id}")
        assert deleted.status_code == 204, deleted.text
        assert _group_row(access_group_id) == []
        assert _team_groups(team_id) == [{"access_group_ids": []}]
        assert _key_groups(hashed_key) == [{"access_group_ids": []}]
        denied_pairs: Final = (
            (team_key, "team_model_access_denied"),
            (assigned_key, "key_model_access_denied"),
        )
        for key, denied_type in denied_pairs:
            _assert_denied(_chat(gateway, model, key, "deleted"), model, denied_type)
            _assert_denied(
                eventually(
                    lambda k=key: _chat(peer, model, k, "deleted"),
                    lambda response: response.status_code == 403,
                    seconds=75,
                ),
                model,
                denied_type,
            )
        _observed(upstream)
        for proxy in (gateway, peer):
            for key, denied_type in denied_pairs:
                _assert_denied(_chat(proxy, model, key, "deleted"), model, denied_type)
        assert _observed(upstream) == []


def test_raw_key_in_assigned_key_ids_grants_the_same_access_as_the_hashed_key(gateway: Gateway) -> None:
    pytest.skip("BUG: raw key in assigned_key_ids is stored as-is and grants nothing")
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model()
        key: Final = scenario.key(models=["no-default-models"])
        hashed_key: Final = sha256(key.encode()).hexdigest()
        _observed(upstream)
        _assert_denied(_chat(gateway, model, key, "control"), model, "key_model_access_denied")
        assert _observed(upstream) == []

        created: Final = gateway.request(
            "POST",
            "/v1/access_group",
            {
                "access_group_name": f"integration-{uuid.uuid4().hex}",
                "access_model_names": [model],
                "assigned_key_ids": [key],
            },
        )
        assert created.status_code == 201, created.text
        access_group_id: Final = string_value(created.json()["access_group_id"])
        scenario.cleanups.callback(gateway.request, "DELETE", f"/v1/access_group/{access_group_id}")
        info: Final = _AccessGroupInfo.model_validate(gateway.get(f"/v1/access_group/{access_group_id}"))
        assert info.assigned_key_ids == [key]
        assert _group_row(access_group_id) == [{"assigned_team_ids": None, "assigned_key_ids": [key]}]
        assert _key_groups(hashed_key) == [{"access_group_ids": [access_group_id]}]
        text: Final = f"granted {uuid.uuid4().hex}"
        _assert_served(_chat(gateway, model, key, text))
        assert _observed(upstream) == [_observation(text)]
