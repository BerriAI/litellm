"""E2E matrix for custom team metadata validation against the DB-backed proxy.

The proxy (see store_model_db_config.yaml) registers
team_metadata_validator_e2e.validate_team_metadata, which dispatches per
request to one of three independent implementations via the
`_e2e_validator_impl` metadata key: a static allowlist function, an
HTTP-backed function calling the cost center service started by CI, and an
immutability-enforcing class instance. Metadata without the dispatch key is
accepted untouched, so the rest of this suite is unaffected.
"""

import os
import uuid

import httpx
import pytest

PROXY_BASE_URL = os.getenv("PROXY_BASE_URL", "http://localhost:4000")
MASTER_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-1234")
HEADERS = {"Authorization": f"Bearer {MASTER_KEY}", "Content-Type": "application/json"}

UNAVAILABLE_MESSAGE = "Cost center validation is unavailable right now; the team was not saved. Contact FinOps."

IMPLS = ["allowlist", "http", "immutable"]

REQUIRED_MESSAGES = {
    "allowlist": "cost_center is required in team metadata",
    "http": "cost_center missing per cost center service",
    "immutable": "cost_center is required in team metadata",
}
UNKNOWN_MESSAGES = {
    "allowlist": "is not recognized",
    "http": "rejected by cost center service",
}


def _meta(impl, **fields):
    return {"_e2e_validator_impl": impl, **fields}


def _create_team(metadata, team_id=None):
    body = {"team_alias": f"meta-validate-{uuid.uuid4().hex[:8]}"}
    if team_id is not None:
        body["team_id"] = team_id
    if metadata is not None:
        body["metadata"] = metadata
    return httpx.post(f"{PROXY_BASE_URL}/team/new", headers=HEADERS, json=body, timeout=30)


def _patch_team(team_id, body):
    return httpx.patch(f"{PROXY_BASE_URL}/team/{team_id}", headers=HEADERS, json=body, timeout=30)


def _post_update(team_id, body):
    return httpx.post(f"{PROXY_BASE_URL}/team/update", headers=HEADERS, json={"team_id": team_id, **body}, timeout=30)


def _team_info(team_id):
    return httpx.get(f"{PROXY_BASE_URL}/team/info", headers=HEADERS, params={"team_id": team_id}, timeout=30)


def _delete_team(team_id):
    httpx.post(f"{PROXY_BASE_URL}/team/delete", headers=HEADERS, json={"team_ids": [team_id]}, timeout=30)


@pytest.fixture
def team_with_cost_center(request):
    impl = request.param
    team_id = f"meta-validate-{impl}-{uuid.uuid4().hex[:8]}"
    response = _create_team(metadata=_meta(impl, cost_center="CC-1001"), team_id=team_id)
    assert response.status_code == 200, response.text
    yield impl, team_id
    _delete_team(team_id)


@pytest.mark.parametrize("impl", IMPLS)
def test_create_with_valid_cost_center_succeeds(impl):
    response = _create_team(metadata=_meta(impl, cost_center="CC-1001"))
    assert response.status_code == 200, response.text
    team_id = response.json()["team_id"]
    try:
        assert response.json()["metadata"]["cost_center"] == "CC-1001"
    finally:
        _delete_team(team_id)


@pytest.mark.parametrize("impl", IMPLS)
def test_create_without_cost_center_is_rejected(impl):
    team_id = f"meta-validate-reject-{impl}-{uuid.uuid4().hex[:8]}"
    response = _create_team(metadata=_meta(impl), team_id=team_id)
    assert response.status_code == 400, response.text
    assert REQUIRED_MESSAGES[impl] in response.text
    info = _team_info(team_id)
    assert info.status_code == 404, "rejected create must not leave a team row behind"


@pytest.mark.parametrize("impl", IMPLS)
def test_create_with_unknown_cost_center(impl):
    response = _create_team(metadata=_meta(impl, cost_center="CC-9999"))
    if impl == "immutable":
        assert response.status_code == 200, response.text
        _delete_team(response.json()["team_id"])
        return
    assert response.status_code == 400, response.text
    assert UNKNOWN_MESSAGES[impl] in response.text


@pytest.mark.parametrize("team_with_cost_center", IMPLS, indirect=True)
def test_patch_changing_cost_center(team_with_cost_center):
    impl, team_id = team_with_cost_center
    response = _patch_team(team_id, {"metadata": {"cost_center": "CC-1002"}})
    if impl == "immutable":
        assert response.status_code == 400, response.text
        assert "immutable once set" in response.text
        info = _team_info(team_id).json()["team_info"]["metadata"]
        assert info["cost_center"] == "CC-1001", "blocked update must leave stored metadata intact"
        return
    assert response.status_code == 200, response.text
    assert response.json()["metadata"]["cost_center"] == "CC-1002"


@pytest.mark.parametrize("team_with_cost_center", IMPLS, indirect=True)
def test_patch_unrelated_key_validates_merged_result(team_with_cost_center):
    impl, team_id = team_with_cost_center
    response = _patch_team(team_id, {"metadata": {"team_notes": "hello"}})
    assert response.status_code == 200, response.text
    merged = response.json()["metadata"]
    assert merged["cost_center"] == "CC-1001"
    assert merged["team_notes"] == "hello"


@pytest.mark.parametrize("team_with_cost_center", IMPLS, indirect=True)
def test_patch_null_deleting_cost_center_is_rejected(team_with_cost_center):
    impl, team_id = team_with_cost_center
    response = _patch_team(team_id, {"metadata": {"cost_center": None}})
    assert response.status_code == 400, response.text
    assert REQUIRED_MESSAGES[impl] in response.text
    info = _team_info(team_id).json()["team_info"]["metadata"]
    assert info["cost_center"] == "CC-1001"


@pytest.mark.parametrize("team_with_cost_center", IMPLS, indirect=True)
def test_post_update_dropping_cost_center_is_rejected(team_with_cost_center):
    impl, team_id = team_with_cost_center
    response = _post_update(team_id, {"metadata": _meta(impl, team_notes="only-notes")})
    assert response.status_code == 400, response.text
    assert REQUIRED_MESSAGES[impl] in response.text


@pytest.mark.parametrize("team_with_cost_center", IMPLS, indirect=True)
def test_update_without_metadata_skips_validation(team_with_cost_center):
    impl, team_id = team_with_cost_center
    response = _post_update(team_id, {"tpm_limit": 55})
    assert response.status_code == 200, response.text


def test_http_service_outage_fails_closed_with_configured_message():
    response = _create_team(metadata=_meta("http_down", cost_center="CC-1001"))
    assert response.status_code == 503, response.text
    assert UNAVAILABLE_MESSAGE in response.text


def test_metadata_without_dispatch_key_is_untouched():
    response = _create_team(metadata={"any_key": "any_value"})
    assert response.status_code == 200, response.text
    team_id = response.json()["team_id"]
    try:
        update = _post_update(team_id, {"metadata": {"any_key": "changed"}})
        assert update.status_code == 200, update.text
    finally:
        _delete_team(team_id)
