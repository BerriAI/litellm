from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from litellm.proxy._types import (
    LiteLLM_VerificationToken,
    LitellmUserRoles,
    UpdateKeyRequest,
)
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.management_endpoints.management_v1 import router
from litellm.proxy.management_endpoints.management_v1.common import (
    MANAGEMENT_V1_PREFIX,
    ManagementProblem,
    problem_response,
    validation_problem,
)

app = FastAPI()


@app.exception_handler(ManagementProblem)
async def management_problem_exception_handler(request: Request, exc: ManagementProblem):
    return problem_response(exc.problem)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """The same translation `proxy_server` installs. Registering only the `ManagementProblem`
    handler here would let FastAPI's default 422 stand in for the real one, and the tests would
    pass against a status code production never returns."""
    return problem_response(validation_problem(exc.errors()))


app.include_router(router)
client = TestClient(app)

KEYS_PATH = f"{MANAGEMENT_V1_PREFIX}/keys"
HASHED_TOKEN = "a1b2c3d4" * 8
PLAINTEXT_KEY = "sk-plaintext-secret-value"


def _row(**overrides: Any) -> dict[str, Any]:
    return {
        "token": HASHED_TOKEN,
        "key_name": "sk-...alue",
        "key_alias": "reporting",
        "user_id": "test-user",
        "spend": 0.0,
        "models": [],
        "metadata": {},
        **overrides,
    }


@pytest.fixture
def key_write(monkeypatch):
    """Mocks the write path and hands back the prisma mock, so a test can assert on the
    exact dict handed to `update_data` as well as on the HTTP response."""
    prisma_client = AsyncMock()
    prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(
        return_value=LiteLLM_VerificationToken(token=HASHED_TOKEN, user_id="test-user")
    )
    prisma_client.update_data = AsyncMock(return_value={"data": _row()})
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma_client)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_api_key_cache", AsyncMock())
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", MagicMock())
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", True)
    monkeypatch.setattr("litellm.store_audit_logs", False)
    return prisma_client


@pytest.fixture
def as_proxy_admin():
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN
    )
    yield
    app.dependency_overrides.clear()


def _patch(body: dict[str, Any], key_id: str = HASHED_TOKEN):
    with patch(
        "litellm.proxy.management_endpoints.key_management_endpoints._delete_cache_key_object",
        new=AsyncMock(),
    ):
        return client.patch(f"{KEYS_PATH}/{key_id}", json=body, headers={"Authorization": "Bearer k"})


async def _drive_post(monkeypatch, existing_metadata: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    """Drive the legacy POST write core against the same mocked row, and return what it wrote."""
    from litellm.proxy.management_endpoints.key_management_endpoints import update_key_fn

    prisma_client = AsyncMock()
    prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(
        return_value=LiteLLM_VerificationToken(token=HASHED_TOKEN, user_id="test-user", metadata=existing_metadata)
    )
    prisma_client.update_data = AsyncMock(return_value={"data": _row()})
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma_client)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_api_key_cache", AsyncMock())
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", MagicMock())
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", True)
    monkeypatch.setattr("litellm.store_audit_logs", False)

    with patch(
        "litellm.proxy.management_endpoints.key_management_endpoints._delete_cache_key_object",
        new=AsyncMock(),
    ):
        await update_key_fn(
            request=MagicMock(),
            data=UpdateKeyRequest(key=HASHED_TOKEN, **body),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-admin", user_id="admin-user"
            ),
            litellm_changed_by=None,
        )
    return prisma_client.update_data.call_args.kwargs["data"]


# (label, stored_metadata, patch_body, what POST writes, what PATCH writes)
_METADATA_MAPPING = [
    (
        "sibling entries survive a merge patch but not a POST",
        {"cost_center": "cc-1", "owner": "data-eng"},
        {"cost_center": "cc-2"},
        {"cost_center": "cc-2"},
        {"cost_center": "cc-2", "owner": "data-eng"},
    ),
    (
        "a nested object recurses instead of being replaced",
        {"nested": {"a": 1, "b": 2}, "owner": "data-eng"},
        {"nested": {"b": 99}},
        {"nested": {"b": 99}},
        {"nested": {"a": 1, "b": 99}, "owner": "data-eng"},
    ),
    (
        "null deletes only its own entry",
        {"cost_center": "cc-1", "owner": "data-eng"},
        {"cost_center": None},
        {"cost_center": None},
        {"owner": "data-eng"},
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "label,stored,body,expected_post,expected_patch",
    _METADATA_MAPPING,
    ids=[row[0] for row in _METADATA_MAPPING],
)
async def test_metadata_merges_where_the_legacy_post_replaces(
    monkeypatch, key_write, as_proxy_admin, label, stored, body, expected_post, expected_patch
):
    """The one sanctioned divergence from `POST /key/update`, which writes the submitted
    metadata verbatim and so drops every entry the caller did not resend."""
    written_post = await _drive_post(monkeypatch, stored, {"metadata": body})
    assert written_post["metadata"] == expected_post

    key_write.db.litellm_verificationtoken.find_unique = AsyncMock(
        return_value=LiteLLM_VerificationToken(token=HASHED_TOKEN, user_id="test-user", metadata=stored)
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", key_write)

    assert _patch({"metadata": body}).status_code == 200
    assert key_write.update_data.call_args.kwargs["data"]["metadata"] == expected_patch


def test_answers_in_the_item_envelope_without_the_plaintext_secret(key_write, as_proxy_admin):
    """`{"data": {...}}`, and `key_id` is the row's hashed token even when the caller addressed
    the key by its plaintext secret, which must not come back in the body."""
    response = _patch({"tpm_limit": 77}, key_id=PLAINTEXT_KEY)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"data"}
    assert body["data"]["key_id"] == HASHED_TOKEN
    assert PLAINTEXT_KEY not in response.text
    assert "key" not in body["data"]


def test_a_row_without_its_own_id_fails_rather_than_falling_back(key_write, as_proxy_admin):
    """`key_id` has exactly one source, the row's hashed token. Without this, a fallback to any
    other field on the row would quietly put the caller's plaintext secret in the response."""
    key_write.update_data = AsyncMock(return_value={"data": {k: v for k, v in _row().items() if k != "token"}})

    response = _patch({"tpm_limit": 1}, key_id=PLAINTEXT_KEY)

    assert response.status_code == 500
    assert PLAINTEXT_KEY not in response.text


def test_null_clears_and_omission_preserves(key_write, as_proxy_admin):
    """Both directions in one test: a route that cleared everything would pass a clear-only
    assertion, and a route that cleared nothing would pass a preserve-only one."""
    assert _patch({"tpm_limit": None}).status_code == 200
    cleared = key_write.update_data.call_args.kwargs["data"]
    assert "tpm_limit" in cleared and cleared["tpm_limit"] is None

    assert _patch({"rpm_limit": 9}).status_code == 200
    preserved = key_write.update_data.call_args.kwargs["data"]
    assert "tpm_limit" not in preserved
    assert preserved["rpm_limit"] == 9


def test_does_not_slide_the_budget_window(key_write, as_proxy_admin):
    """A merge patch is idempotent, so a patch that never mentions `budget_duration` must leave
    `budget_reset_at` alone rather than postponing the key's reset on every save."""
    key_write.db.litellm_verificationtoken.find_unique = AsyncMock(
        return_value=LiteLLM_VerificationToken(token=HASHED_TOKEN, user_id="test-user", budget_duration="30d")
    )

    assert _patch({"rpm_limit": 5}).status_code == 200

    written = key_write.update_data.call_args.kwargs["data"]
    assert written["rpm_limit"] == 5
    assert "budget_reset_at" not in written
    assert "budget_duration" not in written


@pytest.mark.parametrize(
    "body,reason",
    [
        ({"tpm_limitt": 5}, "a misspelled field"),
        ({"key": HASHED_TOKEN}, "the legacy `key` spelling of the identifier"),
    ],
    ids=["misspelled field", "legacy key spelling"],
)
def test_rejects_bodies_that_would_otherwise_no_op(key_write, as_proxy_admin, body, reason):
    """On a merge patch the set of fields present IS the request, so anything unrecognized has to
    fail loudly rather than silently changing nothing the way the legacy POST does.

    422 and `invalid-request-body`, not the 400 `invalid-query-parameter` a body error got before
    this surface had bodies to validate."""
    response = _patch(body)

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["type"] == "urn:litellm:error:invalid-request-body"
    key_write.update_data.assert_not_called()


def test_rejects_an_unknown_query_parameter(key_write, as_proxy_admin):
    """The strictness the list surface already has, which a write route does not get for free:
    the guard is a route dependency, and omitting it silently accepts the parameter."""
    with patch(
        "litellm.proxy.management_endpoints.key_management_endpoints._delete_cache_key_object",
        new=AsyncMock(),
    ):
        response = client.patch(
            f"{KEYS_PATH}/{HASHED_TOKEN}?bogus=1", json={"tpm_limit": 1}, headers={"Authorization": "Bearer k"}
        )

    assert response.status_code == 400
    assert response.json()["type"] == "urn:litellm:error:unknown-query-parameter"
    key_write.update_data.assert_not_called()


def test_identifier_mismatch_is_a_problem_document(key_write, as_proxy_admin):
    """The path is authoritative, and the refusal must not echo either identifier back."""
    response = _patch({"key_id": "a-different-key", "tpm_limit": 1})

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/problem+json")
    problem = response.json()
    assert problem["type"] == "urn:litellm:error:identifier-mismatch"
    assert "a-different-key" not in response.text
    assert HASHED_TOKEN not in response.text
    key_write.update_data.assert_not_called()


def test_a_missing_key_is_a_problem_document(key_write, as_proxy_admin):
    """The legacy write core raises the OpenAI error shape; this surface answers RFC 9457."""
    key_write.db.litellm_verificationtoken.find_unique = AsyncMock(return_value=None)

    response = _patch({"tpm_limit": 1})

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["type"] == "urn:litellm:error:key-not-found"
    key_write.update_data.assert_not_called()
