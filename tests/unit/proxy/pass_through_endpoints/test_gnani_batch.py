from collections.abc import Iterator
from contextlib import closing
from typing import Final

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

import litellm
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.proxy_server import app

JOBS: Final = "/gnani/stt/v3/batch/jobs"


@pytest.fixture
def admin(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("GNANI_API_KEY", "server-key")
    monkeypatch.setenv("GNANI_API_BASE", "https://gnani.example/prefix")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    monkeypatch.setitem(
        app.dependency_overrides, user_api_key_auth, lambda: UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    )
    with closing(TestClient(app)) as client:
        yield client


def test_admin_upload_preserves_multipart_audio_and_uses_proxy_credentials(admin: TestClient) -> None:
    upload: Final = httpx.Request(
        "POST",
        "http://testserver" + JOBS,
        files=[
            ("config", (None, '{"model":"prisma-v3.0","language_code":"en-IN"}', "application/json")),
            ("files", ("first.wav", b"RIFF\x00\xff", "audio/wav")),
            ("files", ("second.wav", b"RIFF\xff\x80", "audio/wav")),
        ],
    )
    audio: Final = upload.read()
    created: Final = {"job_id": "00000000-0000-4000-8000-000000000001", "status": "CREATED"}
    with respx.mock as provider:
        route: Final = provider.post("https://gnani.example/prefix/stt/v3/batch/jobs")
        route.return_value = httpx.Response(201, json=created)
        result: Final = admin.post(
            JOBS,
            content=audio,
            headers={
                "Content-Type": upload.headers["content-type"],
                "Authorization": "Bearer gateway-key",
                "x-pass-x-api-key-id": "caller-supplied-key",
            },
        )
    assert (result.status_code, result.json()) == (201, created)
    sent: Final = route.calls.last.request
    assert sent.content == audio
    assert sent.headers["content-type"] == upload.headers["content-type"]
    assert sent.headers["x-api-key-id"] == "server-key"
    assert "authorization" not in sent.headers


def test_gateway_submits_cloud_source_and_returns_paginated_jobs(admin: TestClient) -> None:
    body: Final = {
        "config": {"model": "prisma-v2.5", "language_code": "en-IN", "with_diarization": True},
        "source": {"type": "cloud_storage", "auth": {"mode": "public"}, "paths": ["https://media.example/call.wav"]},
        "callback_url": "https://app.example/callback",
    }
    job: Final = {"job_id": "00000000-0000-4000-8000-000000000001", "status": "CREATED"}
    page: Final = {"data": [job], "pagination": {"has_more": False, "next_cursor": None}}
    params: Final = {"status": "CREATED", "limit": "1", "cursor": "a+/="}
    upstream: Final = "https://gnani.example/prefix/stt/v3/batch/jobs"
    with respx.mock as provider:
        provider.post(upstream, json=body).return_value = httpx.Response(201, json=job)
        provider.get(upstream, params=params).return_value = httpx.Response(200, json=page)
        created: Final = admin.post(JOBS, json=body)
        listed: Final = admin.get(JOBS, params=params)
    assert (created.status_code, created.json()) == (201, job)
    assert (listed.status_code, listed.json()) == (200, page)


def test_gateway_returns_the_provider_redirect_to_the_caller(admin: TestClient) -> None:
    with respx.mock as provider:
        route: Final = provider.get("https://gnani.example/prefix/stt/v3/batch/jobs")
        route.return_value = httpx.Response(307, headers={"Location": "/status"}, json={"message": "Moved temporarily"})
        response: Final = admin.get(JOBS, follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/status"
    assert response.json() == {"message": "Moved temporarily"}


def test_presigned_job_creation_returns_the_storage_upload_url(admin: TestClient) -> None:
    body: Final = {
        "config": {"model": "prisma-v3.0", "language_code": "en-IN"},
        "source": {"type": "direct_upload_url", "files": [{"filename": "meeting.flac"}]},
    }
    created: Final = {
        "job_id": "00000000-0000-4000-8000-000000000001",
        "uploads": [{"filename": "meeting.flac", "url": "https://storage.example/meeting.flac?signature=value"}],
    }
    with respx.mock as provider:
        provider.post("https://gnani.example/prefix/stt/v3/batch/v2/jobs", json=body).return_value = httpx.Response(
            201, json=created
        )
        response: Final = admin.post("/gnani/stt/v3/batch/v2/jobs", json=body)
    assert (response.status_code, response.json()) == (201, created)


@pytest.mark.parametrize(
    ("method", "suffix"),
    (
        ("GET", "/jobs"),
        ("POST", "/jobs"),
        ("POST", "/v2/jobs"),
        ("GET", "/jobs/00000000-0000-4000-8000-000000000001"),
        ("GET", "/jobs/00000000-0000-4000-8000-000000000001/files"),
        ("POST", "/jobs/00000000-0000-4000-8000-000000000001/start"),
        ("POST", "/jobs/00000000-0000-4000-8000-000000000001/cancel"),
    ),
)
def test_shared_audio_jobs_require_an_admin(method: str, suffix: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(app.dependency_overrides, user_api_key_auth, lambda: UserAPIKeyAuth(api_key="virtual-key"))
    with closing(TestClient(app)) as caller, respx.mock as provider:
        response: Final = caller.request(method, "/gnani/stt/v3/batch" + suffix)
    assert response.status_code == 403
    assert not provider.calls


def test_cancel_preserves_provider_conflict_and_rejects_the_wrong_method(admin: TestClient) -> None:
    cancel: Final = "/stt/v3/batch/jobs/00000000-0000-4000-8000-000000000001/cancel"
    conflict: Final = {"error": "JOB_CANCELLED", "message": "Job is already cancelled"}
    with respx.mock as provider:
        route: Final = provider.post("https://gnani.example/prefix" + cancel, json={"reason": "Wrong recording"})
        route.return_value = httpx.Response(409, json=conflict)
        denied: Final = admin.get("/gnani" + cancel)
        response: Final = admin.post("/gnani" + cancel, json={"reason": "Wrong recording"})
    assert (denied.status_code, denied.headers["allow"]) == (405, "POST")
    assert (response.status_code, response.json()) == (409, conflict)
    assert route.call_count == 1
