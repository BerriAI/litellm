"""
Ownership tests for the proxy video endpoints.

Requests go through the real FastAPI routes and the real ownership module. Only two
I/O boundaries are replaced: the provider call (``base_process_llm_request``) and the
Prisma ``litellm_managedobjecttable`` delegate, which is an in-memory table honoring
the subset of the query API the ownership module uses.
"""

import uuid
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import litellm.proxy.proxy_server as proxy_server
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth, hash_token
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.video_endpoints import endpoints
from litellm.types.videos.utils import encode_video_id_with_provider

ALICE = UserAPIKeyAuth(user_id="alice", api_key=hash_token("sk-alice"))
BOB = UserAPIKeyAuth(user_id="bob", api_key=hash_token("sk-bob"))
KEY_ONLY_A = UserAPIKeyAuth(api_key=hash_token("sk-service-a"))
KEY_ONLY_B = UserAPIKeyAuth(api_key=hash_token("sk-service-b"))
ADMIN = UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN)


def _matches(row: SimpleNamespace, where: dict) -> bool:
    for field, condition in where.items():
        value = getattr(row, field, None)
        if isinstance(condition, dict):
            if value not in condition["in"]:
                return False
        elif value != condition:
            return False
    return True


class InMemoryManagedObjectTable:
    def __init__(self, fail_writes: bool = False):
        self.rows: dict[str, SimpleNamespace] = {}
        self.fail_writes = fail_writes

    async def upsert(self, where: dict, data: dict) -> SimpleNamespace:
        if self.fail_writes:
            raise RuntimeError("database unavailable")
        key = where["model_object_id"]
        if key in self.rows:
            self.rows[key] = SimpleNamespace(**{**vars(self.rows[key]), **data["update"]})
        else:
            self.rows[key] = SimpleNamespace(**data["create"])
        return self.rows[key]

    async def find_first(self, where: dict) -> SimpleNamespace | None:
        return next((row for row in self.rows.values() if _matches(row, where)), None)

    async def find_many(self, where: dict) -> list[SimpleNamespace]:
        return [row for row in self.rows.values() if _matches(row, where)]


class Provider:
    """The provider behind ``base_process_llm_request``: creates and serves videos."""

    def __init__(self):
        self.calls: list[str] = []

    async def respond(self, processor, *, route_type: str, **kwargs):
        self.calls.append(route_type)
        if route_type in ("avideo_generation", "avideo_remix", "avideo_edit", "avideo_extension"):
            return {"id": f"video_{uuid.uuid4().hex}", "object": "video", "status": "queued"}
        if route_type == "avideo_content":
            return b"mp4-bytes"
        if route_type == "avideo_list":
            return {"object": "list", "data": processor.data["listed"], "has_more": False}
        return {"id": processor.data["video_id"], "object": "video", "status": "completed"}


@pytest.fixture
def provider(monkeypatch) -> Provider:
    fake = Provider()

    async def base_process_llm_request(processor, **kwargs):
        return await fake.respond(processor, **kwargs)

    monkeypatch.setattr(ProxyBaseLLMRequestProcessing, "base_process_llm_request", base_process_llm_request)
    return fake


@pytest.fixture
def table(monkeypatch) -> InMemoryManagedObjectTable:
    rows = InMemoryManagedObjectTable()
    monkeypatch.setattr(
        proxy_server, "prisma_client", SimpleNamespace(db=SimpleNamespace(litellm_managedobjecttable=rows))
    )
    return rows


def _client(auth: UserAPIKeyAuth) -> TestClient:
    app = FastAPI()
    app.include_router(endpoints.router)
    app.dependency_overrides[user_api_key_auth] = lambda: auth
    return TestClient(app)


def _create_video(auth: UserAPIKeyAuth) -> str:
    response = _client(auth).post("/v1/videos", json={"model": "sora-2", "prompt": "a sunset"})
    assert response.status_code == 200
    return response.json()["id"]


def _access(auth: UserAPIKeyAuth, video_id: str) -> dict[str, int]:
    client = _client(auth)
    return {
        "status": client.get(f"/v1/videos/{video_id}").status_code,
        "content": client.get(f"/v1/videos/{video_id}/content").status_code,
        "remix": client.post(f"/v1/videos/{video_id}/remix", json={"prompt": "again"}).status_code,
        "edit": client.post("/v1/videos/edits", json={"prompt": "brighter", "video": {"id": video_id}}).status_code,
        "extension": client.post(
            "/v1/videos/extensions", json={"prompt": "continue", "video": {"id": video_id}}
        ).status_code,
    }


ALL_OK = {"status": 200, "content": 200, "remix": 200, "edit": 200, "extension": 200}
ALL_FORBIDDEN = {"status": 403, "content": 403, "remix": 403, "edit": 403, "extension": 403}


def test_owner_can_use_every_video_route(provider, table):
    video_id = _create_video(ALICE)

    assert table.rows[f"video:{video_id}"].created_by == "alice"
    assert _access(ALICE, video_id) == ALL_OK


def test_other_user_is_forbidden_and_the_provider_is_never_called(provider, table):
    video_id = _create_video(ALICE)
    provider.calls.clear()

    assert _access(BOB, video_id) == ALL_FORBIDDEN
    assert provider.calls == []


def test_key_scoped_owner_blocks_a_different_key(provider, table):
    video_id = _create_video(KEY_ONLY_A)

    assert _access(KEY_ONLY_B, video_id) == ALL_FORBIDDEN
    assert _access(KEY_ONLY_A, video_id) == ALL_OK


def test_proxy_admin_can_access_any_video(provider, table):
    video_id = _create_video(ALICE)

    assert _access(ADMIN, video_id) == ALL_OK


def test_untracked_video_is_admin_only_when_a_database_is_connected(provider, table):
    video_id = f"video_{uuid.uuid4().hex}"

    assert _access(ALICE, video_id) == ALL_FORBIDDEN
    assert _access(ADMIN, video_id) == ALL_OK


def test_rewrapping_the_provider_id_does_not_bypass_the_owner_check(provider, table):
    video_id = _create_video(ALICE)
    rewrapped = encode_video_id_with_provider(video_id, "azure", "attacker-deployment")

    assert rewrapped != video_id
    assert _access(BOB, rewrapped) == ALL_FORBIDDEN


def test_videos_derived_from_an_owned_video_belong_to_the_caller(provider, table):
    video_id = _create_video(ALICE)
    remixed = _client(ALICE).post(f"/v1/videos/{video_id}/remix", json={"prompt": "again"}).json()["id"]

    assert _access(ALICE, remixed)["status"] == 200
    assert _access(BOB, remixed)["status"] == 403


def test_list_only_returns_the_callers_videos(provider, table):
    alice_video = _create_video(ALICE)
    bob_video = _create_video(BOB)
    listed = [{"id": alice_video, "object": "video"}, {"id": bob_video, "object": "video"}]
    provider_listing = {"listed": listed}

    def list_as(auth: UserAPIKeyAuth) -> list[str]:
        async def with_listing(processor, **kwargs):
            processor.data.update(provider_listing)
            return await provider.respond(processor, **kwargs)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(ProxyBaseLLMRequestProcessing, "base_process_llm_request", with_listing)
            response = _client(auth).get("/v1/videos")
        assert response.status_code == 200
        return [item["id"] for item in response.json()["data"]]

    assert list_as(ALICE) == [alice_video]
    assert list_as(BOB) == [bob_video]
    assert list_as(ADMIN) == [alice_video, bob_video]


def test_without_a_database_ownership_is_not_enforced(provider, monkeypatch):
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    video_id = _create_video(ALICE)

    assert _access(BOB, video_id) == ALL_OK


def test_failed_ownership_write_still_returns_the_created_video(provider, table):
    table.fail_writes = True

    video_id = _create_video(ALICE)

    assert video_id.startswith("video_")
    assert table.rows == {}
    assert _access(ALICE, video_id)["status"] == 403
