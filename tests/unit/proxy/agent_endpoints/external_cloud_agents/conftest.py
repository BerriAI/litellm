import json
import uuid
from datetime import datetime, timezone
from typing import Final

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from typing_extensions import LiteralString

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.agent_endpoints.external_cloud_agents import endpoints
from litellm.proxy.agent_endpoints.external_cloud_agents.models import DailySpend, SourceRecord
from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper


NOW: Final = datetime(2026, 9, 18, 12, tzinfo=timezone.utc)
INGEST: Final = "test-ingest-credential"


class DatabaseDouble:
    def __init__(self) -> None:
        self.source: SourceRecord | None = SourceRecord(
            source_id=str(uuid.uuid4()), ingest_key=encrypt_value_helper(INGEST), tenant_id="tenant"
        )
        self.rows: object = ()
        self.writes: tuple[tuple[object, ...], ...] = ()
        self.persisted: tuple[DailySpend, ...] = ()
        self.fail_reads = False
        self.fail_writes = False

    async def query_raw(self, query: LiteralString, *args: object) -> object:
        if self.fail_reads:
            raise RuntimeError("database offline")
        if query.startswith('SELECT * FROM "LiteLLM_ExternalCloudAgentSource"'):
            return (self.source,) if self.source and self.source.source_id == args[0] else ()
        return self.rows

    async def execute_raw(self, query: LiteralString, *args: object) -> int:
        if self.fail_writes:
            raise RuntimeError("database offline")
        self.writes = (*self.writes, args)
        if query.startswith('INSERT INTO "LiteLLM_ExternalCloudAgentSource"'):
            self.source = SourceRecord(source_id=str(args[0]), ingest_key=str(args[1]))
        if self.source and "SET admin_api_key=$2" in query:
            self.source = self.source.model_copy(update={"admin_api_key": args[1]})
        if self.source and "SET tenant_id=$2" in query:
            self.source = self.source.model_copy(update={"tenant_id": self.source.tenant_id or args[1]})
        if "WITH eligible" in query:
            self.persisted = tuple(DailySpend.model_validate(row) for row in json.loads(str(args[1])))
        return 1


class Harness:
    def __init__(self, db: DatabaseDouble, client: httpx.AsyncClient) -> None:
        self.db = db
        self.http = client
        self.app = FastAPI()
        self.app.include_router(endpoints.router)
        self.app.dependency_overrides = {
            endpoints.database: lambda: db,
            endpoints.billing_client: lambda: client,
            endpoints.user_api_key_auth: lambda: UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin"
            ),
            endpoints.utc_now: lambda: NOW,
        }
        self.client = TestClient(self.app)

    @property
    def source_id(self) -> str:
        assert self.db.source is not None
        return self.db.source.source_id

    def post_hook(self, body: object, key: str = INGEST) -> httpx.Response:
        return self.client.post(
            f"/grok_bot/inference_hook/{self.source_id}", json=body, headers={"Authorization": f"Bearer {key}"}
        )


@pytest.fixture
def db(monkeypatch: pytest.MonkeyPatch) -> DatabaseDouble:
    monkeypatch.setenv("LITELLM_SALT_KEY", "unit-test-encryption-key")
    return DatabaseDouble()


@pytest.fixture
def harness(db: DatabaseDouble):
    client: Final = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(401)))
    instance: Final = Harness(db, client)
    with instance.client:
        yield instance
