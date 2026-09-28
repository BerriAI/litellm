import base64
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain, repeat

import httpx
import pytest

from litellm.integrations.zerobus.client import ZerobusIngestClient
from litellm.types.integrations.zerobus import ZerobusAccessToken, ZerobusConnection, ZerobusIngestFailure

CONNECTION = ZerobusConnection(
    workspace_url="https://dbc-a1b2c3d4-e5f6.cloud.databricks.com/",
    workspace_id="1234567890123456",
    server_endpoint="https://1234567890123456.zerobus.us-west-2.cloud.databricks.com",
    client_id="sp-client-id",
    client_secret="sp-client-secret",
    table_name="main.litellm.traces",
)
ROWS = ({"id": "a", "model": "gpt-4o"}, {"id": "b", "model": "gpt-4o"})


def _token(value: str = "tok-1", expires_in: float = 3600) -> httpx.Response:
    return httpx.Response(200, text=json.dumps({"access_token": value, "expires_in": expires_in}))


def _accepted() -> httpx.Response:
    return httpx.Response(200, text="{}")


@dataclass(frozen=True, slots=True)
class TokenCall:
    url: str
    data: Mapping[str, str]
    headers: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class InsertCall:
    url: str
    content: bytes
    headers: Mapping[str, str]


def _results(results: Sequence[httpx.Response | Exception]) -> Iterator[httpx.Response | Exception]:
    """Results are served in order, and the last one repeats."""
    return chain(results[:-1], repeat(results[-1]))


class FakeHTTPClient:
    """Stands in for AsyncHTTPHandler, including its habit of raising on error statuses."""

    def __init__(
        self,
        token: Sequence[httpx.Response | Exception] = (),
        insert: Sequence[httpx.Response | Exception] = (),
    ) -> None:
        self.token_results = _results(token or (_token(),))
        self.insert_results = _results(insert or (_accepted(),))
        self.token_calls: tuple[TokenCall, ...] = ()
        self.insert_calls: tuple[InsertCall, ...] = ()

    async def post(
        self,
        url: str,
        data: Mapping[str, str] | None = None,
        content: bytes | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        if url.endswith("/oidc/v1/token"):
            self.token_calls = (*self.token_calls, TokenCall(url, data or {}, headers or {}))
            return _raise_like_the_handler(next(self.token_results), url)
        self.insert_calls = (*self.insert_calls, InsertCall(url, content or b"", headers or {}))
        return _raise_like_the_handler(next(self.insert_results), url)


def _raise_like_the_handler(result: httpx.Response | Exception, url: str) -> httpx.Response:
    if isinstance(result, Exception):
        raise result
    if result.status_code >= 300:
        raise httpx.HTTPStatusError(
            "boom",
            request=httpx.Request("POST", url),
            response=httpx.Response(result.status_code, text=result.text),
        )
    return result


class FakeClock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _client(http_client: FakeHTTPClient, clock: FakeClock | None = None) -> ZerobusIngestClient:
    return ZerobusIngestClient(connection=CONNECTION, http_client=http_client, clock=clock or FakeClock())


@pytest.mark.asyncio
async def test_rows_are_posted_as_one_json_list_to_the_table_insert_endpoint():
    http_client = FakeHTTPClient()

    outcome = await _client(http_client).insert(ROWS)

    assert outcome is None
    (call,) = http_client.insert_calls
    # Insert endpoint per the Zerobus Ingest docs, read 2026-09-19:
    # https://docs.databricks.com/aws/en/ingestion/lakeflow-connect/zerobus-ingest
    assert call.url == (
        "https://1234567890123456.zerobus.us-west-2.cloud.databricks.com/zerobus/v1/tables/main.litellm.traces/insert"
    )
    assert json.loads(call.content) == [{"id": "a", "model": "gpt-4o"}, {"id": "b", "model": "gpt-4o"}]
    assert call.headers["Content-Type"] == "application/json"
    assert call.headers["Authorization"] == "Bearer tok-1"


@pytest.mark.asyncio
async def test_the_token_is_minted_for_the_zerobus_resource_with_the_table_privileges():
    """Zerobus refuses a plain workspace token: it must name its own resource and the table's UC privileges."""
    http_client = FakeHTTPClient()

    await _client(http_client).insert(ROWS)

    (call,) = http_client.token_calls
    # Token form per the Zerobus Ingest docs (REST API authentication), read 2026-09-19:
    # https://docs.databricks.com/aws/en/ingestion/lakeflow-connect/zerobus-ingest
    assert call.url == "https://dbc-a1b2c3d4-e5f6.cloud.databricks.com/oidc/v1/token"
    assert call.data["grant_type"] == "client_credentials"
    assert call.data["scope"] == "all-apis"
    assert call.data["resource"] == "api://databricks/workspaces/1234567890123456/zerobusDirectWriteApi"
    details = json.loads(call.data["authorization_details"])
    assert [(d["object_type"], d["object_full_path"], d["privileges"]) for d in details] == [
        ("CATALOG", "main", ["USE CATALOG"]),
        ("SCHEMA", "main.litellm", ["USE SCHEMA"]),
        ("TABLE", "main.litellm.traces", ["SELECT", "MODIFY"]),
    ]
    assert all(d["type"] == "unity_catalog_privileges" for d in details)


@pytest.mark.asyncio
async def test_the_service_principal_authenticates_with_http_basic():
    http_client = FakeHTTPClient()

    await _client(http_client).insert(ROWS)

    scheme, credentials = http_client.token_calls[0].headers["Authorization"].split(" ")
    assert scheme == "Basic"
    assert base64.b64decode(credentials).decode() == "sp-client-id:sp-client-secret"


def test_the_client_secret_and_minted_token_stay_out_of_reprs_and_tracebacks():
    token = ZerobusAccessToken(value="tok-secret", expires_at=1.0)

    assert "sp-client-secret" not in repr(CONNECTION)
    assert "sp-client-id" in repr(CONNECTION)
    assert "tok-secret" not in repr(token)
    assert "expires_at=1.0" in repr(token)


@pytest.mark.asyncio
async def test_the_token_is_reused_across_inserts_until_it_nears_expiry():
    clock = FakeClock(now=1_000.0)
    http_client = FakeHTTPClient(token=[_token("tok-1", expires_in=600), _token("tok-2")])
    client = _client(http_client, clock)

    await client.insert(ROWS)
    clock.now = 1_000.0 + 600 - 61
    await client.insert(ROWS)
    clock.now = 1_000.0 + 600 - 59
    await client.insert(ROWS)

    assert len(http_client.token_calls) == 2
    assert [call.headers["Authorization"] for call in http_client.insert_calls] == [
        "Bearer tok-1",
        "Bearer tok-1",
        "Bearer tok-2",
    ]


@pytest.mark.asyncio
async def test_a_401_discards_the_token_so_the_next_insert_mints_a_fresh_one():
    http_client = FakeHTTPClient(
        token=[_token("tok-1"), _token("tok-2")],
        insert=[httpx.Response(401, text="expired"), _accepted()],
    )
    client = _client(http_client)

    first = await client.insert(ROWS)
    second = await client.insert(ROWS)

    assert first == ZerobusIngestFailure(detail="insert returned 401, token discarded", retryable=True)
    assert second is None
    assert http_client.insert_calls[1].headers["Authorization"] == "Bearer tok-2"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 500, 503])
async def test_a_transient_insert_status_is_retryable(status: int):
    http_client = FakeHTTPClient(insert=[httpx.Response(status, text="later")])

    outcome = await _client(http_client).insert(ROWS)

    assert isinstance(outcome, ZerobusIngestFailure)
    assert outcome.retryable is True
    assert str(status) in outcome.detail


@pytest.mark.asyncio
async def test_a_schema_rejection_is_not_retryable_and_says_why():
    http_client = FakeHTTPClient(insert=[httpx.Response(400, text="unknown column foo")])

    outcome = await _client(http_client).insert(ROWS)

    assert outcome == ZerobusIngestFailure(detail="insert returned 400: unknown column foo", retryable=False)


@pytest.mark.asyncio
async def test_a_network_failure_on_insert_is_retryable():
    http_client = FakeHTTPClient(insert=[httpx.ConnectError("connection refused")])

    outcome = await _client(http_client).insert(ROWS)

    assert isinstance(outcome, ZerobusIngestFailure)
    assert outcome.retryable is True


@pytest.mark.asyncio
async def test_bad_credentials_fail_the_insert_without_posting_rows():
    http_client = FakeHTTPClient(token=[httpx.Response(401, text="invalid_client")])

    outcome = await _client(http_client).insert(ROWS)

    assert outcome == ZerobusIngestFailure(detail="token request returned 401: invalid_client", retryable=False)
    assert http_client.insert_calls == ()


@pytest.mark.asyncio
async def test_a_token_endpoint_outage_is_retryable():
    http_client = FakeHTTPClient(token=[httpx.Response(503, text="try later")])

    outcome = await _client(http_client).insert(ROWS)

    assert isinstance(outcome, ZerobusIngestFailure)
    assert outcome.retryable is True


@pytest.mark.asyncio
async def test_a_token_response_without_a_token_is_reported_not_raised():
    http_client = FakeHTTPClient(token=[httpx.Response(200, text='{"token_type": "Bearer"}')])

    outcome = await _client(http_client).insert(ROWS)

    assert isinstance(outcome, ZerobusIngestFailure)
    assert outcome.retryable is False
    assert "token response" in outcome.detail
