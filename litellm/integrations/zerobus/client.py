"""
Writes rows to a Unity Catalog table through the Zerobus Ingest REST API.

Zerobus only accepts a Databricks OAuth token minted for its own resource and scoped to
the target table's privileges, so the client mints that token itself with the service
principal's client credentials and reuses it until shortly before it expires.
"""

import asyncio
import base64
import json
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Final

import httpx
from pydantic import BaseModel, ValidationError

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.integrations.zerobus import (
    RETRYABLE_INGEST_STATUS_CODES,
    TOKEN_REFRESH_LEEWAY_SECONDS,
    ZerobusAccessToken,
    ZerobusConnection,
    ZerobusIngestFailure,
)

TOKEN_PATH: Final = "/oidc/v1/token"
OAUTH_SCOPE: Final = "all-apis"


class _TokenResponse(BaseModel):
    access_token: str
    expires_in: float = 3600


class ZerobusIngestError(Exception):
    """A batch could not be written and the failure is worth retrying."""


def zerobus_resource(workspace_id: str) -> str:
    return f"api://databricks/workspaces/{workspace_id}/zerobusDirectWriteApi"


def authorization_details(table_name: str) -> str:
    """The Unity Catalog privileges Zerobus requires the token to carry, as the token endpoint expects them."""
    catalog, schema, _table = table_name.split(".", 2)
    return json.dumps(
        (
            {
                "type": "unity_catalog_privileges",
                "privileges": ("USE CATALOG",),
                "object_type": "CATALOG",
                "object_full_path": catalog,
            },
            {
                "type": "unity_catalog_privileges",
                "privileges": ("USE SCHEMA",),
                "object_type": "SCHEMA",
                "object_full_path": f"{catalog}.{schema}",
            },
            {
                "type": "unity_catalog_privileges",
                "privileges": ("SELECT", "MODIFY"),
                "object_type": "TABLE",
                "object_full_path": table_name,
            },
        )
    )


def insert_url(connection: ZerobusConnection) -> str:
    return f"{connection.server_endpoint.rstrip('/')}/zerobus/v1/tables/{connection.table_name}/insert"


def token_url(connection: ZerobusConnection) -> str:
    return f"{connection.workspace_url.rstrip('/')}{TOKEN_PATH}"


def _basic_auth(client_id: str, client_secret: str) -> str:
    return "Basic " + base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()


def _status_failure(what: str, error: httpx.HTTPStatusError) -> ZerobusIngestFailure:
    status: Final = error.response.status_code
    return ZerobusIngestFailure(
        detail=f"{what} returned {status}: {error.response.text}"[:500],
        retryable=status in RETRYABLE_INGEST_STATUS_CODES,
    )


class ZerobusIngestClient:
    def __init__(
        self,
        connection: ZerobusConnection,
        http_client: AsyncHTTPHandler,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.connection: Final = connection
        self.http_client: Final = http_client
        self.clock: Final = clock
        self._token: ZerobusAccessToken | None = None
        self._token_lock: Final = asyncio.Lock()

    async def insert(self, rows: Sequence[Mapping[str, object]]) -> ZerobusIngestFailure | None:
        """Write ``rows`` as one request. ``None`` means Zerobus accepted every row."""
        token: Final = await self.access_token()
        if isinstance(token, ZerobusIngestFailure):
            return token
        try:
            await self.http_client.post(
                insert_url(self.connection),
                content=json.dumps([dict(row) for row in rows]).encode(),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {token.value}"},
            )
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 401:
                self._token = None
                return ZerobusIngestFailure(detail="insert returned 401, token discarded", retryable=True)
            return _status_failure("insert", error)
        except (httpx.HTTPError, litellm.Timeout) as error:
            return ZerobusIngestFailure(detail=f"insert failed: {error}", retryable=True)
        return None

    async def access_token(self) -> ZerobusAccessToken | ZerobusIngestFailure:
        """The cached token while it has more than the leeway left, otherwise a fresh one."""
        async with self._token_lock:
            cached: Final = self._token
            if cached is not None and cached.expires_at - self.clock() > TOKEN_REFRESH_LEEWAY_SECONDS:
                return cached
            minted: Final = await self._mint_token()
            if isinstance(minted, ZerobusAccessToken):
                self._token = minted
            return minted

    async def _mint_token(self) -> ZerobusAccessToken | ZerobusIngestFailure:
        connection: Final = self.connection
        try:
            response: Final = await self.http_client.post(
                token_url(connection),
                data={
                    "grant_type": "client_credentials",
                    "scope": OAUTH_SCOPE,
                    "resource": zerobus_resource(connection.workspace_id),
                    "authorization_details": authorization_details(connection.table_name),
                },
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Authorization": _basic_auth(connection.client_id, connection.client_secret),
                },
            )
        except httpx.HTTPStatusError as error:
            return _status_failure("token request", error)
        except (httpx.HTTPError, litellm.Timeout) as error:
            return ZerobusIngestFailure(detail=f"token request failed: {error}", retryable=True)
        try:
            parsed: Final = _TokenResponse.model_validate_json(response.text)
        except ValidationError as error:
            return ZerobusIngestFailure(detail=f"token response was not understood: {error}", retryable=False)
        return ZerobusAccessToken(value=parsed.access_token, expires_at=self.clock() + parsed.expires_in)
