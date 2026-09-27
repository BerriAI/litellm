"""Read-back for the azure_storage logging tests against the real filesystem the
proxy ships StandardLoggingPayload objects to (litellm_settings.callbacks:
["azure_storage"]).

Delivery is judged on what actually landed in the filesystem: the proxy writes
with its own credentials exactly as in production (account key or Entra ID).
The object layout differs by credential: the account-key path writes
{date}/{id}.json under a per-day directory, while the Entra ID path writes
{id}.json at the filesystem root, so reads match on the {id}.json name
wherever it sits. The tests list and download the objects back with the Azure
Data Lake SDK
(already a litellm proxy dependency, so the e2e runner image carries it; it is
an Azure SDK, not a raw HTTP client, so the e2e_http-only transport rule is
untouched). The filesystem comes from AZURE_STORAGE_ACCOUNT_NAME +
AZURE_STORAGE_FILE_SYSTEM - on the cluster the secret manager injects them,
locally tests/e2e/.env provides them. Missing configuration is a hard failure,
never a skip.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

import pytest
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import ClientSecretCredential
from azure.storage.filedatalake import DataLakeServiceClient
from pydantic import BaseModel, ConfigDict

from e2e_config import POLL_INTERVAL, POLL_TIMEOUT

DEFAULT_ENDPOINT_SUFFIX = "core.windows.net"


class AzureLogRecord(BaseModel):
    """The StandardLoggingPayload fields the azure_storage scenarios pin."""

    model_config = ConfigDict(extra="ignore")

    id: str
    status: str
    model_group: str | None = None
    response_cost: float | None = None
    total_tokens: int | None = None
    error_str: str | None = None


def _candidate_days() -> tuple[str, ...]:
    """The date directories an account-key-written object can sit under: the
    proxy names them from its host's local date. Only used in failure messages,
    since reads match the {id}.json name wherever it sits."""
    now = datetime.now()
    yesterday = now - timedelta(days=1)
    return tuple(dict.fromkeys((now.strftime("%Y-%m-%d"), yesterday.strftime("%Y-%m-%d"))))


@dataclass(frozen=True, slots=True)
class AzureStorageLogReader:
    file_system: str
    client: DataLakeServiceClient

    def _paths_for_id(self, response_id: str) -> list[str]:
        fs_client = self.client.get_file_system_client(self.file_system)
        try:
            return [
                path.name
                for path in fs_client.get_paths()
                if path.name == f"{response_id}.json" or path.name.endswith(f"/{response_id}.json")
            ]
        except ResourceNotFoundError:
            return []

    def read_record(self, path: str) -> AzureLogRecord:
        file_client = self.client.get_file_client(self.file_system, path)
        body = file_client.download_file().readall()
        return AzureLogRecord.model_validate_json(body)

    def count_objects_for_id(self, response_id: str) -> int:
        """Exactly-one check: distinct-path duplicates are what this catches.
        A duplicate write that reuses the exact same path overwrites the first
        object and no listing can see it."""
        return len(self._paths_for_id(response_id))

    def poll_record(
        self, response_id: str, *, timeout: float = POLL_TIMEOUT, interval: float = POLL_INTERVAL
    ) -> AzureLogRecord:
        """Poll the filesystem for the {response_id}.json object (under a per-day
        directory for account-key auth, at the root for Entra ID auth) until it
        exists - the callback flushes on a timer - then download and parse it.
        A timeout is a hard failure."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            paths = self._paths_for_id(response_id)
            if paths:
                return self.read_record(paths[0])
            time.sleep(interval)
        pytest.fail(
            f"no azure_storage object {response_id}.json (under {_candidate_days()} for account-key "
            f"auth, at the filesystem root for Entra ID auth) reached the filesystem within {timeout}s"
        )


def build_azure_storage_reader() -> AzureStorageLogReader:
    account = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME", "")
    file_system = os.environ.get("AZURE_STORAGE_FILE_SYSTEM", "")
    if not (account and file_system):
        pytest.fail(
            "AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_FILE_SYSTEM must be set: the "
            "azure_storage tests read the proxy's delivery back from the real filesystem "
            "(the cluster secret manager injects them; locally set them in tests/e2e/.env "
            "to the same values the proxy carries)"
        )
    suffix = os.environ.get("AZURE_STORAGE_ENDPOINT_SUFFIX", DEFAULT_ENDPOINT_SUFFIX)
    account_key = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY", "")
    if account_key:
        credential: object = account_key
    else:
        tenant = os.environ.get("AZURE_STORAGE_TENANT_ID", "")
        client_id = os.environ.get("AZURE_STORAGE_CLIENT_ID", "")
        client_secret = os.environ.get("AZURE_STORAGE_CLIENT_SECRET", "")
        if not (tenant and client_id and client_secret):
            pytest.fail(
                "the azure_storage tests need credentials to read the filesystem back: set "
                "AZURE_STORAGE_ACCOUNT_KEY, or all of AZURE_STORAGE_TENANT_ID / "
                "AZURE_STORAGE_CLIENT_ID / AZURE_STORAGE_CLIENT_SECRET (Entra ID)"
            )
        credential = ClientSecretCredential(tenant, client_id, client_secret)
    return AzureStorageLogReader(
        file_system=file_system,
        client=DataLakeServiceClient(f"https://{account}.dfs.{suffix}", credential=credential),
    )
