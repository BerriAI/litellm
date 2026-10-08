import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final
from uuid import uuid4

from pydantic import AwareDatetime, Field

from litellm.proxy.lens.models import Record


class IngestionKeyRequest(Record):
    name: str = Field(default="Agent tracing", min_length=1, max_length=128)
    team_id: str = Field(default="", max_length=256)
    expires_at: AwareDatetime | None = None


class IngestionTenant(Record):
    team_id: str = ""
    user_id: str
    org_id: str = ""
    api_key_hash: str


class IngestionKey(Record):
    id: str
    name: str
    tenant: IngestionTenant
    created_at: AwareDatetime
    expires_at: int | None


class IngestionCredential(Record):
    token_hash: str
    tenant: IngestionTenant
    expires_at: int | None


class IngestionSnapshot(Record):
    issued_at: int
    keys: tuple[IngestionCredential, ...]


class IngestionKeyCreated(Record):
    key: str
    record: IngestionKey
    active: bool = False


class ServiceStatus(Record):
    storage_ready: bool = False
    credentials_ready: bool = False
    release: str = ""
    protocol_version: int = 0


class ServiceConnection(Record):
    url: str
    connected: bool
    status: ServiceStatus


@dataclass(frozen=True, slots=True)
class InvalidExpiry:
    pass


def new_key(request: IngestionKeyRequest, user_id: str) -> IngestionKeyCreated | InvalidExpiry:
    now: Final = datetime.now(timezone.utc)
    if request.expires_at is not None and request.expires_at <= now:
        return InvalidExpiry()
    token: Final = f"lens-trace-{int(now.timestamp())}-" + secrets.token_urlsafe(40)
    digest: Final = hashlib.sha256(token.encode()).hexdigest()
    return IngestionKeyCreated(
        key=token,
        record=IngestionKey(
            id=str(uuid4()),
            name=request.name,
            tenant=IngestionTenant(team_id=request.team_id, user_id=user_id, api_key_hash=digest),
            created_at=now,
            expires_at=int(request.expires_at.timestamp()) if request.expires_at is not None else None,
        ),
    )
