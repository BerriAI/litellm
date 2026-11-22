import json
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Final, Literal, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from litellm.types.mcp_server.events import CallbackFailure, JsonObject

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

SubscriptionStatus: TypeAlias = Literal["pending", "active", "suspended", "terminated"]
LIVE_STATUSES: Final[tuple[SubscriptionStatus, ...]] = ("pending", "active", "suspended")
PENDING_TTL: Final = timedelta(minutes=10)


class SubscriptionRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    principal: str
    key_hash: str | None = None
    user_id: str | None = None
    team_id: str | None = None
    server_id: str
    event_name: str
    arguments: JsonObject = Field(default_factory=dict)
    client_url: str
    client_secret: str
    upstream_subscription_id: str | None = None
    upstream_secret: str
    upstream_refresh_before: datetime | None = None
    refresh_before: datetime | None = None
    cursor: str | None = None
    status: SubscriptionStatus = "pending"
    verified_at: datetime | None = None
    last_delivery_at: datetime | None = None
    last_error: CallbackFailure | None = None
    failed_since: datetime | None = None
    consecutive_failures: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None


class Activation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    client_secret: str
    upstream_subscription_id: str | None
    upstream_refresh_before: datetime | None
    refresh_before: datetime | None
    cursor: str | None


class EventStore(Protocol):
    async def get(self, subscription_id: str) -> SubscriptionRow | None: ...

    async def reserve(self, row: SubscriptionRow) -> bool:
        """Insert ``row`` as pending unless a live row with its id exists; False when one does."""
        ...

    async def discard_pending(self, subscription_id: str) -> None: ...

    async def activate(
        self,
        subscription_id: str,
        activation: Activation,
        from_statuses: tuple[SubscriptionStatus, ...] = LIVE_STATUSES,
    ) -> bool:
        """Mark the row active with fresh secrets and deadlines while its status is still one of ``from_statuses``;
        False when it changed, so a renewal cannot revive a row an unsubscribe is holding."""
        ...


_SECRET_FIELDS: Final = ("client_secret", "upstream_secret")
_RECORD: Final = TypeAdapter[dict[str, object]](dict[str, object])
_NOT_TERMINATED: Final = {"not": "terminated"}


def _encrypt(value: str) -> str:
    from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper

    return encrypt_value_helper(value)


def _decrypt(value: str, field: str) -> str:
    from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper

    decrypted: Final = decrypt_value_helper(value, key=field, exception_type="debug")
    return decrypted if decrypted is not None else ""


def _row_from_record(record: BaseModel) -> SubscriptionRow:
    raw: Final = _RECORD.validate_python(record.model_dump())
    arguments: Final = raw.get("arguments")
    return SubscriptionRow.model_validate(
        {
            **raw,
            "arguments": json.loads(arguments) if isinstance(arguments, str) else arguments or {},
            **{field: _decrypt(str(raw[field]), field) for field in _SECRET_FIELDS},
        }
    )


def _record_from_row(row: SubscriptionRow) -> Mapping[str, object]:
    data: Final = _RECORD.validate_python(row.model_dump(exclude={"created_at", "updated_at"}))
    return {
        **data,
        "arguments": json.dumps(dict(row.arguments), sort_keys=True),
        "client_secret": _encrypt(row.client_secret),
        "upstream_secret": _encrypt(row.upstream_secret),
    }


def pending_row(row: SubscriptionRow) -> SubscriptionRow:
    return row.model_copy(update={"status": "pending", "refresh_before": datetime.now(timezone.utc) + PENDING_TTL})


class PrismaEventStore:
    def __init__(self, prisma_client: "PrismaClient") -> None:
        from litellm.repositories.table_repositories import MCPEventSubscriptionRepository

        # Every read follows a write it must see and the raw statements are UPDATEs, so a read replica is never used.
        self._subscriptions: Final = MCPEventSubscriptionRepository(prisma_client, use_writer=True).table

    async def get(self, subscription_id: str) -> SubscriptionRow | None:
        record: Final = await self._subscriptions.find_unique(where={"id": subscription_id})
        return _row_from_record(record) if record is not None else None

    async def reserve(self, row: SubscriptionRow) -> bool:
        from prisma.errors import UniqueViolationError

        data: Final = dict(_record_from_row(pending_row(row)))
        replaced: Final = await self._subscriptions.update_many(
            where={"id": row.id, "status": "terminated"},
            data={key: value for key, value in data.items() if key != "id"},
        )
        if replaced:
            return True
        try:
            await self._subscriptions.create(data=data)
        except UniqueViolationError:
            return False
        return True

    async def discard_pending(self, subscription_id: str) -> None:
        await self._subscriptions.delete_many(where={"id": subscription_id, "status": "pending"})

    async def activate(
        self,
        subscription_id: str,
        activation: Activation,
        from_statuses: tuple[SubscriptionStatus, ...] = LIVE_STATUSES,
    ) -> bool:
        updated: Final = await self._subscriptions.update_many(
            where={"id": subscription_id, "status": {"in": list(from_statuses)}},
            data={
                "status": "active",
                "client_secret": _encrypt(activation.client_secret),
                "upstream_subscription_id": activation.upstream_subscription_id,
                "upstream_refresh_before": activation.upstream_refresh_before,
                "refresh_before": activation.refresh_before,
                "cursor": activation.cursor,
                "consecutive_failures": 0,
            },
        )
        return updated > 0

    async def terminate(self, subscription_id: str) -> None:
        await self._subscriptions.update_many(where={"id": subscription_id}, data={"status": "terminated"})

    async def delete(self, subscription_id: str) -> None:
        await self._subscriptions.delete_many(where={"id": subscription_id})
