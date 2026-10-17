import ipaddress
import json
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Annotated, Final, Literal, Protocol, TypeAlias, runtime_checkable
from urllib.parse import urlsplit

from pydantic import ConfigDict, Field, SecretStr, TypeAdapter, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from litellm.types.llms.base import LiteLLMBaseModel

Digest: TypeAlias = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
TokenCount: TypeAlias = Annotated[int, Field(strict=True, ge=0, le=2**63 - 1)]


class ChargeReceipt(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = 1
    event_id: str = Field(min_length=1, max_length=4096)
    request_id: str = Field(min_length=1, max_length=200)
    gateway_scope: Digest
    key_hash: Digest
    currency: Literal["USD"] = "USD"
    amount_usd: str = Field(pattern=r"^(?:0|[1-9][0-9]{0,15})(?:\.[0-9]{1,24})?$")
    model: str = Field(min_length=1, max_length=200)
    input_tokens: TokenCount | None = None
    output_tokens: TokenCount | None = None
    total_tokens: TokenCount | None = None
    cache_read_tokens: TokenCount | None = None
    cache_write_tokens: TokenCount | None = None
    reasoning_tokens: TokenCount | None = None

    def fingerprint(self) -> str:
        amount: Final = self.amount_usd.rstrip("0").rstrip(".") if "." in self.amount_usd else self.amount_usd
        canonical: Final = {**self.model_dump(), "amount_usd": amount}
        return sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ChargeReceiptCapabilities(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    version: Literal[1] = 1
    required: bool
    gateway_scope: Digest | None


class ChargeReceiptSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LITELLM_CHARGE_RECEIPTS_",
        extra="ignore",
        frozen=True,
        hide_input_in_errors=True,
    )

    enabled: bool = False
    gateway_scope: str = ""
    key_hashes: frozenset[Digest] = frozenset()
    destination_url: str = ""
    destination_secret: SecretStr = SecretStr("")
    commit_timeout_seconds: float = Field(default=5, gt=0, le=30)
    drain_timeout_seconds: float = Field(default=120, gt=0, le=600)
    delivery_timeout_seconds: float = Field(default=5, gt=0, le=10)
    poll_seconds: float = Field(default=0.1, ge=0.02, le=10)
    batch_size: int = Field(default=32, ge=1, le=32)

    def configuration_error(self) -> str | None:
        if not self.enabled:
            return None
        try:
            TypeAdapter(Digest).validate_python(self.gateway_scope)
        except ValidationError:
            return "Charge receipts require a gateway scope digest."
        secret: Final = self.destination_secret.get_secret_value()
        if (
            not self.key_hashes
            or len(secret) < 32
            or not secret.isascii()
            or any(character.isspace() for character in secret)
        ):
            return "Charge receipts require scoped key hashes and a destination secret of at least 32 characters."
        try:
            destination: Final = urlsplit(self.destination_url)
            _port: Final = destination.port
        except ValueError:
            return "Charge receipts require a valid fixed destination URL."
        if (
            not destination.hostname
            or destination.username
            or destination.password
            or destination.fragment
            or destination.query
            or not destination.path
        ):
            return "Charge receipts require a fixed destination URL without credentials, query, or fragment."
        if destination.scheme == "https":
            return None
        try:
            loopback: Final = (
                destination.hostname == "localhost" or ipaddress.ip_address(destination.hostname).is_loopback
            )
        except ValueError:
            return "Charge receipt delivery requires HTTPS or a loopback test destination."
        return None if destination.scheme == "http" and loopback else "Charge receipt delivery requires HTTPS."


@runtime_checkable
class ReceiptQuery(Protocol):
    def __call__(self, query: str, *parameters: object) -> Awaitable[Sequence[Mapping[str, object]]]: ...


@dataclass(frozen=True, slots=True)
class ReceiptCommit:
    event_id: str
    duplicate: bool
    key_counter_updated: bool


@dataclass(frozen=True, slots=True)
class ReceiptFailure:
    reason: Literal[
        "database_unavailable", "unsafe_durability", "receipt_conflict", "invalid_database_result", "invalid_charge"
    ]


class _StoredReceipt(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    event_id: str
    payload_hash: str
    key_counter_updated: bool


_STORED_RECEIPTS: Final = TypeAdapter(tuple[_StoredReceipt, ...])
_PENDING_RECEIPTS: Final = TypeAdapter(tuple[ChargeReceipt, ...])

APPEND_RECEIPT_SQL: Final = """
WITH parameters AS (
    SELECT $1::text AS event_id, $2::text AS key_hash, $3::jsonb AS payload,
           $4::text AS payload_hash, $5::double precision AS amount
), locked_key AS (
    SELECT k.token FROM "LiteLLM_VerificationToken" k, parameters p
    WHERE k.token = p.key_hash FOR UPDATE OF k
), inserted AS (
    INSERT INTO "LiteLLM_ChargeReceipt"
        (event_id, key_hash, payload, payload_hash, key_counter_updated)
    SELECT p.event_id, p.key_hash, p.payload, p.payload_hash, EXISTS(SELECT 1 FROM locked_key)
    FROM parameters p
    WHERE current_setting('synchronous_commit') IN ('on', 'remote_apply', 'remote_write')
      AND current_setting('fsync') = 'on'
    ON CONFLICT (event_id) DO NOTHING
    RETURNING event_id, key_hash, payload_hash, key_counter_updated
), charged AS (
    UPDATE "LiteLLM_VerificationToken" k
    SET spend = k.spend + p.amount, total_spend = k.total_spend + p.amount,
        last_active = clock_timestamp() AT TIME ZONE 'UTC', updated_at = clock_timestamp() AT TIME ZONE 'UTC'
    FROM inserted i, parameters p
    WHERE k.token = i.key_hash AND i.key_counter_updated
    RETURNING k.token
)
SELECT i.event_id, i.payload_hash, i.key_counter_updated
FROM inserted i LEFT JOIN charged c ON c.token = i.key_hash
"""

CLAIM_RECEIPTS_SQL: Final = """
WITH selected AS (
    SELECT event_id FROM "LiteLLM_ChargeReceipt"
    WHERE delivered_at IS NULL AND retry_at <= clock_timestamp()
      AND (lease_until IS NULL OR lease_until <= clock_timestamp())
    ORDER BY retry_at, event_id
    FOR UPDATE SKIP LOCKED LIMIT $1
)
UPDATE "LiteLLM_ChargeReceipt" r
SET lease_owner = $2, lease_until = clock_timestamp() + interval '30 seconds',
    attempts = r.attempts + 1
FROM selected s WHERE r.event_id = s.event_id
RETURNING r.payload
"""


class ChargeReceiptStore:
    def __init__(self, query: ReceiptQuery) -> None:
        self._query: Final = query

    async def ready(self) -> bool:
        try:
            rows: Final = await self._query(
                """SELECT to_regclass('"LiteLLM_ChargeReceipt"') IS NOT NULL AS installed,
                   current_setting('synchronous_commit') IN ('on','remote_apply','remote_write')
                   AND current_setting('fsync') = 'on' AS durable"""
            )
            return len(rows) == 1 and rows[0].get("installed") is True and rows[0].get("durable") is True
        except Exception:  # noqa: BLE001  # database adapters vary; never expose their credential-bearing errors
            return False

    async def append(self, receipt: ChargeReceipt) -> ReceiptCommit | ReceiptFailure:
        try:
            appended: Final = _STORED_RECEIPTS.validate_python(
                await self._query(
                    APPEND_RECEIPT_SQL,
                    receipt.event_id,
                    receipt.key_hash,
                    receipt.model_dump_json(),
                    receipt.fingerprint(),
                    receipt.amount_usd,
                )
            )
            if appended:
                return self._matching_commit(receipt, appended, duplicate=False)
            existing: Final = _STORED_RECEIPTS.validate_python(
                await self._query(
                    """SELECT event_id,payload_hash,key_counter_updated
                       FROM "LiteLLM_ChargeReceipt" WHERE event_id=$1""",
                    receipt.event_id,
                )
            )
            return (
                self._matching_commit(receipt, existing, duplicate=True)
                if existing
                else ReceiptFailure("unsafe_durability")
            )
        except ValidationError:
            return ReceiptFailure("invalid_database_result")
        except Exception:  # noqa: BLE001  # a lost commit ACK is retryable with the same immutable event identity
            return ReceiptFailure("database_unavailable")

    @staticmethod
    def _matching_commit(
        receipt: ChargeReceipt, rows: tuple[_StoredReceipt, ...], *, duplicate: bool
    ) -> ReceiptCommit | ReceiptFailure:
        if len(rows) != 1 or rows[0].event_id != receipt.event_id:
            return ReceiptFailure("invalid_database_result")
        if rows[0].payload_hash != receipt.fingerprint():
            return ReceiptFailure("receipt_conflict")
        return ReceiptCommit(receipt.event_id, duplicate, rows[0].key_counter_updated)

    async def claim(self, owner: str, limit: int) -> tuple[ChargeReceipt, ...]:
        rows: Final = await self._query(CLAIM_RECEIPTS_SQL, limit, owner)
        return _PENDING_RECEIPTS.validate_python(tuple(row["payload"] for row in rows))

    async def acknowledge(self, owner: str, event_ids: tuple[str, ...]) -> None:
        if not event_ids:
            return
        await self._query(
            """UPDATE "LiteLLM_ChargeReceipt"
               SET delivered_at=clock_timestamp(),lease_owner=NULL,lease_until=NULL,last_error=NULL
               WHERE event_id=ANY($1::text[]) AND lease_owner=$2 AND delivered_at IS NULL
               RETURNING event_id""",
            event_ids,
            owner,
        )

    async def retry(self, owner: str, event_ids: tuple[str, ...], reason: str) -> None:
        if not event_ids:
            return
        await self._query(
            """UPDATE "LiteLLM_ChargeReceipt"
               SET retry_at=clock_timestamp() + make_interval(secs => LEAST(300, power(2,LEAST(attempts,8)))::int),
                   lease_owner=NULL,lease_until=NULL,last_error=$3
               WHERE event_id=ANY($1::text[]) AND lease_owner=$2 AND delivered_at IS NULL
               RETURNING event_id""",
            event_ids,
            owner,
            reason,
        )
