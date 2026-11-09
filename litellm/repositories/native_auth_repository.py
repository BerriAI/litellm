import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from inspect import getattr_static
from itertools import chain
from typing import Final, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, TypeAdapter

from litellm._auth.failure import NativeAuthFailure, NativeAuthRuntimeError
from litellm.litellm_core_utils.core_helpers import require_attributes
from litellm.models.budget import LiteLLM_BudgetTable
from litellm.models.team import Member
from litellm.proxy.db.db_lookup_gate import DBLookupDeadlineExceeded, bounded_db_lookup, db_lookup_gate
from litellm.types.proxy.auth.user_api_key_auth import LiteLLM_VerificationTokenView

_Model = TypeVar("_Model", bound=BaseModel)  # rebind-ok: TypeVar declarations cannot be annotated Final
_ROWS: Final = TypeAdapter(list[dict[str, object]])
_SPEND: Final[TypeAdapter[float | None]] = TypeAdapter(float | None)
_BATCH_ID_KINDS: Final = frozenset(
    {
        "key",
        "user",
        "team",
        "org",
        "project",
        "end_user",
        "budget",
        "object_permission",
        "access_group",
        "tag",
        "model_access_group",
        "agent",
        "agent_name",
        "retired_agent",
        "managed_vector_store",
        "agent_access_group",
    }
)
_BATCH_SIZE: Final = 1000


class ModelAccessGroupRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    access_group_name: str
    spend: float | None = 0.0
    litellm_budget_table: LiteLLM_BudgetTable | None = None


class EndUserRegistryRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    user_id: str


class TagRegistryRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    tag_name: str


class AgentIdRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    agent_id: str


class SpendRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    spend: float | None = None


class BudgetWindowRow(SpendRow):
    window_start: datetime


class NativeAuthDatabaseError(NativeAuthFailure):
    pass


class NativeAuthDeadlineExceeded(NativeAuthDatabaseError, DBLookupDeadlineExceeded):
    pass


class AuthReadHandle(Protocol):
    async def auth_read(
        self,
        kind: str,
        ids: list[str],  # mutable-ok: PyO3 requires a Python list at this boundary
        writer: bool = False,
    ) -> object: ...

    async def auth_spend(self, kind: str, id: str) -> object: ...


class _NativeAuthClient(Protocol):
    @property
    def native_auth_database(self) -> AuthReadHandle | None: ...


@dataclass(frozen=True, slots=True)
class NativeAuthRepository:
    database: AuthReadHandle

    async def key(self, token: str, *, writer: bool = False) -> LiteLLM_VerificationTokenView | None:
        rows: Final = await self.read("key", [token], writer=writer)
        if not rows:
            return None
        try:
            row: Final = rows[0]
            members: Final = _ROWS.validate_python(row.get("team_members_with_roles") or [])
            matches: Final = tuple(
                Member.model_validate(member)
                for member in members
                if member.get("user_id") is not None and member.get("user_id") == row.get("user_id")
            )
            projection: Final = LiteLLM_VerificationTokenView.model_validate(
                {
                    **row,
                    "team_models": row.get("team_models") if row.get("team_models") is not None else [],
                    "team_blocked": row.get("team_blocked") if row.get("team_blocked") is not None else False,
                    "team_member": matches[-1] if matches else None,
                    "last_refreshed_at": time.time(),
                }
            )
            return (
                projection.model_copy(update={"expires": projection.expires.isoformat()})
                if isinstance(projection.expires, datetime)
                else projection
            )
        except Exception as error:  # noqa: BLE001  # preserve terminal native projection errors
            raise NativeAuthRuntimeError("Invalid native auth key projection") from error

    async def read(self, kind: str, ids: Sequence[str], writer: bool = False) -> Sequence[Mapping[str, object]]:
        try:
            value: Final = await bounded_db_lookup(self._read(kind, ids, writer), name=kind)
            return _ROWS.validate_python(value)
        except NativeAuthFailure:
            raise
        except DBLookupDeadlineExceeded as error:
            raise NativeAuthDeadlineExceeded(error.lookup, error.deadline_seconds) from error
        except OSError as error:
            raise NativeAuthDatabaseError(f"Native auth {kind} database read failed") from error
        except Exception as error:  # noqa: BLE001  # native errors must not enter legacy fallback paths
            raise NativeAuthRuntimeError(f"Native auth {kind} read failed") from error

    async def _read(self, kind: str, ids: Sequence[str], writer: bool) -> object:
        async with db_lookup_gate.current():
            return await self.database.auth_read(kind, list(ids), writer=writer)

    async def one(self, kind: str, ids: Sequence[str], model: type[_Model], *, writer: bool = False) -> _Model | None:
        rows: Final = await self.many(kind, ids, model, writer=writer)
        return rows[0] if rows else None

    async def many(
        self, kind: str, ids: Sequence[str], model: type[_Model], *, writer: bool = False
    ) -> Sequence[_Model]:
        rows: Final = await self._many_rows(kind, ids, writer=writer)
        try:
            return tuple(model.model_validate(row) for row in rows)
        except Exception as error:  # noqa: BLE001  # invalid native projections are terminal
            raise NativeAuthRuntimeError(f"Invalid native auth {kind} projection") from error

    async def _many_rows(self, kind: str, ids: Sequence[str], *, writer: bool) -> Sequence[Mapping[str, object]]:
        if kind not in _BATCH_ID_KINDS or len(ids) <= _BATCH_SIZE:
            return await self.read(kind, ids, writer=writer)
        unique_ids: Final = tuple(dict.fromkeys(ids))
        chunks: Final = [
            await self.read(kind, unique_ids[start : start + _BATCH_SIZE], writer=writer)
            for start in range(0, len(unique_ids), _BATCH_SIZE)
        ]
        return tuple(chain.from_iterable(chunks))

    async def spend(self, kind: str, entity_id: str) -> float | None:
        try:
            value: Final = await bounded_db_lookup(self._spend(kind, entity_id), name="spend_counter")
            return _SPEND.validate_python(value)
        except NativeAuthFailure:
            raise
        except DBLookupDeadlineExceeded as error:
            raise NativeAuthDeadlineExceeded(error.lookup, error.deadline_seconds) from error
        except OSError as error:
            raise NativeAuthDatabaseError(f"Native auth {kind} database spend read failed") from error
        except Exception as error:  # noqa: BLE001  # a failed native read is not an authoritative miss
            raise NativeAuthRuntimeError(f"Native auth {kind} spend read failed") from error

    async def _spend(self, kind: str, entity_id: str) -> object:
        async with db_lookup_gate.current():
            return await self.database.auth_spend(kind, entity_id)


def native_auth_repository(client: object) -> NativeAuthRepository | None:
    if getattr_static(client, "native_auth_database", None) is None:
        return None
    database: Final = require_attributes(client, _NativeAuthClient, "native_auth_database").native_auth_database
    return NativeAuthRepository(database) if database is not None else None
