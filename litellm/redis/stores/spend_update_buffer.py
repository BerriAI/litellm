import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, Protocol

from pydantic import TypeAdapter

from litellm.constants import REDIS_UPDATE_BUFFER_KEY
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy._types import DBSpendUpdateTransactions
from litellm.redis.transport.contract import BatchReply, Popped, RedisCommandBatch, RedisFailure, RedisTransport

_OBJECT: Final = TypeAdapter(dict[str, object])
_ENTITIES: Final = TypeAdapter(dict[str, int | float])


def _entities(value: object) -> Mapping[str, float]:
    return dict(_ENTITIES.validate_python(value, strict=True)) if isinstance(value, dict) else {}


def decode_transaction(raw: str | bytes, non_finite: Callable[[str], float]) -> DBSpendUpdateTransactions:
    payload: Final = _OBJECT.validate_python(json.loads(raw, parse_constant=non_finite))
    return DBSpendUpdateTransactions(
        user_list_transactions=dict(_entities(payload.get("user_list_transactions"))),
        end_user_list_transactions=dict(_entities(payload.get("end_user_list_transactions"))),
        key_list_transactions=dict(_entities(payload.get("key_list_transactions"))),
        team_list_transactions=dict(_entities(payload.get("team_list_transactions"))),
        team_member_list_transactions=dict(_entities(payload.get("team_member_list_transactions"))),
        org_list_transactions=dict(_entities(payload.get("org_list_transactions"))),
        org_member_list_transactions=dict(_entities(payload.get("org_member_list_transactions"))),
        project_list_transactions=dict(_entities(payload.get("project_list_transactions"))),
        tag_list_transactions=dict(_entities(payload.get("tag_list_transactions"))),
        agent_list_transactions=dict(_entities(payload.get("agent_list_transactions"))),
        model_access_group_list_transactions=dict(_entities(payload.get("model_access_group_list_transactions"))),
    )


def _decode(
    reply: Popped | RedisFailure, non_finite: Callable[[str], float]
) -> tuple[DBSpendUpdateTransactions, ...] | None | RedisFailure:
    if reply is None or isinstance(reply, RedisFailure):
        return reply
    entries: Final = (reply,) if isinstance(reply, bytes) else reply
    try:
        return tuple(decode_transaction(entry, non_finite) for entry in entries)
    except ValueError as error:
        return RedisFailure(kind="decode", message=f"invalid spend update buffer: {error}")


@dataclass(frozen=True, slots=True)
class PendingSpendUpdates:
    reply: BatchReply[Popped | RedisFailure]
    non_finite: Callable[[str], float]

    async def result(self) -> tuple[DBSpendUpdateTransactions, ...] | None | RedisFailure:
        return _decode(await self.reply, self.non_finite)


class SpendUpdateBufferStore(Protocol):
    def append(
        self, batch: RedisCommandBatch, transactions: DBSpendUpdateTransactions
    ) -> BatchReply[int | RedisFailure]: ...

    def take(self, batch: RedisCommandBatch, limit: int) -> PendingSpendUpdates: ...

    async def drain(self, limit: int) -> tuple[DBSpendUpdateTransactions, ...] | None | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportSpendUpdateBufferStore:
    transport: RedisTransport
    non_finite: Callable[[str], float]

    def append(
        self, batch: RedisCommandBatch, transactions: DBSpendUpdateTransactions
    ) -> BatchReply[int | RedisFailure]:
        return batch.rpush(REDIS_UPDATE_BUFFER_KEY, safe_dumps(transactions))

    def take(self, batch: RedisCommandBatch, limit: int) -> PendingSpendUpdates:
        return PendingSpendUpdates(batch.lpop(REDIS_UPDATE_BUFFER_KEY, limit), self.non_finite)

    async def drain(self, limit: int) -> tuple[DBSpendUpdateTransactions, ...] | None | RedisFailure:
        return _decode(await self.transport.lpop(REDIS_UPDATE_BUFFER_KEY, limit), self.non_finite)
