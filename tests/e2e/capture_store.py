from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Protocol

from botocore.exceptions import BotoCoreError, ClientError
from capture_session import AttemptDenied, AttemptReserved, AttemptUncertain, Reservation
from pydantic import JsonValue


@dataclass(frozen=True, slots=True)
class StoreFailure:
    reason: str
    uncertain: bool = False


class CaptureLeaseStore(Protocol):
    def acquire(self, *, scenario_key: str, owner: str, expires_at: int) -> StoreFailure | None: ...

    def reserve(self, *, scenario_key: str, owner: str, attempt_id: str) -> Reservation: ...

    def complete(self, *, scenario_key: str, owner: str, attempt_id: str, successful: bool) -> str | None: ...

    def release(self, *, scenario_key: str, owner: str) -> StoreFailure | None: ...


class DynamoClient(Protocol):
    def put_item(
        self,
        *,
        TableName: str,
        Item: dict[str, JsonValue],
        ConditionExpression: str,
        ExpressionAttributeValues: dict[str, JsonValue] = ...,
    ) -> object: ...

    def update_item(
        self,
        *,
        TableName: str,
        Key: dict[str, JsonValue],
        UpdateExpression: str,
        ConditionExpression: str,
        ExpressionAttributeValues: dict[str, JsonValue],
    ) -> object: ...

    def transact_write_items(self, *, TransactItems: list[dict[str, JsonValue]], ClientRequestToken: str) -> object: ...


def _store_failure(error: ClientError | BotoCoreError) -> StoreFailure:
    if isinstance(error, ClientError):
        code: Final = str(error.response.get("Error", {}).get("Code", ""))
        if code in ("ConditionalCheckFailedException", "TransactionCanceledException"):
            return StoreFailure("store condition rejected")
    return StoreFailure("store operation outcome uncertain", uncertain=True)


@dataclass(frozen=True, slots=True)
class DynamoCaptureStore:
    client: DynamoClient
    table: str
    clock: Callable[[], float] = time.time

    def create_run(self, *, owner: str, cap: int, expires_at: int) -> StoreFailure | None:
        if not owner or not 1 <= cap <= 12 or expires_at <= self.clock():
            return StoreFailure("invalid capture run bounds")
        try:
            self.client.put_item(
                TableName=self.table,
                Item={
                    "pk": {"S": f"run#{owner}"},
                    "attempt_count": {"N": "0"},
                    "attempt_cap": {"N": str(cap)},
                    "lease_expires": {"N": str(expires_at)},
                },
                ConditionExpression="attribute_not_exists(pk)",
            )
        except (ClientError, BotoCoreError) as error:
            return _store_failure(error)
        return None

    def acquire(self, *, scenario_key: str, owner: str, expires_at: int) -> StoreFailure | None:
        now: Final = int(self.clock())
        if not owner or not scenario_key or expires_at <= now:
            return StoreFailure("invalid lease bounds")
        try:
            self.client.put_item(
                TableName=self.table,
                Item={
                    "pk": {"S": f"lease#{scenario_key}"},
                    "lease_owner": {"S": owner},
                    "lease_expires": {"N": str(expires_at)},
                    "lease_settled": {"BOOL": False},
                },
                ConditionExpression="attribute_not_exists(pk) OR (lease_settled = :true AND lease_expires <= :now)",
                ExpressionAttributeValues={":true": {"BOOL": True}, ":now": {"N": str(now)}},
            )
        except (ClientError, BotoCoreError) as error:
            return _store_failure(error)
        return None

    def reserve(self, *, scenario_key: str, owner: str, attempt_id: str) -> Reservation:
        if not owner or not scenario_key or not 1 <= len(attempt_id) <= 36:
            return AttemptDenied("invalid attempt identity")
        now: Final = str(int(self.clock()))
        transaction: Final[list[dict[str, JsonValue]]] = [
            {
                "Update": {
                    "TableName": self.table,
                    "Key": {"pk": {"S": f"run#{owner}"}},
                    "UpdateExpression": "SET attempt_count = attempt_count + :one",
                    "ConditionExpression": "attempt_count < attempt_cap AND lease_expires > :now",
                    "ExpressionAttributeValues": {":one": {"N": "1"}, ":now": {"N": now}},
                }
            },
            {
                "Update": {
                    "TableName": self.table,
                    "Key": {"pk": {"S": f"lease#{scenario_key}"}},
                    "UpdateExpression": "SET active_attempt = :attempt",
                    "ConditionExpression": "lease_owner = :owner AND lease_expires > :now AND lease_settled = :false AND attribute_not_exists(active_attempt)",
                    "ExpressionAttributeValues": {
                        ":owner": {"S": owner},
                        ":now": {"N": now},
                        ":false": {"BOOL": False},
                        ":attempt": {"S": attempt_id},
                    },
                }
            },
            {
                "Put": {
                    "TableName": self.table,
                    "Item": {
                        "pk": {"S": f"attempt#{owner}#{attempt_id}"},
                        "recorded_scenario": {"S": scenario_key},
                        "attempt_completed": {"BOOL": False},
                    },
                    "ConditionExpression": "attribute_not_exists(pk)",
                }
            },
        ]
        try:
            self.client.transact_write_items(TransactItems=transaction, ClientRequestToken=attempt_id)
        except (ClientError, BotoCoreError) as error:
            failure: Final = _store_failure(error)
            return AttemptUncertain(failure.reason) if failure.uncertain else AttemptDenied(failure.reason)
        return AttemptReserved(attempt_id)

    def complete(self, *, scenario_key: str, owner: str, attempt_id: str, successful: bool) -> str | None:
        transaction: Final[list[dict[str, JsonValue]]] = [
            {
                "Update": {
                    "TableName": self.table,
                    "Key": {"pk": {"S": f"lease#{scenario_key}"}},
                    "UpdateExpression": "REMOVE active_attempt",
                    "ConditionExpression": "lease_owner = :owner AND active_attempt = :attempt",
                    "ExpressionAttributeValues": {":owner": {"S": owner}, ":attempt": {"S": attempt_id}},
                }
            },
            {
                "Update": {
                    "TableName": self.table,
                    "Key": {"pk": {"S": f"attempt#{owner}#{attempt_id}"}},
                    "UpdateExpression": "SET attempt_completed = :true, attempt_successful = :attempt_successful",
                    "ConditionExpression": "recorded_scenario = :recorded_scenario AND attempt_completed = :false",
                    "ExpressionAttributeValues": {
                        ":recorded_scenario": {"S": scenario_key},
                        ":false": {"BOOL": False},
                        ":true": {"BOOL": True},
                        ":attempt_successful": {"BOOL": successful},
                    },
                }
            },
        ]
        try:
            self.client.transact_write_items(
                TransactItems=transaction,
                ClientRequestToken=hashlib.sha256(f"done:{owner}:{attempt_id}".encode()).hexdigest()[:36],
            )
        except (ClientError, BotoCoreError) as error:
            return _store_failure(error).reason
        return None

    def release(self, *, scenario_key: str, owner: str) -> StoreFailure | None:
        try:
            self.client.update_item(
                TableName=self.table,
                Key={"pk": {"S": f"lease#{scenario_key}"}},
                UpdateExpression="SET lease_settled = :true, lease_expires = :now",
                ConditionExpression="lease_owner = :owner AND attribute_not_exists(active_attempt)",
                ExpressionAttributeValues={
                    ":owner": {"S": owner},
                    ":true": {"BOOL": True},
                    ":now": {"N": str(int(self.clock()))},
                },
            )
        except (ClientError, BotoCoreError) as error:
            return _store_failure(error)
        return None
