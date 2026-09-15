from __future__ import annotations

import base64
import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Protocol

from botocore.exceptions import BotoCoreError, ClientError
from botocore.response import StreamingBody
from capture_policy import SCENARIO_BYTES, ScenarioIdentity
from capture_snapshot import SnapshotFailure, verify_snapshot
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError


class SnapshotPointer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    revision: str
    scenario_key: str = Field(pattern=r"^[a-f0-9]{64}$")
    object_key: str
    version_id: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size: int = Field(gt=0, le=SCENARIO_BYTES)


class ObjectVersion(BaseModel):
    VersionId: str = Field(min_length=1)


class ObjectDownload(ObjectVersion):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    Body: StreamingBody
    ContentLength: int


class PointerRead(BaseModel):
    Item: dict[str, dict[str, str]] = {}


class ObjectClient(Protocol):
    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        IfNoneMatch: str,
        ContentType: str,
        ChecksumSHA256: str,
        ServerSideEncryption: str,
        SSEKMSKeyId: str,
    ) -> object: ...

    def get_object(self, *, Bucket: str, Key: str, VersionId: str) -> object: ...


class PointerClient(Protocol):
    def get_item(self, *, TableName: str, Key: dict[str, JsonValue], ConsistentRead: bool) -> object: ...

    def put_item(
        self,
        *,
        TableName: str,
        Item: dict[str, JsonValue],
        ConditionExpression: str,
        ExpressionAttributeValues: dict[str, JsonValue] = ...,
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class SnapshotRepository:
    objects: ObjectClient
    pointers: PointerClient
    bucket: str
    table: str
    kms_key: str

    def read_pointer(self, identity: ScenarioIdentity) -> SnapshotPointer | SnapshotFailure | None:
        try:
            response: Final = PointerRead.model_validate(
                self.pointers.get_item(
                    TableName=self.table,
                    Key={"pk": {"S": f"pointer#{identity.key}"}},
                    ConsistentRead=True,
                )
            )
            if not response.Item:
                return None
            encoded: Final = response.Item.get("pointer_json", {}).get("S")
            if encoded is None:
                return SnapshotFailure("approved pointer is malformed")
            pointer: Final = SnapshotPointer.model_validate_json(encoded)
            return (
                pointer
                if pointer.scenario_key == identity.key
                else SnapshotFailure("approved pointer identity mismatch")
            )
        except (ClientError, BotoCoreError, ValidationError):
            return SnapshotFailure("approved pointer unavailable")

    def download(
        self, pointer: SnapshotPointer, identity: ScenarioIdentity, *, now: datetime
    ) -> bytes | SnapshotFailure:
        if pointer.scenario_key != identity.key or pointer.version_id == "null":
            return SnapshotFailure("approved pointer identity or version mismatch")
        if pointer.object_key != f"approved/{identity.key}/{pointer.sha256}.json":
            return SnapshotFailure("approved object key mismatch")
        try:
            response: Final = ObjectDownload.model_validate(
                self.objects.get_object(
                    Bucket=self.bucket,
                    Key=pointer.object_key,
                    VersionId=pointer.version_id,
                )
            )
            try:
                if response.VersionId != pointer.version_id or response.ContentLength != pointer.size:
                    return SnapshotFailure("approved object version or size mismatch")
                content: Final = response.Body.read(SCENARIO_BYTES + 1)
            finally:
                response.Body.close()
            if len(content) != pointer.size:
                return SnapshotFailure("approved object size mismatch")
            verified: Final = verify_snapshot(content, expected_sha256=pointer.sha256, identity=identity, now=now)
            return verified if isinstance(verified, SnapshotFailure) else content
        except (ClientError, BotoCoreError, ValidationError, OSError):
            return SnapshotFailure("approved snapshot unavailable")

    def publish(
        self,
        content: bytes,
        identity: ScenarioIdentity,
        *,
        expected_revision: str | None,
        now: datetime,
    ) -> SnapshotPointer | SnapshotFailure:
        digest: Final = hashlib.sha256(content).hexdigest()
        verified: Final = verify_snapshot(content, expected_sha256=digest, identity=identity, now=now)
        if isinstance(verified, SnapshotFailure):
            return verified
        key: Final = f"approved/{identity.key}/{digest}.json"
        try:
            version: Final = ObjectVersion.model_validate(
                self.objects.put_object(
                    Bucket=self.bucket,
                    Key=key,
                    Body=content,
                    IfNoneMatch="*",
                    ContentType="application/json",
                    ChecksumSHA256=base64.b64encode(hashlib.sha256(content).digest()).decode(),
                    ServerSideEncryption="aws:kms",
                    SSEKMSKeyId=self.kms_key,
                )
            )
        except (ClientError, BotoCoreError, ValidationError):
            return SnapshotFailure("create-only snapshot publication failed or is uncertain")
        pointer: Final = SnapshotPointer(
            revision=uuid.uuid4().hex,
            scenario_key=identity.key,
            object_key=key,
            version_id=version.VersionId,
            sha256=digest,
            size=len(content),
        )
        readback: Final = self.download(pointer, identity, now=now)
        if isinstance(readback, SnapshotFailure):
            return readback
        item: Final[dict[str, JsonValue]] = {
            "pk": {"S": f"pointer#{identity.key}"},
            "pointer_revision": {"S": pointer.revision},
            "pointer_json": {"S": pointer.model_dump_json()},
        }
        try:
            if expected_revision is None:
                self.pointers.put_item(TableName=self.table, Item=item, ConditionExpression="attribute_not_exists(pk)")
            else:
                self.pointers.put_item(
                    TableName=self.table,
                    Item=item,
                    ConditionExpression="pointer_revision = :expected",
                    ExpressionAttributeValues={":expected": {"S": expected_revision}},
                )
        except (ClientError, BotoCoreError):
            return SnapshotFailure("pointer promotion failed or is uncertain; object remains unreferenced")
        return pointer
