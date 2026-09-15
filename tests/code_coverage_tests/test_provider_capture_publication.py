from __future__ import annotations

import base64
import hashlib
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Final, cast

import pytest
from botocore.config import Config
from botocore.response import StreamingBody
from botocore.session import get_session
from botocore.stub import ANY, Stubber
from capture_policy import ScenarioIdentity, ScenarioOutcome
from capture_publication import ObjectClient, PointerClient, SnapshotPointer, SnapshotRepository
from capture_snapshot import CaptureProvenance, ScenarioSnapshot, SnapshotFailure, materialize_snapshot
from fixture_bundle import Interaction, Manifest, RecordedRequest
from fixture_profile import StrictIdentity, strict_identity
from test_provider_capture import successful_response

NOW: Final = datetime(2031, 4, 5, tzinfo=timezone.utc)
IDENTITY: Final = ScenarioIdentity("test_example.py::test_one", "a" * 64, "synthetic")


def snapshot_content() -> bytes:
    request: Final = strict_identity(
        method="POST",
        path="/openai/v1/chat/completions",
        query="",
        headers={"content-type": "application/json"},
        body=b'{"prompt":"blue"}',
        mount="openai",
        upstream_base="https://api.openai.com",
    )
    assert isinstance(request, StrictIdentity)
    return (
        ScenarioSnapshot(
            identity=IDENTITY,
            manifest=Manifest(
                format_version=5, recorded_at=NOW, harness_version="synthetic", match_profile="stateless_v1"
            ),
            interactions=(
                Interaction(
                    request=RecordedRequest(
                        method="post", path="/openai/v1/chat/completions", headers={}, strict_identity=request
                    ),
                    response=successful_response(),
                ),
            ),
            provenance=CaptureProvenance(
                test_revision="b" * 40, candidate_revision="c" * 40, runner_digest="sha256:" + "d" * 64
            ),
            owner="owner",
            attempts=("attempt",),
            outcome=ScenarioOutcome(True, True, True),
        )
        .model_dump_json()
        .encode()
    )


def clients() -> tuple[ObjectClient, PointerClient]:
    session: Final = get_session()
    config: Final = Config(retries={"total_max_attempts": 1})
    return (
        cast(
            ObjectClient,
            session.create_client(
                "s3",
                region_name="us-east-1",
                aws_access_key_id="synthetic",
                aws_secret_access_key="synthetic",
                config=config,
            ),
        ),
        cast(
            PointerClient,
            session.create_client(
                "dynamodb",
                region_name="us-east-1",
                aws_access_key_id="synthetic",
                aws_secret_access_key="synthetic",
                config=config,
            ),
        ),
    )


class TestConditionalPublication:
    @pytest.mark.parametrize("conflict", [False, True])
    def test_object_readback_precedes_conditional_pointer_promotion(self, conflict: bool) -> None:
        objects, pointers = clients()
        content: Final = snapshot_content()
        sha: Final = hashlib.sha256(content).hexdigest()
        key: Final = f"approved/{IDENTITY.key}/{sha}.json"
        with Stubber(objects) as s3, Stubber(pointers) as dynamo:
            s3.add_response(
                "put_object",
                {"VersionId": "version-one"},
                {
                    "Bucket": "synthetic-bucket",
                    "Key": key,
                    "Body": content,
                    "IfNoneMatch": "*",
                    "ContentType": "application/json",
                    "ChecksumSHA256": base64.b64encode(hashlib.sha256(content).digest()).decode(),
                    "ServerSideEncryption": "aws:kms",
                    "SSEKMSKeyId": "synthetic-key",
                },
            )
            s3.add_response(
                "get_object",
                {
                    "VersionId": "version-one",
                    "ContentLength": len(content),
                    "Body": StreamingBody(BytesIO(content), len(content)),
                },
                {"Bucket": "synthetic-bucket", "Key": key, "VersionId": "version-one"},
            )
            expected: Final = {
                "TableName": "synthetic-table",
                "Item": ANY,
                "ConditionExpression": "pointer_revision = :expected",
                "ExpressionAttributeValues": {":expected": {"S": "previous"}},
            }
            if conflict:
                dynamo.add_client_error(
                    "put_item", service_error_code="ConditionalCheckFailedException", expected_params=expected
                )
            else:
                dynamo.add_response("put_item", {}, expected)
            result: Final = SnapshotRepository(
                objects, pointers, "synthetic-bucket", "synthetic-table", "synthetic-key"
            ).publish(content, IDENTITY, expected_revision="previous", now=NOW)
            if conflict:
                assert isinstance(result, SnapshotFailure) and "unreferenced" in result.reason
            else:
                assert (
                    isinstance(result, SnapshotPointer) and result.version_id == "version-one" and result.sha256 == sha
                )
            s3.assert_no_pending_responses()
            dynamo.assert_no_pending_responses()

    @pytest.mark.parametrize("fault", ["upload", "version", "digest"])
    def test_interrupted_or_corrupt_blob_never_reaches_pointer_write(self, fault: str) -> None:
        objects, pointers = clients()
        content: Final = snapshot_content()
        with Stubber(objects) as s3, Stubber(pointers) as dynamo:
            if fault == "upload":
                s3.add_client_error("put_object", service_error_code="InternalError")
            else:
                s3.add_response("put_object", {"VersionId": "version-one"})
                downloaded: Final = content.replace(b"blue", b"gold") if fault == "digest" else content
                s3.add_response(
                    "get_object",
                    {
                        "VersionId": "wrong" if fault == "version" else "version-one",
                        "ContentLength": len(downloaded),
                        "Body": StreamingBody(BytesIO(downloaded), len(downloaded)),
                    },
                )
            result: Final = SnapshotRepository(
                objects, pointers, "synthetic-bucket", "synthetic-table", "synthetic-key"
            ).publish(content, IDENTITY, expected_revision=None, now=NOW)
            assert isinstance(result, SnapshotFailure)
            s3.assert_no_pending_responses()
            dynamo.assert_no_pending_responses()


def test_materialization_write_failure_removes_partial_bundle_and_can_retry(tmp_path: Path) -> None:
    snapshot = ScenarioSnapshot.model_validate_json(snapshot_content())
    first = snapshot.interactions[0]
    # Force a real filesystem failure after the manifest and first interaction were written.
    bad_request = first.request.model_copy(update={"method": "x" * 300})
    broken = snapshot.model_copy(update={"interactions": (first, first.model_copy(update={"request": bad_request}))})
    destination = tmp_path / "snapshot"
    with pytest.raises(OSError, match="File name too long"):
        materialize_snapshot(broken, destination)
    assert not destination.exists()
    materialize_snapshot(snapshot, destination)
    assert len(list(destination.rglob("*.json"))) == 2
    with pytest.raises(FileExistsError):
        materialize_snapshot(snapshot, destination)
    assert len(list(destination.rglob("*.json"))) == 2
