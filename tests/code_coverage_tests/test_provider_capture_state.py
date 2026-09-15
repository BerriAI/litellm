from __future__ import annotations

import hashlib
from datetime import timedelta
from io import BytesIO
from typing import Final

from botocore.response import StreamingBody
from botocore.stub import ANY, Stubber
from capture_policy import ScenarioIdentity, ScenarioOutcome
from capture_publication import SnapshotPointer, SnapshotRepository
from capture_session import CaptureSession
from capture_snapshot import ScenarioSnapshot, SnapshotFailure, refresh_due, verify_snapshot
from fixture_bundle import RecordedHttpResponse
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule
from test_provider_capture import ScriptedReservations, successful_response
from test_provider_capture_publication import IDENTITY, NOW, clients, snapshot_content


class CaptureLifecycleMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.store = ScriptedReservations()
        self.session = CaptureSession(
            ScenarioIdentity("test_synthetic.py::test_capture", "a" * 64, "synthetic"),
            "owner",
            self.store,
            max_attempts=3,
        )
        self.reservations = 0
        self.responses = 0
        self.in_flight = False
        self.failed = False
        self.closed = False

    @rule(uncertain=st.booleans())
    def reserve(self, uncertain: bool) -> None:
        self.store.uncertain = uncertain
        expected_denial: Final = self.closed or self.failed or self.in_flight or self.reservations == 3 or uncertain
        response: Final = self.session.before_attempt()
        assert (response is not None) == expected_denial
        if not expected_denial:
            self.reservations += 1
            self.in_flight = True
        elif not self.closed:
            self.failed = True

    @precondition(lambda self: self.in_flight and not self.closed)
    @rule(success=st.booleans())
    def receive(self, success: bool) -> None:
        self.session.response_finished(
            successful_response() if success else RecordedHttpResponse(status_code=503, headers={}, body_b64="")
        )
        self.responses += 1
        self.in_flight = False
        if not success:
            self.failed = True

    @precondition(lambda self: not self.closed)
    @rule(setup=st.booleans(), call=st.booleans(), teardown=st.booleans())
    def finish(self, setup: bool, call: bool, teardown: bool) -> None:
        expected: Final = setup and call and teardown and not self.failed and not self.in_flight and self.responses > 0
        result: Final = self.session.finish(ScenarioOutcome(setup, call, teardown))
        assert result.publishable == expected
        assert len(result.attempts) == self.reservations
        assert result.response_count == self.responses
        self.closed = True

    @invariant()
    def budget_is_never_exceeded(self) -> None:
        assert self.reservations <= 3
        assert len(self.store.calls) <= 4


TestCaptureLifecycle = CaptureLifecycleMachine.TestCase
TestCaptureLifecycle.settings = settings(max_examples=30, stateful_step_count=25, derandomize=True, deadline=None)


class PublicationLifecycleMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        objects, pointers = clients()
        self.s3 = Stubber(objects)
        self.ddb = Stubber(pointers)
        self.s3.activate()
        self.ddb.activate()
        self.repository = SnapshotRepository(objects, pointers, "synthetic-bucket", "synthetic-table", "synthetic-key")
        self.approved: SnapshotPointer | None = None
        self.content: bytes | None = None
        self.sequence = 0
        self.age = 0

    @rule(interrupted=st.booleans(), stale_writer=st.booleans())
    def publish(self, interrupted: bool, stale_writer: bool) -> None:
        self.sequence += 1
        candidate: Final = (
            ScenarioSnapshot.model_validate_json(snapshot_content())
            .model_copy(update={"owner": f"owner-{self.sequence}"})
            .model_dump_json()
            .encode()
        )
        digest: Final = hashlib.sha256(candidate).hexdigest()
        key: Final = f"approved/{IDENTITY.key}/{digest}.json"
        version: Final = f"version-{self.sequence}"
        self.s3.add_response("put_object", {"VersionId": version})
        self.s3.add_response(
            "get_object",
            {
                "VersionId": version,
                "ContentLength": len(candidate),
                "Body": StreamingBody(BytesIO(candidate), len(candidate)),
            },
            {"Bucket": "synthetic-bucket", "Key": key, "VersionId": version},
        )
        expected: Final = "stale" if stale_writer else (self.approved.revision if self.approved else None)
        params: Final = {
            "TableName": "synthetic-table",
            "Item": ANY,
            "ConditionExpression": "attribute_not_exists(pk)" if expected is None else "pointer_revision = :expected",
        }
        if expected is not None:
            params["ExpressionAttributeValues"] = {":expected": {"S": expected}}
        if interrupted or stale_writer:
            self.ddb.add_client_error(
                "put_item",
                service_error_code="InternalServerError" if interrupted else "ConditionalCheckFailedException",
                expected_params=params,
            )
        else:
            self.ddb.add_response("put_item", {}, params)
        result: Final = self.repository.publish(candidate, IDENTITY, expected_revision=expected, now=NOW)
        assert isinstance(result, SnapshotFailure) == (interrupted or stale_writer)
        if isinstance(result, SnapshotPointer):
            self.approved = result
            self.content = candidate
            self.age = 0
        self.s3.assert_no_pending_responses()
        self.ddb.assert_no_pending_responses()

    @rule(hours=st.sampled_from([0, 23, 24, 167, 168, 169]))
    def expire(self, hours: int) -> None:
        self.age = hours

    @precondition(lambda self: self.approved is not None)
    @rule(outage=st.booleans(), corrupt=st.booleans())
    def download(self, outage: bool, corrupt: bool) -> None:
        assert self.approved is not None and self.content is not None
        pointer: Final = self.approved
        if outage:
            self.s3.add_client_error("get_object", service_error_code="ServiceUnavailable")
        else:
            data: Final = self.content.replace(b"blue", b"gold") if corrupt else self.content
            self.s3.add_response(
                "get_object",
                {
                    "VersionId": pointer.version_id,
                    "ContentLength": len(data),
                    "Body": StreamingBody(BytesIO(data), len(data)),
                },
            )
        result: Final = self.repository.download(pointer, IDENTITY, now=NOW + timedelta(hours=self.age))
        assert isinstance(result, bytes) == (not outage and not corrupt and self.age < 168)
        local: Final = verify_snapshot(
            self.content, expected_sha256=pointer.sha256, identity=IDENTITY, now=NOW + timedelta(hours=self.age)
        )
        assert isinstance(local, ScenarioSnapshot) == (self.age < 168)
        if isinstance(local, ScenarioSnapshot):
            assert refresh_due(local, now=NOW + timedelta(hours=self.age)) == (self.age >= 24)
        self.s3.assert_no_pending_responses()

    def teardown(self) -> None:
        self.s3.assert_no_pending_responses()
        self.ddb.assert_no_pending_responses()
        self.s3.deactivate()
        self.ddb.deactivate()


TestPublicationLifecycle = PublicationLifecycleMachine.TestCase
TestPublicationLifecycle.settings = settings(max_examples=20, stateful_step_count=15, derandomize=True, deadline=None)
