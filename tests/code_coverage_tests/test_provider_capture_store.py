from __future__ import annotations

from typing import Final, cast

import boto3
import pytest
from botocore.config import Config
from botocore.stub import Stubber
from capture_session import AttemptDenied, AttemptReserved, AttemptUncertain
from capture_store import DynamoCaptureStore, DynamoClient


def dynamo_client() -> DynamoClient:
    return cast(
        DynamoClient,
        boto3.client(
            "dynamodb",
            region_name="us-east-1",
            aws_access_key_id="synthetic",
            aws_secret_access_key="synthetic",
            config=Config(retries={"max_attempts": 0}),
        ),
    )


class TestDurableReservations:
    def test_run_cannot_reset_an_existing_budget(self) -> None:
        client: Final = dynamo_client()
        with Stubber(client) as stub:
            stub.add_response(
                "put_item",
                {},
                {
                    "TableName": "synthetic-table",
                    "Item": {
                        "pk": {"S": "run#owner"},
                        "attempt_count": {"N": "0"},
                        "attempt_cap": {"N": "2"},
                        "lease_expires": {"N": "200"},
                    },
                    "ConditionExpression": "attribute_not_exists(pk)",
                },
            )
            store: Final = DynamoCaptureStore(client, "synthetic-table", clock=lambda: 100)
            assert store.create_run(owner="owner", cap=2, expires_at=200) is None
            stub.add_client_error("put_item", service_error_code="ConditionalCheckFailedException")
            failure: Final = store.create_run(owner="owner", cap=2, expires_at=200)
            assert failure is not None and not failure.uncertain
            stub.assert_no_pending_responses()

    def test_expired_but_unsettled_lease_cannot_be_stolen(self) -> None:
        client: Final = dynamo_client()
        with Stubber(client) as stub:
            expected: Final = {
                "TableName": "synthetic-table",
                "Item": {
                    "pk": {"S": "lease#scenario"},
                    "lease_owner": {"S": "owner"},
                    "lease_expires": {"N": "200"},
                    "lease_settled": {"BOOL": False},
                },
                "ConditionExpression": "attribute_not_exists(pk) OR (lease_settled = :true AND lease_expires <= :now)",
                "ExpressionAttributeValues": {":true": {"BOOL": True}, ":now": {"N": "100"}},
            }
            stub.add_client_error(
                "put_item", service_error_code="ConditionalCheckFailedException", expected_params=expected
            )
            failure: Final = DynamoCaptureStore(client, "synthetic-table", clock=lambda: 100).acquire(
                scenario_key="scenario", owner="owner", expires_at=200
            )
            assert failure is not None and not failure.uncertain
            stub.assert_no_pending_responses()

    @pytest.mark.parametrize(
        "code,kind", [("TransactionCanceledException", AttemptDenied), ("InternalServerError", AttemptUncertain)]
    )
    def test_reservation_denial_and_uncertainty_remain_distinct(
        self, code: str, kind: type[AttemptDenied | AttemptUncertain]
    ) -> None:
        client: Final = dynamo_client()
        with Stubber(client) as stub:
            stub.add_client_error("transact_write_items", service_error_code=code)
            result: Final = DynamoCaptureStore(client, "synthetic-table", clock=lambda: 100).reserve(
                scenario_key="scenario", owner="owner", attempt_id="attempt-one"
            )
            assert isinstance(result, kind)
            stub.assert_no_pending_responses()

    def test_successful_atomic_reservation_keeps_attempt_identity(self) -> None:
        client: Final = dynamo_client()
        with Stubber(client) as stub:
            stub.add_response("transact_write_items", {})
            result: Final = DynamoCaptureStore(client, "synthetic-table", clock=lambda: 100).reserve(
                scenario_key="scenario", owner="owner", attempt_id="attempt-one"
            )
            assert result == AttemptReserved("attempt-one")
            stub.assert_no_pending_responses()
