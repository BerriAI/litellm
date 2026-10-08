"""Real RDS IAM cross-region proof: writer and replica each signed in its own region."""

import os
from collections.abc import Iterator
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import Final

import pytest
from e2e_config import unique_marker
from e2e_http import unwrap
from models import ChatBody, ChatMessage, ChatResponse, KeyGenerateBody

from rds_gateway import (
    NOVA_MICRO_MODEL,
    RdsGateway,
    hostname_region,
    owned_rds_gateway,
    replica_connections,
    replica_now,
)

pytestmark = [pytest.mark.rds_iam, pytest.mark.timeout(600)]


def _chat_once(gateway: RdsGateway, key: str) -> ChatResponse:
    return unwrap(
        gateway.proxy.chat(
            key,
            ChatBody(
                model=NOVA_MICRO_MODEL,
                messages=[ChatMessage(role="user", content=f"say ok: {unique_marker()}")],
                max_tokens=16,
            ),
        )
    )


class TestRdsIamCrossRegionReplica:
    @pytest.fixture
    def replica_since(self) -> datetime:
        return replica_now(
            os.environ["E2E_RDS_READER_HOST"],
            hostname_region(os.environ["E2E_RDS_READER_HOST"]),
            os.environ["E2E_RDS_USER"],
            os.environ["E2E_RDS_DATABASE"],
        )

    @pytest.fixture
    def gateway(self, tmp_path: Path, replica_since: datetime) -> Iterator[RdsGateway]:
        with ExitStack() as cleanup:
            yield owned_rds_gateway(tmp_path, cleanup, {})

    @pytest.fixture
    def wrong_reader_gateway(self, tmp_path: Path) -> Iterator[RdsGateway]:
        with ExitStack() as cleanup:
            yield owned_rds_gateway(
                tmp_path,
                cleanup,
                {"AWS_RDS_READ_REPLICA_REGION": hostname_region(os.environ["E2E_RDS_WRITER_HOST"])},
            )

    def test_cross_region_writer_and_replica_serve_with_no_overrides(
        self, gateway: RdsGateway, replica_since: datetime
    ) -> None:
        key: Final = gateway.proxy.generate_key(KeyGenerateBody())
        try:
            response: Final = _chat_once(gateway, key)
            assert response.id, "chat completion carried no request id"
            message: Final = response.choices[0].message if response.choices else None
            content: Final = message.content if message else None
            assert content, "chat completion returned empty content"
            rows: Final = gateway.proxy.poll_logs_for_request_id(response.id)
            assert rows, f"no spend row for request {response.id}"
            connections: Final = replica_connections(
                os.environ["E2E_RDS_READER_HOST"],
                gateway.reader_region,
                os.environ["E2E_RDS_USER"],
                os.environ["E2E_RDS_DATABASE"],
                replica_since,
            )
            assert connections, (
                f"no post-boot connections for {os.environ['E2E_RDS_USER']} on the replica; "
                "reads were not served by it"
            )
        finally:
            gateway.proxy.delete_key(key)

    def test_wrong_reader_override_falls_back_to_writer(self, wrong_reader_gateway: RdsGateway) -> None:
        log: Final = wrong_reader_gateway.log_text()
        assert "Failed to connect to read replica DB" in log, (
            "reader signed in the wrong region did not fail at boot"
        )
        assert "Falling back to the writer" in log, "writer fallback was not logged"
        key: Final = wrong_reader_gateway.proxy.generate_key(KeyGenerateBody())
        try:
            response: Final = _chat_once(wrong_reader_gateway, key)
            assert response.id, "chat completion carried no request id"
            rows: Final = wrong_reader_gateway.proxy.poll_logs_for_request_id(response.id)
            assert rows, f"no spend row for request {response.id}"
        finally:
            wrong_reader_gateway.proxy.delete_key(key)
