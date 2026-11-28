import json
import os
import socket
import ssl
import struct
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Final
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo
from pydantic import SecretStr

from litellm.integrations.postgres_olap import PostgresOlapConfig, PostgresOlapLogger, write_batch
from tests.integration._support.ports import allocate_port
from tests.integration._support.process import self_signed_certificate


def test_pg_protocol_batch_roundtrip_with_quoted_identifiers_and_content() -> None:
    dsn: Final = make_conninfo(os.environ["DATABASE_URL"], sslmode="disable")
    table: Final = "logs_" + uuid4().hex + '"quoted'
    now: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
    payload: Final = {
        "id": "request-'中文",
        "model": "synthetic",
        "total_tokens": 5,
        "response_cost": 0.125,
        "messages": [{"role": "user", "content": "'quoted 中文"}],
    }
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(
            sql.SQL("""CREATE TABLE {} (
            event_id TEXT NOT NULL, request_id TEXT, status TEXT, model TEXT,
            start_time TIMESTAMPTZ, end_time TIMESTAMPTZ, prompt_tokens BIGINT,
            completion_tokens BIGINT, total_tokens BIGINT, response_cost DOUBLE PRECISION, payload TEXT
        )""").format(sql.Identifier(table))
        )
        logger: Final = PostgresOlapLogger(
            PostgresOlapConfig(
                dsn=SecretStr(dsn),
                table_name=table,
                batch_size=2,
                turn_off_message_logging=False,
            )
        )
        try:
            logger.log_success_event({"standard_logging_object": {**payload, "status": "success"}}, None, now, now)
            logger.log_failure_event({"standard_logging_object": {**payload, "status": "failure"}}, None, now, now)
            assert logger.flush(20)
            assert logger.statistics == {"accepted": 2, "delivered": 2, "dropped": 0, "pending": 0}
            rows: Final = connection.execute(
                sql.SQL("SELECT request_id,status,total_tokens,response_cost,payload FROM {} ORDER BY status").format(
                    sql.Identifier(table)
                )
            ).fetchall()
            assert len(rows) == 2
            assert tuple(row[:4] for row in rows) == (
                (payload["id"], "failure", 5, 0.125),
                (payload["id"], "success", 5, 0.125),
            )
            assert tuple(json.loads(row[4]) for row in rows) == (
                {**payload, "status": "failure"},
                {**payload, "status": "success"},
            )
        finally:
            logger.close()
            connection.execute(sql.SQL("DROP TABLE {}").format(sql.Identifier(table)))


def test_missing_table_reports_dropped_batch_without_payload_disclosure(caplog: pytest.LogCaptureFixture) -> None:
    logger: Final = PostgresOlapLogger(
        PostgresOlapConfig(
            dsn=SecretStr(make_conninfo(os.environ["DATABASE_URL"], sslmode="disable")),
            table_name="missing_" + uuid4().hex,
            retries=0,
        )
    )
    now: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
    try:
        logger.log_success_event({"standard_logging_object": {"id": "private-synthetic-id"}}, None, now, now)
        assert logger.flush(20)
        assert logger.statistics == {"accepted": 1, "delivered": 0, "dropped": 1, "pending": 0}
        assert "private-synthetic-id" not in caplog.text
    finally:
        logger.close()


@pytest.mark.parametrize("trust", ["trusted", "untrusted", "wrong_host"])
def test_default_tls_verifies_ca_and_hostname(trust: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PGSSLMODE", raising=False)
    certificate: Final = self_signed_certificate(tmp_path)
    other_certificate: Final = self_signed_certificate(tmp_path)
    context: Final = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate.certificate, certificate.key)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", allocate_port()))
        listener.listen(1)
        listener.settimeout(5)

        def serve() -> str:
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(5)
                assert connection.recv(8, socket.MSG_WAITALL) == struct.pack("!II", 8, 80877103)
                connection.sendall(b"S")
                try:
                    with context.wrap_socket(connection, server_side=True) as secure, secure.makefile("rb") as stream:
                        header: Final = stream.read(4)
                        if not header:
                            return "rejected"
                        length: Final = struct.unpack("!I", header)[0]
                        stream.read(length - 4)
                        error: Final = b"SFATAL\x00C28000\x00Msynthetic authentication failure\x00\x00"
                        secure.sendall(b"E" + struct.pack("!I", len(error) + 4) + error)
                except ssl.SSLError:
                    return "rejected"
            return "tls"

        dsn: Final = make_conninfo(
            host="wrong.invalid" if trust == "wrong_host" else "127.0.0.1",
            hostaddr="127.0.0.1",
            port=listener.getsockname()[1],
            user="synthetic",
            dbname="test",
            gssencmode="disable",
            sslrootcert=str(other_certificate.certificate if trust == "untrusted" else certificate.certificate),
        )
        with ThreadPoolExecutor(max_workers=1) as pool:
            exchange: Final = pool.submit(serve)
            expected: Final = {
                "trusted": "synthetic authentication failure",
                "untrusted": "certificate verify failed",
                "wrong_host": "does not match host name",
            }
            with pytest.raises(psycopg.OperationalError) as error:
                write_batch(PostgresOlapConfig(dsn=SecretStr(dsn)), (("event", *([None] * 10)),))
            assert exchange.result(timeout=5) == ("tls" if trust == "trusted" else "rejected")
            assert expected[trust] in str(error.value)
