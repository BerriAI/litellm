from __future__ import annotations

import os
import re
import shutil
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Final, cast
from urllib.parse import quote, urlsplit, urlunsplit

import psycopg
import pytest
from psycopg import sql
from psycopg.types.json import Jsonb

from tests.integration._support.client import eventually
from tests.integration._support.process import _free_port as free_port

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
MIGRATIONS_DIR: Final = REPO_ROOT / "litellm-proxy-extras" / "litellm_proxy_extras" / "migrations"
MIGRATION_NAME: Final = "20260921190000_agent_identity"
BASELINE_DIR: Final = MIGRATIONS_DIR / "0_init"
PASSWORD: Final = "wr ong'pw9"
FRAGMENTS: Final = ("wr ong", "ong'pw9", "wr%20ong", "ong%27pw9", "pw9")
SHIPPED_MIGRATIONS: Final = tuple(
    sorted(path.name for path in MIGRATIONS_DIR.iterdir() if path.is_dir() and path.name != "0_init")
)
LOG_PREFIX: Final = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - [^\n]+ - "
LOG_RECORD_START: Final = rf"{LOG_PREFIX}(?:DEBUG|INFO|WARNING|ERROR|CRITICAL) - "
ERROR_RECORD: Final = re.compile(rf"(?ms)^({LOG_PREFIX}ERROR - .*?)(?={LOG_RECORD_START}|\Z)")
RETRY_COUNT: Final = re.compile(r"Retrying\.\.\. \((\d+) attempts left\)")
FRAGMENT_PATTERN: Final = re.compile(
    "|".join(re.escape(fragment) for fragment in sorted(FRAGMENTS, key=len, reverse=True))
)


def _database_url(
    admin_url: str,
    role: str,
    password: str,
    database: str,
    *,
    encode_password: bool = True,
) -> str:
    parsed: Final = urlsplit(admin_url)
    authority: Final = parsed.netloc.rsplit("@", 1)[-1]
    encoded_password: Final = quote(password, safe="") if encode_password else password
    netloc: Final = f"{quote(role, safe='')}:{encoded_password}@{authority}"
    return urlunsplit(parsed._replace(netloc=netloc, path=f"/{database}"))


def _replace_port(database_url: str, port: int, hostname: str | None = None) -> str:
    parsed: Final = urlsplit(database_url)
    target_host: Final = hostname or parsed.hostname
    assert target_host is not None
    host: Final = f"[{target_host}]" if ":" in target_host else target_host
    userinfo: Final = parsed.netloc.rsplit("@", 1)[0]
    return urlunsplit(parsed._replace(netloc=f"{userinfo}@{host}:{port}"))


@contextmanager
def owned_database(password: str) -> Iterator[str]:
    admin_url: Final = os.environ["DATABASE_URL"]
    role: Final = f"v1_migration_{uuid.uuid4().hex}"
    database: Final = f"v1_migration_{uuid.uuid4().hex}"
    database_url: Final = _database_url(admin_url, role, password, database)
    try:
        with psycopg.connect(admin_url, autocommit=True) as admin:
            admin.execute(
                sql.SQL("CREATE ROLE {} WITH LOGIN PASSWORD {}").format(sql.Identifier(role), sql.Literal(password))
            )
            admin.execute(sql.SQL("CREATE DATABASE {} OWNER {}").format(sql.Identifier(database), sql.Identifier(role)))
        yield database_url
    finally:
        with psycopg.connect(admin_url, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(database)))
            admin.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))


def _migration_invocation(
    database_url: str, tmp_path: Path, extra_env: Mapping[str, str]
) -> tuple[tuple[str, ...], dict[str, str]]:
    config_path: Final = tmp_path / "config.yaml"
    config_path.write_text(
        "model_list:\n  - model_name: integration-fake\n    litellm_params:\n      model: openai/integration-fake\n"
    )
    command: Final = (
        sys.executable,
        "-I",
        "-m",
        "litellm.proxy.proxy_cli",
        "--config",
        str(config_path),
        "--use_legacy_migration_resolver",
        "--skip_server_startup",
    )
    environment: Final = {
        **{key: value for key, value in os.environ.items() if key not in ("DIRECT_URL", "USE_V2_MIGRATION_RESOLVER")},
        "DATABASE_URL": database_url,
        "LITELLM_LOG": "ERROR",
        **extra_env,
    }
    return command, environment


def run_v1_migrations(
    database_url: str, tmp_path: Path, extra_env: Mapping[str, str]
) -> subprocess.CompletedProcess[str]:
    command, environment = _migration_invocation(database_url, tmp_path, extra_env)
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=900,
    )


def error_lines(output: str) -> tuple[str, ...]:
    return tuple(match.group(1) for match in ERROR_RECORD.finditer(output))


def _retry_count_texts(lines: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(value for value in (_retry_count_text(line) for line in lines) if value is not None)


def _retry_count_text(line: str) -> str | None:
    match: Final = RETRY_COUNT.search(line)
    return match.group(1) if match is not None else None


def _safe_output(output: str) -> str:
    return FRAGMENT_PATTERN.sub("[REDACTED]", output)


def _assert_no_password_fragments(output: str) -> None:
    assert [fragment for fragment in FRAGMENTS if fragment in output] == [], _safe_output(output)


def _applied_migrations(database_url: str) -> tuple[str, ...]:
    with psycopg.connect(database_url) as connection:
        rows: Final = connection.execute(
            'SELECT migration_name FROM "_prisma_migrations" '
            "WHERE finished_at IS NOT NULL AND rolled_back_at IS NULL ORDER BY migration_name"
        ).fetchall()
    return tuple(str(row[0]) for row in rows)


def _duplicate_migrations(database_url: str) -> tuple[tuple[str, int], ...]:
    with psycopg.connect(database_url) as connection:
        rows: Final = connection.execute(
            'SELECT migration_name, COUNT(*) FROM "_prisma_migrations" '
            "GROUP BY migration_name HAVING COUNT(*) > 1 ORDER BY migration_name"
        ).fetchall()
    return tuple((str(row[0]), int(row[1])) for row in rows)


def _retry_p3018_migration(database_url: str) -> None:
    agent_ids: Final = (uuid.uuid4().hex, uuid.uuid4().hex)
    with psycopg.connect(database_url) as connection:
        connection.execute('DROP INDEX "LiteLLM_AgentIdentity_provider_tenant_id_client_id_key"')
        for agent_id in agent_ids:
            connection.execute(
                'INSERT INTO "LiteLLM_AgentsTable" '
                '("agent_id", "agent_name", "agent_card_params", "created_by", "updated_by") '
                "VALUES (%s, %s, %s, %s, %s)",
                (agent_id, f"audit-agent-{agent_id}", Jsonb({}), "integration", "integration"),
            )
            connection.execute(
                'INSERT INTO "LiteLLM_AgentIdentity" '
                '("agent_id", "provider", "issuer", "tenant_id", "client_id", "revision") '
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (agent_id, "entra", f"https://audit.invalid/{agent_id}", "tenant-1", "client-1", uuid.uuid4().hex),
            )
        connection.execute('DELETE FROM "_prisma_migrations" WHERE migration_name = %s', (MIGRATION_NAME,))


def _relay(source: socket.socket, destination: socket.socket) -> None:
    try:
        while data := source.recv(65536):
            destination.sendall(data)
    except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
        return


class _PostgresForwardingServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    target: tuple[str, int]

    def __init__(self, port: int, target: tuple[str, int]) -> None:
        self.target = target
        super().__init__(("127.0.0.1", port), _PostgresForwardingHandler)


class _PostgresForwardingHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server: Final = cast(_PostgresForwardingServer, self.server)
        with socket.create_connection(server.target, timeout=10) as upstream:
            reply: Final = threading.Thread(target=_relay, args=(upstream, self.request), daemon=True)
            reply.start()
            try:
                _relay(self.request, upstream)
            finally:
                with suppress(OSError):
                    self.request.shutdown(socket.SHUT_WR)
                with suppress(OSError):
                    upstream.shutdown(socket.SHUT_WR)
                reply.join(timeout=10)


@contextmanager
def _postgres_forwarder(port: int, target: tuple[str, int]) -> Iterator[None]:
    server: Final = _PostgresForwardingServer(port, target)
    thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)


def _migration_log_has_p1001_or_process_exited(output_path: Path, process: subprocess.Popen[str]) -> tuple[bool, bool]:
    p1001_logged: Final = any("P1001" in line for line in error_lines(output_path.read_text()))
    process_exited: Final = process.poll() is not None
    return p1001_logged, process_exited


def test_unreachable_database_emits_four_p1001_errors_without_password_fragments(tmp_path: Path) -> None:
    with owned_database(PASSWORD) as database_url:
        parsed: Final = urlsplit(database_url)
        unreachable_url: Final = _replace_port(
            _database_url(
                database_url,
                parsed.username or "",
                PASSWORD,
                parsed.path.lstrip("/"),
                encode_password=False,
            ),
            free_port(),
        )
        completed: Final = run_v1_migrations(unreachable_url, tmp_path, {})
        output: Final = completed.stdout + completed.stderr
        errors: Final = error_lines(output)
        p1001_errors: Final = tuple(line for line in errors if "P1001" in line)
        assert completed.returncode == 1, _safe_output(output)
        assert len(errors) == 4, _safe_output(output)
        assert len(p1001_errors) == 4, _safe_output(output)
        assert _retry_count_texts(errors) == (), _safe_output(output)
        _assert_no_password_fragments(output)


def test_wrong_password_emits_four_p1000_errors_without_password_fragments(tmp_path: Path) -> None:
    with owned_database(f"correct-{uuid.uuid4().hex}") as database_url:
        parsed: Final = urlsplit(database_url)
        wrong_url: Final = _database_url(
            database_url,
            parsed.username or "",
            PASSWORD,
            parsed.path.lstrip("/"),
        )
        completed: Final = run_v1_migrations(wrong_url, tmp_path, {})
        output: Final = completed.stdout + completed.stderr
        errors: Final = error_lines(output)
        p1000_errors: Final = tuple(line for line in errors if "P1000" in line)
        assert completed.returncode == 1, _safe_output(output)
        assert len(errors) == 4, _safe_output(output)
        assert len(p1000_errors) == 4, _safe_output(output)
        assert _retry_count_texts(errors) == (), _safe_output(output)
        _assert_no_password_fragments(output)


@pytest.mark.timeout(900)
def test_duplicate_agent_identity_logs_the_p3018_migration_error(tmp_path: Path) -> None:
    with owned_database(PASSWORD) as database_url:
        setup: Final = run_v1_migrations(database_url, tmp_path, {})
        setup_output: Final = setup.stdout + setup.stderr
        assert setup.returncode == 0, _safe_output(setup_output)
        _retry_p3018_migration(database_url)
        completed: Final = run_v1_migrations(database_url, tmp_path, {})
        output: Final = completed.stdout + completed.stderr
        errors: Final = error_lines(output)
        p3018_errors: Final = tuple(line for line in errors if "P3018" in line)
        expected_markers: Final = ((True, True), (True, True))
        assert completed.returncode == 1, _safe_output(output)
        assert len(errors) == 2, _safe_output(output)
        assert tuple((MIGRATION_NAME in line, "P3018" in line) for line in p3018_errors) == expected_markers, (
            _safe_output(output)
        )
        assert _retry_count_texts(errors) == ("3", "1"), _safe_output(output)
        _assert_no_password_fragments(output)


@pytest.mark.timeout(900)
def test_p3005_baseline_recovery_records_zero_init_and_logs_no_errors(tmp_path: Path) -> None:
    existed_before: Final = BASELINE_DIR.exists()
    try:
        with owned_database(PASSWORD) as database_url:
            with psycopg.connect(database_url) as connection:
                connection.execute('CREATE TABLE "audit_unrelated_table" ("id" INTEGER)')
            completed: Final = run_v1_migrations(database_url, tmp_path, {})
            output: Final = completed.stdout + completed.stderr
            expected_migrations: Final = tuple(sorted(("0_init", *SHIPPED_MIGRATIONS)))
            assert completed.returncode == 0, _safe_output(output)
            assert error_lines(output) == (), _safe_output(output)
            assert _applied_migrations(database_url) == expected_migrations, _safe_output(output)
            _assert_no_password_fragments(output)
    finally:
        if not existed_before and BASELINE_DIR.exists():
            shutil.rmtree(BASELINE_DIR)


def test_clean_database_applies_exactly_the_shipped_migrations(tmp_path: Path) -> None:
    with owned_database(PASSWORD) as database_url:
        completed: Final = run_v1_migrations(database_url, tmp_path, {})
        output: Final = completed.stdout + completed.stderr
        assert completed.returncode == 0, _safe_output(output)
        assert error_lines(output) == (), _safe_output(output)
        assert _applied_migrations(database_url) == SHIPPED_MIGRATIONS, _safe_output(output)
        _assert_no_password_fragments(output)


def test_unreachable_database_recovers_after_postgres_forwarder_starts(tmp_path: Path) -> None:
    with owned_database(PASSWORD) as database_url:
        admin_url: Final = os.environ["DATABASE_URL"]
        admin: Final = urlsplit(admin_url)
        hostname: Final = admin.hostname
        port: Final = admin.port
        assert hostname is not None and port is not None
        forwarding_port: Final = free_port()
        forwarded_url: Final = _replace_port(database_url, forwarding_port, "127.0.0.1")
        command, environment = _migration_invocation(forwarded_url, tmp_path, {})
        output_path: Final = tmp_path / "migration-output.log"
        with output_path.open("w") as output_file:
            process: Final = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=output_file,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                observation: Final = eventually(
                    lambda: _migration_log_has_p1001_or_process_exited(output_path, process),
                    lambda state: state[0] or state[1],
                    seconds=900,
                )
                assert observation[0], _safe_output(output_path.read_text())
                target_host: Final = "127.0.0.1" if hostname == "localhost" else hostname
                with _postgres_forwarder(forwarding_port, (target_host, port)):
                    completed_returncode: Final = process.wait(timeout=900)
            finally:
                _stop_process(process)
        output: Final = output_path.read_text()
        errors: Final = error_lines(output)
        p1001_errors: Final = tuple(line for line in errors if "P1001" in line)
        assert completed_returncode == 0, _safe_output(output)
        assert len(p1001_errors) == 1, _safe_output(output)
        assert _retry_count_texts(errors) == (), _safe_output(output)
        assert _duplicate_migrations(database_url) == (), _safe_output(output)
        assert _applied_migrations(database_url) == SHIPPED_MIGRATIONS, _safe_output(output)
        _assert_no_password_fragments(output)


def test_unreachable_database_keeps_password_masked_when_shape_redaction_is_disabled(tmp_path: Path) -> None:
    with owned_database(PASSWORD) as database_url:
        parsed: Final = urlsplit(database_url)
        unreachable_url: Final = _replace_port(
            _database_url(
                database_url,
                parsed.username or "",
                PASSWORD,
                parsed.path.lstrip("/"),
                encode_password=False,
            ),
            free_port(),
        )
        completed: Final = run_v1_migrations(
            unreachable_url,
            tmp_path,
            {"LITELLM_DISABLE_REDACT_SECRETS": "true"},
        )
        output: Final = completed.stdout + completed.stderr
        errors: Final = error_lines(output)
        p1001_errors: Final = tuple(line for line in errors if "P1001" in line)
        assert completed.returncode == 1, _safe_output(output)
        assert len(errors) == 4, _safe_output(output)
        assert len(p1001_errors) == 4, _safe_output(output)
        assert _retry_count_texts(errors) == (), _safe_output(output)
        _assert_no_password_fragments(output)
