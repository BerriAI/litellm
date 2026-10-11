from __future__ import annotations

import json
import os
import re
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast
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
pytestmark: Final = pytest.mark.timeout(300)
PASSWORD: Final = "wr ong'pw9"
FRAGMENTS: Final = ("wr ong", "wr+ong", "ong'pw9", "wr%20ong", "ong%27pw9", "pw9")
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


@dataclass(frozen=True, slots=True)
class MigrationResult:
    returncode: int
    output: str


def _migration_log_path(test_name: str, tmp_path: Path) -> Path:
    log_directory: Final = (
        Path(os.environ["INTEGRATION_RESULTS_DIR"]) if "INTEGRATION_RESULTS_DIR" in os.environ else tmp_path
    )
    log_path: Final = log_directory / f"v1-migration-{test_name}-{uuid.uuid4().hex[:8]}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path


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


def _unreachable_database_url(database_url: str) -> str:
    parsed: Final = urlsplit(database_url)
    url: Final = _database_url(
        database_url,
        parsed.username or "",
        PASSWORD,
        parsed.path.lstrip("/"),
        encode_password=False,
    )
    return _replace_port(url, free_port())


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


def _migration_environment(database_url: str | None, extra_env: Mapping[str, str]) -> dict[str, str]:
    excluded_variables: Final = (
        ("DIRECT_URL", "USE_V2_MIGRATION_RESOLVER")
        if database_url is not None
        else ("DATABASE_URL", "DIRECT_URL", "USE_V2_MIGRATION_RESOLVER")
    )
    inherited_environment: Final = {key: value for key, value in os.environ.items() if key not in excluded_variables}
    database_environment: Final = {"DATABASE_URL": database_url} if database_url is not None else {}
    return {
        **inherited_environment,
        **database_environment,
        "LITELLM_LOG": "ERROR",
        **extra_env,
    }


def _migration_invocation(
    database_url: str | None,
    tmp_path: Path,
    extra_env: Mapping[str, str],
    resolver: Literal["legacy", "v2"] = "legacy",
) -> tuple[tuple[str, ...], dict[str, str]]:
    config_path: Final = tmp_path / "config.yaml"
    config_path.write_text(
        "model_list:\n  - model_name: integration-fake\n    litellm_params:\n      model: openai/integration-fake\n"
    )
    resolver_flag: Final = "--use_legacy_migration_resolver" if resolver == "legacy" else "--use_v2_migration_resolver"
    command: Final = (
        sys.executable,
        "-I",
        "-m",
        "litellm.proxy.proxy_cli",
        "--config",
        str(config_path),
        resolver_flag,
        "--skip_server_startup",
    )
    environment: Final = _migration_environment(database_url, extra_env)
    return command, environment


def run_v1_migrations(
    database_url: str | None,
    tmp_path: Path,
    extra_env: Mapping[str, str],
    test_name: str,
    resolver: Literal["legacy", "v2"] = "legacy",
) -> MigrationResult:
    command, environment = _migration_invocation(database_url, tmp_path, extra_env, resolver)
    output_path: Final = _migration_log_path(test_name, tmp_path)
    with output_path.open("w") as output_file:
        completed: Final = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=output_file,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=240,
        )
    output: Final = output_path.read_text()
    return MigrationResult(completed.returncode, output)


def error_lines(output: str) -> tuple[str, ...]:
    return tuple(match.group(1) for match in ERROR_RECORD.finditer(output))


def _json_error_records(output: str) -> tuple[dict[str, object], ...]:
    records: Final = tuple(_parse_json_record(line, output) for line in output.splitlines() if line.startswith("{"))
    return tuple(record for record in records if record.get("level") == "ERROR")


def _parse_json_record(line: str, output: str) -> dict[str, object]:
    try:
        record: Final = json.loads(line)
    except json.JSONDecodeError:
        pytest.fail(_safe_output(output))
    assert isinstance(record, dict), _safe_output(output)
    return cast(dict[str, object], record)


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
    request_queue_size = 64
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
def _gated_postgres_forwarder(port: int, target: tuple[str, int]) -> Iterator[Callable[[], None]]:
    server: Final = _PostgresForwardingServer(port, target)
    thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
    try:
        yield thread.start
    finally:
        if thread.is_alive():
            server.shutdown()
        server.server_close()
        if thread.ident is not None:
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


def test_unreachable_database_emits_four_p1001_errors_without_password_fragments(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    with owned_database(PASSWORD) as database_url:
        unreachable_url: Final = _unreachable_database_url(database_url)
        completed: Final = run_v1_migrations(unreachable_url, tmp_path, {}, request.node.name)
        output: Final = completed.output
        errors: Final = error_lines(output)
        p1001_errors: Final = tuple(line for line in errors if "P1001" in line)
        assert completed.returncode == 1, _safe_output(output)
        assert len(errors) == 4, _safe_output(output)
        assert len(p1001_errors) == 4, _safe_output(output)
        assert _retry_count_texts(errors) == (), _safe_output(output)
        _assert_no_password_fragments(output)


def test_wrong_password_emits_four_p1000_errors_without_password_fragments(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    with owned_database(f"correct-{uuid.uuid4().hex}") as database_url:
        parsed: Final = urlsplit(database_url)
        wrong_url: Final = _database_url(
            database_url,
            parsed.username or "",
            PASSWORD,
            parsed.path.lstrip("/"),
        )
        completed: Final = run_v1_migrations(wrong_url, tmp_path, {}, request.node.name)
        output: Final = completed.output
        errors: Final = error_lines(output)
        p1000_errors: Final = tuple(line for line in errors if "P1000" in line)
        assert completed.returncode == 1, _safe_output(output)
        assert len(errors) == 4, _safe_output(output)
        assert len(p1000_errors) == 4, _safe_output(output)
        assert _retry_count_texts(errors) == (), _safe_output(output)
        _assert_no_password_fragments(output)


def test_duplicate_agent_identity_logs_the_p3018_migration_error(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    with owned_database(PASSWORD) as database_url:
        setup: Final = run_v1_migrations(database_url, tmp_path, {}, request.node.name)
        setup_output: Final = setup.output
        assert setup.returncode == 0, _safe_output(setup_output)
        _retry_p3018_migration(database_url)
        completed: Final = run_v1_migrations(database_url, tmp_path, {}, request.node.name)
        output: Final = completed.output
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


def test_clean_database_applies_exactly_the_shipped_migrations(tmp_path: Path, request: pytest.FixtureRequest) -> None:
    with owned_database(PASSWORD) as database_url:
        completed: Final = run_v1_migrations(database_url, tmp_path, {}, request.node.name)
        output: Final = completed.output
        assert completed.returncode == 0, _safe_output(output)
        assert error_lines(output) == (), _safe_output(output)
        assert _applied_migrations(database_url) == SHIPPED_MIGRATIONS, _safe_output(output)
        _assert_no_password_fragments(output)


def test_unreachable_database_recovers_after_postgres_forwarder_starts(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    with owned_database(PASSWORD) as database_url:
        admin_url: Final = os.environ["DATABASE_URL"]
        admin: Final = urlsplit(admin_url)
        hostname: Final = admin.hostname
        port: Final = admin.port
        assert hostname is not None and port is not None
        target_host: Final = "127.0.0.1" if hostname == "localhost" else hostname
        forwarding_port: Final = free_port()
        forwarded_url: Final = _replace_port(database_url, forwarding_port, "127.0.0.1")
        command, environment = _migration_invocation(forwarded_url, tmp_path, {})
        output_path: Final = _migration_log_path(request.node.name, tmp_path)
        with _gated_postgres_forwarder(forwarding_port, (target_host, port)) as open_forwarder:
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
                        seconds=240,
                    )
                    assert observation[0], _safe_output(output_path.read_text())
                    open_forwarder()
                    completed_returncode: Final = process.wait(timeout=240)
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


def test_unreachable_database_keeps_password_masked_when_shape_redaction_is_disabled(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    with owned_database(PASSWORD) as database_url:
        unreachable_url: Final = _unreachable_database_url(database_url)
        completed: Final = run_v1_migrations(
            unreachable_url,
            tmp_path,
            {"LITELLM_DISABLE_REDACT_SECRETS": "true"},
            request.node.name,
        )
        output: Final = completed.output
        errors: Final = error_lines(output)
        p1001_errors: Final = tuple(line for line in errors if "P1001" in line)
        assert completed.returncode == 1, _safe_output(output)
        assert len(errors) == 4, _safe_output(output)
        assert len(p1001_errors) == 4, _safe_output(output)
        assert _retry_count_texts(errors) == (), _safe_output(output)
        _assert_no_password_fragments(output)


def test_component_database_env_vars_with_wrong_password_emit_four_p1000_errors_without_password_fragments(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    correct_password: Final = f"correct-{uuid.uuid4().hex}"
    with owned_database(correct_password) as database_url:
        parsed: Final = urlsplit(database_url)
        host: Final = parsed.hostname
        port: Final = parsed.port
        username: Final = parsed.username
        assert host is not None and port is not None and username is not None
        extra_env: Final = {
            "DATABASE_HOST": f"{host}:{port}",
            "DATABASE_USERNAME": username,
            "DATABASE_PASSWORD": PASSWORD,
            "DATABASE_NAME": parsed.path.lstrip("/"),
        }
        completed: Final = run_v1_migrations(None, tmp_path, extra_env, request.node.name)
        output: Final = completed.output
        errors: Final = error_lines(output)
        p1000_errors: Final = tuple(line for line in errors if "P1000" in line)
        assert completed.returncode == 1, _safe_output(output)
        assert len(errors) == 4, _safe_output(output)
        assert len(p1000_errors) == 4, _safe_output(output)
        assert _retry_count_texts(errors) == (), _safe_output(output)
        _assert_no_password_fragments(output)


def test_json_logs_emit_four_valid_json_p1001_error_records_without_password_fragments(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    with owned_database(PASSWORD) as database_url:
        unreachable_url: Final = _unreachable_database_url(database_url)
        completed: Final = run_v1_migrations(unreachable_url, tmp_path, {"JSON_LOGS": "true"}, request.node.name)
        output: Final = completed.output
        errors: Final = _json_error_records(output)
        messages: Final = tuple(record.get("message") for record in errors)
        assert completed.returncode == 1, _safe_output(output)
        assert len(errors) == 4, _safe_output(output)
        assert tuple(isinstance(message, str) and "P1001" in message for message in messages) == (
            True,
            True,
            True,
            True,
        ), _safe_output(output)
        assert error_lines(output) == (), _safe_output(output)
        _assert_no_password_fragments(output)


def test_migration_job_entrypoint_emits_four_p1001_errors_without_password_fragments(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    with owned_database(PASSWORD) as database_url:
        unreachable_url: Final = _unreachable_database_url(database_url)
        command: Final = (sys.executable, "-I", "-m", "litellm.proxy.prisma_migration")
        environment: Final = _migration_environment(
            unreachable_url,
            {"USE_V2_MIGRATION_RESOLVER": "false"},
        )
        output_path: Final = _migration_log_path(request.node.name, tmp_path)
        with output_path.open("w") as output_file:
            completed: Final = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=output_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=240,
            )
        output: Final = output_path.read_text()
        errors: Final = error_lines(output)
        p1001_errors: Final = tuple(line for line in errors if "P1001" in line)
        assert completed.returncode == 1, _safe_output(output)
        assert len(errors) == 4, _safe_output(output)
        assert len(p1001_errors) == 4, _safe_output(output)
        _assert_no_password_fragments(output)


def test_v2_resolver_unreachable_database_exits_2_and_names_p1001(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    with owned_database(PASSWORD) as database_url:
        unreachable_url: Final = _unreachable_database_url(database_url)
        completed: Final = run_v1_migrations(unreachable_url, tmp_path, {}, request.node.name, resolver="v2")
        output: Final = completed.output
        assert completed.returncode == 2, _safe_output(output)
        assert "P1001" in output, _safe_output(output)
        assert error_lines(output) == (), _safe_output(output)
        _assert_no_password_fragments(output)


def test_v2_resolver_clean_database_applies_exactly_the_shipped_migrations(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    with owned_database(PASSWORD) as database_url:
        completed: Final = run_v1_migrations(database_url, tmp_path, {}, request.node.name, resolver="v2")
        output: Final = completed.output
        assert completed.returncode == 0, _safe_output(output)
        assert error_lines(output) == (), _safe_output(output)
        assert _applied_migrations(database_url) == SHIPPED_MIGRATIONS, _safe_output(output)
        _assert_no_password_fragments(output)
