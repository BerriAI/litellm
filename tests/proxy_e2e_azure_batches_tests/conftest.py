"""
Pytest configuration for Azure Batch E2E Tests.

This conftest manages:
1. Mock Azure Batch server (FastAPI on port 8090)
2. LiteLLM proxy server (port 4000)
3. PostgreSQL database setup
"""

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Generator

import httpx
import pytest

_test_dir = Path(__file__).parent
sys.path.insert(0, str(_test_dir.parent.parent))  # litellm root
sys.path.insert(0, str(_test_dir))  # test directory for local imports

LOG_DIR = _test_dir


def pytest_configure(config):
    """Ensure test directory is in Python path before collection."""
    test_dir = Path(__file__).parent
    if str(test_dir) not in sys.path:
        sys.path.insert(0, str(test_dir))


MOCK_SERVER_PORT = 8090
MOCK_SERVER_URL = f"http://localhost:{MOCK_SERVER_PORT}"
LITELLM_PROXY_PORT = 4000
LITELLM_PROXY_URL = f"http://localhost:{LITELLM_PROXY_PORT}"
DATABASE_URL = "postgresql://llmproxy:dbpassword9090@localhost:5432/litellm"


def kill_process_on_port(port: int) -> None:
    """Kill any process using the specified port."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                try:
                    subprocess.run(["kill", "-9", pid.strip()], timeout=5)
                except Exception:
                    pass
            time.sleep(1)
    except Exception:
        pass


def wait_for_server(url: str, max_attempts: int = 30, delay: float = 1.0) -> bool:
    """Wait for a server to become available at url/health.

    Any HTTP response (including 401) means the server is up.
    Only connection errors count as "not ready yet".
    """
    for attempt in range(max_attempts):
        try:
            response = httpx.get(f"{url}/health", timeout=2.0)
            return True
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError):
            pass
        except Exception:
            pass
        if attempt < max_attempts - 1:
            time.sleep(delay)
    return False


def _read_log_tail(log_path: Path, max_lines: int = 80) -> str:
    """Read the last N lines of a log file, returning empty string if not found."""
    if not log_path.exists():
        return "(log file not found)"
    try:
        text = log_path.read_text()
        lines = text.strip().splitlines()
        if len(lines) > max_lines:
            return f"... ({len(lines) - max_lines} lines truncated) ...\n" + "\n".join(
                lines[-max_lines:]
            )
        return text
    except Exception as e:
        return f"(error reading log: {e})"


def _check_process_alive(process: subprocess.Popen, label: str, log_path: Path):
    """Check if a subprocess crashed immediately after starting.
    Raises pytest.fail with log output if the process has already exited.
    """
    time.sleep(1)
    exit_code = process.poll()
    if exit_code is not None:
        log_output = _read_log_tail(log_path)
        pytest.fail(
            f"{label} exited immediately with code {exit_code}.\n"
            f"--- {label} log ({log_path}) ---\n{log_output}\n"
            f"--- end log ---"
        )


def setup_database() -> bool:
    """Ensure PostgreSQL database exists and is accessible."""
    try:
        import psycopg2

        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="litellm",
            user="llmproxy",
            password="dbpassword9090",
            connect_timeout=5,
        )
        conn.close()
        return True
    except ImportError:
        print("WARNING: psycopg2 not installed — cannot verify database")
        return False
    except Exception:
        return False


@pytest.fixture(scope="session")
def mock_azure_server() -> Generator[str, None, None]:
    """Start mock Azure batch server as a subprocess."""
    print(f"\n{'=' * 60}")
    print("Setting up Mock Azure Batch Server")
    print(f"{'=' * 60}")

    kill_process_on_port(MOCK_SERVER_PORT)

    runner_script = Path(__file__).parent / "fixtures" / "run_mock_server.py"
    runner_script.write_text(
        """
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fixtures.mock_azure_batch_server import create_mock_azure_batch_server
import uvicorn

if __name__ == "__main__":
    app = create_mock_azure_batch_server()
    uvicorn.run(app, host="0.0.0.0", port=8090, log_level="info", access_log=False)
"""
    )

    mock_log = LOG_DIR / "mock_server.log"
    log_file = open(mock_log, "w")

    print(f"Starting mock server on port {MOCK_SERVER_PORT}...")
    print(f"Log file: {mock_log}")
    process = subprocess.Popen(
        [sys.executable, str(runner_script)],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=Path(__file__).parent,
    )

    _check_process_alive(process, "Mock server", mock_log)

    if not wait_for_server(MOCK_SERVER_URL, max_attempts=30, delay=1.0):
        log_output = _read_log_tail(mock_log)
        exit_code = process.poll()
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log_file.close()
        pytest.fail(
            f"Mock server failed to start on port {MOCK_SERVER_PORT} "
            f"(process exit_code={exit_code}).\n"
            f"--- mock server log ---\n{log_output}\n--- end log ---\n"
            f"Hint: ensure 'uvicorn' and 'fastapi' are installed."
        )

    print(f"Mock Azure server ready at {MOCK_SERVER_URL}")
    yield MOCK_SERVER_URL

    print("\nShutting down mock server...")
    try:
        process.terminate()
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    log_file.close()
    print("Mock server stopped")


@pytest.fixture(scope="session")
def litellm_proxy_server(mock_azure_server: str) -> Generator[str, None, None]:
    """Start LiteLLM proxy server for the test session."""
    print(f"\n{'=' * 60}")
    print("Setting up LiteLLM Proxy Server")
    print(f"{'=' * 60}")

    if not setup_database():
        pytest.skip(
            "PostgreSQL database not available at localhost:5432. "
            "Start PostgreSQL and create a 'litellm' database:\n"
            "  docker run -d --name litellm-db -p 5432:5432 "
            '-e POSTGRES_USER=llmproxy -e POSTGRES_PASSWORD=dbpassword9090 '
            "-e POSTGRES_DB=litellm postgres:15\n"
            "Then run: prisma db push --schema=litellm/proxy/schema.prisma"
        )
    print("Database connection verified")

    config_path = Path(__file__).parent / "fixtures" / "config.yml"
    if not config_path.exists():
        pytest.fail(f"Config file not found: {config_path}")
    print("Config file found")

    kill_process_on_port(LITELLM_PROXY_PORT)

    os.environ["MOCK_SERVER_URL_V1"] = f"{mock_azure_server}/v1"
    os.environ["MOCK_SERVER_URL_OPENAI_V1"] = f"{mock_azure_server}/openai/v1"
    os.environ["DATABASE_URL"] = DATABASE_URL
    os.environ["USE_LOCAL_LITELLM"] = "true"
    os.environ["USE_MOCK_MODELS"] = "true"
    os.environ["USE_STATE_TRACKER"] = "true"
    os.environ["PROXY_BATCH_POLLING_INTERVAL"] = "10"

    print("Environment configured")

    print(f"Starting LiteLLM proxy on port {LITELLM_PROXY_PORT}...")
    litellm_root = Path(__file__).parent.parent.parent

    cmd = [
        sys.executable,
        "-m",
        "litellm.proxy.proxy_cli",
        "--config",
        str(config_path),
        "--port",
        str(LITELLM_PROXY_PORT),
        "--detailed_debug",
    ]

    proxy_log = LOG_DIR / "proxy_server.log"
    log_file = open(proxy_log, "w")
    print(f"Log file: {proxy_log}")

    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
        cwd=litellm_root,
    )

    _check_process_alive(process, "LiteLLM proxy", proxy_log)

    if not wait_for_server(LITELLM_PROXY_URL, max_attempts=60, delay=1.0):
        log_output = _read_log_tail(proxy_log)
        exit_code = process.poll()
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log_file.close()
        pytest.fail(
            f"LiteLLM proxy failed to start on port {LITELLM_PROXY_PORT} "
            f"(process exit_code={exit_code}).\n"
            f"--- proxy log (last 80 lines) ---\n{log_output}\n--- end log ---\n"
            f"Hints:\n"
            f"  1. Ensure Prisma client is generated: "
            f"cd {litellm_root} && prisma generate --schema=litellm/proxy/schema.prisma\n"
            f"  2. Ensure DB migrations are applied: "
            f"prisma db push --schema=litellm/proxy/schema.prisma\n"
            f"  3. Check the full log at: {proxy_log}"
        )

    print(f"LiteLLM proxy ready at {LITELLM_PROXY_URL}")
    yield LITELLM_PROXY_URL

    print("\nShutting down LiteLLM proxy...")
    try:
        process.terminate()
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    log_file.close()
    print("LiteLLM proxy stopped")


@pytest.fixture(scope="session")
def event_loop():
    """Provide an event loop for async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()
