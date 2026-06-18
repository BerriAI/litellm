import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]

REPRO_CODE = r"""
import asyncio
import os
import signal
import sys
import types
from pathlib import Path


async def noop_failure_handler(*_args, **_kwargs) -> None:
    return None


def add_repo_root_to_python_path() -> None:
    repo_root = Path(os.environ["LITELLM_REPO_ROOT"])
    sys.path.insert(0, str(repo_root))


def configure_breaker_for_fast_repro() -> None:
    breaker_action = os.environ["REPRO_BREAKER_ACTION"]
    os.environ["PRISMA_RECONNECT_COOLDOWN_SECONDS"] = "1"
    os.environ["LITELLM_LOG"] = "INFO"
    os.environ["PRISMA_HEALTH_WATCHDOG_PROBE_TIMEOUT_SECONDS"] = "0.5"
    os.environ["PRISMA_WATCHDOG_RECONNECT_TIMEOUT_SECONDS"] = "1"
    os.environ["PRISMA_RECONNECT_CIRCUIT_BREAKER_ENABLED"] = "true"
    os.environ["PRISMA_RECONNECT_CIRCUIT_BREAKER_ACTION"] = breaker_action
    os.environ["PRISMA_RECONNECT_CIRCUIT_BREAKER_WINDOW_SECONDS"] = "60"
    os.environ["PRISMA_RECONNECT_CIRCUIT_BREAKER_MAX_ATTEMPTS"] = "10"
    os.environ["PRISMA_RECONNECT_CIRCUIT_BREAKER_MAX_FAILURES"] = "2"
    os.environ["PRISMA_RECONNECT_CIRCUIT_BREAKER_MAX_ENGINE_DEATHS"] = (
        "1" if breaker_action == "exit" else "10"
    )


def handle_sigterm(_signum, _frame) -> None:
    print("Received SIGTERM from Prisma reconnect circuit breaker; exiting cleanly.")
    raise SystemExit(0)


async def main() -> None:
    add_repo_root_to_python_path()
    configure_breaker_for_fast_repro()
    signal.signal(signal.SIGTERM, handle_sigterm)

    from litellm.proxy.utils import PrismaClient

    proxy_logging = types.SimpleNamespace(failure_handler=noop_failure_handler)
    client = PrismaClient(
        database_url=os.environ["DATABASE_URL"],
        proxy_logging_obj=proxy_logging,
    )
    client._db_reconnect_cooldown_seconds = 0
    client._db_health_watchdog_interval_seconds = 0
    client._db_health_watchdog_probe_timeout_seconds = 0.5
    client._db_watchdog_reconnect_timeout_seconds = 1

    breaker_action = os.environ["REPRO_BREAKER_ACTION"]
    print(f"Breaker action: {breaker_action}")
    print("DATABASE_URL: <redacted>")
    if breaker_action == "exit":
        print("Expected: engine death opens breaker and sends SIGTERM.")
        await client.attempt_db_reconnect(
            reason="engine_process_death",
            force=True,
            timeout_seconds=1,
            engine_pid=1234,
        )
    else:
        print("Expected: breaker opens in log mode and process remains alive.")
        for attempt in range(1, 3):
            result = await client.attempt_db_reconnect(
                reason="db_health_watchdog_connection_error",
                force=True,
                timeout_seconds=1,
            )
            print(f"Reconnect attempt {attempt} result: {result}")
        print("Log action complete: breaker opened and process is still alive.")
        return

    raise RuntimeError("repro failed: circuit breaker did not terminate process")


if __name__ == "__main__":
    asyncio.run(main())
"""


pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="Requires DATABASE_URL for Prisma DB e2e test",
)


def _run_repro(action: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "DATABASE_URL": os.environ["DATABASE_URL"],
        "LITELLM_REPO_ROOT": str(REPO_ROOT),
        "REPRO_BREAKER_ACTION": action,
    }
    return subprocess.run(
        [
            sys.executable,
            "-c",
            REPRO_CODE,
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_prisma_reconnect_circuit_breaker_log_action_does_not_exit():
    result = _run_repro("log")
    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    assert "Breaker action: log" in output
    assert "Prisma DB reconnect circuit breaker opened" in output
    assert "Log action complete: breaker opened and process is still alive." in output
    assert "Received SIGTERM from Prisma reconnect circuit breaker" not in output


def test_prisma_reconnect_circuit_breaker_exit_action_exits_cleanly():
    result = _run_repro("exit")
    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    assert "Breaker action: exit" in output
    assert "Prisma DB reconnect circuit breaker opened" in output
    assert "Received SIGTERM from Prisma reconnect circuit breaker; exiting cleanly." in output
