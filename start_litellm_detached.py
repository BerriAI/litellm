import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


CREATE_NEW_PROCESS_GROUP = 0x00000200
DETACHED_PROCESS = 0x00000008
CREATE_BREAKAWAY_FROM_JOB = 0x01000000

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / ".venv" / "Scripts"
LITELLM = SCRIPTS / "litellm.exe"
CONFIG = ROOT / "litellm_config.yaml"
PG_DATA = Path(r"C:\Users\18747\scoop\persist\postgresql\data")
PG_LOG = ROOT / "postgresql.log"
OUT_LOG = ROOT / "litellm_server.out.log"
ERR_LOG = ROOT / "litellm_server.err.log"
PID_FILE = ROOT / "litellm_server.pid"
ENV_FILE = ROOT / ".env"
HEALTH_URL = "http://127.0.0.1:4000/health/liveliness"


def port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PATH"] = str(SCRIPTS) + os.pathsep + env.get("PATH", "")
    if ENV_FILE.exists():
        for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            env.setdefault(key, value)
    return env


def health_ok(timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=timeout) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def wait_for_health(seconds: int) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if health_ok(timeout=2.0):
            return True
        time.sleep(1)
    return False


def tail(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(data[-lines:])


def ensure_postgres(env: dict[str, str]) -> None:
    if port_open("127.0.0.1", 5432):
        return

    with PG_LOG.open("ab") as log:
        result = subprocess.run(
            ["pg_ctl", "start", "-D", str(PG_DATA), "-l", str(PG_LOG)],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=log,
        )
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    for _ in range(30):
        if port_open("127.0.0.1", 5432):
            return
        time.sleep(1)
    raise SystemExit("PostgreSQL started command returned, but port 5432 is still closed.")


def main() -> int:
    env = build_env()
    if health_ok():
        print("LiteLLM is already running: http://localhost:4000/ui/")
        return 0

    if port_open("127.0.0.1", 4000):
        print("Port 4000 is occupied, but LiteLLM health check is not responding.")
        print("Close the process using port 4000, then run this script again.")
        return 1

    ensure_postgres(env)

    out = OUT_LOG.open("ab", buffering=0)
    err = ERR_LOG.open("ab", buffering=0)
    creationflags = CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS | CREATE_BREAKAWAY_FROM_JOB

    proc = subprocess.Popen(
        [str(LITELLM), "--config", str(CONFIG), "--port", "4000"],
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=out,
        stderr=err,
        close_fds=True,
        creationflags=creationflags,
    )
    PID_FILE.write_text(str(proc.pid), encoding="utf-8")
    print(f"Started LiteLLM process PID {proc.pid}. Waiting for health check...")

    if wait_for_health(90):
        print("LiteLLM is running: http://localhost:4000/ui/")
        return 0

    print("LiteLLM did not become healthy within 90 seconds.")
    print("")
    print("--- stderr tail ---")
    print(tail(ERR_LOG) or "(empty)")
    print("")
    print("--- stdout tail ---")
    print(tail(OUT_LOG) or "(empty)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
