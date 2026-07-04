import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "litellm_proxy_4000.supervisor.log"


def write_log(message: str) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"[{datetime.now().isoformat()}] {message}\n")
        log_file.flush()


def load_dotenv(env: dict[str, str]) -> None:
    dotenv_path = ROOT / ".env"
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        env[name] = value


def main() -> int:
    env = os.environ.copy()
    load_dotenv(env)
    env["ANTHROPIC_BASE_URL"] = "https://api.kimi.com/coding?beta=true"
    env["ANTHROPIC_API_BASE"] = ""
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONLEGACYWINDOWSSTDIO"] = "0"

    python_exe = ROOT / ".venv" / "Scripts" / "python.exe"
    cmd = [
        str(python_exe),
        "-m",
        "litellm.proxy.proxy_cli",
        "--config",
        "litellm_config.yaml",
        "--port",
        "4000",
    ]

    write_log(f"supervisor pid={os.getpid()} cwd={ROOT}")
    while True:
        write_log("starting litellm on port 4000")
        with LOG_PATH.open("ab", buffering=0) as log_file:
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
            proc = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=env,
                stdin=subprocess.PIPE,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )

            write_log(f"litellm child pid={proc.pid}")
            while proc.poll() is None:
                time.sleep(5)

            write_log(f"litellm exited code={proc.returncode}")

        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
