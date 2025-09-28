#!/usr/bin/env python3
from __future__ import annotations

import os, shlex, socket, subprocess, sys, time, importlib.util

try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass


def _can(host: str, port: int, t: float = 0.5) -> bool:
    import socket as _s
    try:
        with _s.create_connection((host, port), timeout=t):
            return True
    except OSError:
        return False


def _health_ok(host: str, port: int) -> bool:
    try:
        import urllib.request, json as _json
        with urllib.request.urlopen(f"http://{host}:{port}/healthz", timeout=0.75) as r:
            if r.status != 200:
                return False
            data = _json.loads(r.read().decode("utf-8", errors="replace"))
            return bool(isinstance(data, dict) and data.get("ok") is True)
    except Exception:
        return False


def _run(cmd: str, env: dict | None = None) -> int:
    p = subprocess.Popen(cmd, shell=True, env=env)
    try:
        return p.wait()
    except KeyboardInterrupt:
        p.kill()
        return 130


def _free_port(port:int):
    import subprocess, os
    try:
        out=subprocess.check_output(["bash","-lc",f"lsof -t -i :{port} -sTCP:LISTEN || true"], text=True)
        for pid in [int(x) for x in out.strip().split() if x.strip().isdigit()]:
            try:
                os.kill(pid, 9)
            except Exception:
                pass
    except Exception:
        pass


def main() -> int:
    host = os.getenv("MINI_AGENT_API_HOST", "127.0.0.1")
    try:
        port = int(os.getenv("MINI_AGENT_API_PORT", "8788"))
    except Exception:
        port = 8788

    shim_env = os.environ.copy()
    # Ensure node gateway test port is free to avoid collisions
    _free_port(8791)
    # Stop docker tools-stub publishing 8791 if present
    try:
        import subprocess
        out = subprocess.check_output(["bash","-lc","docker ps --format '{{.Names}} {{.Ports}}' | grep 8791 | awk '{print $1}' || true"], text=True).strip()
        if out:
            subprocess.call(["docker","stop", out])
    except Exception:
        pass

    if shim_env.get("MINI_AGENT_ALLOW_DUMMY") is None:
        shim_env["MINI_AGENT_ALLOW_DUMMY"] = "0"

    proc = None
    if not _can(host, port):
        cmd = f"uvicorn local.tools.mini_agent_shim_app:app --host {host} --port {port} --log-level warning"
        proc = subprocess.Popen(shlex.split(cmd), env=shim_env)
    else:
        if not _health_ok(host, port):
            s = socket.socket(); s.bind((host, 0)); free_port = s.getsockname()[1]; s.close()
            port = int(free_port)
            cmd = f"uvicorn local.tools.mini_agent_shim_app:app --host {host} --port {port} --log-level warning"
            proc = subprocess.Popen(shlex.split(cmd), env=shim_env)

    # Wait briefly for readiness of whichever port we chose
    for _ in range(50):
        if _can(host, port):
            break
        time.sleep(0.1)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    env["MINI_AGENT_API_HOST"] = host
    env["MINI_AGENT_API_PORT"] = str(port)
    env.setdefault("MINI_AGENT_ALLOW_DUMMY", "0")
    # Start codex-agent provider by default and point base at shim app
    env.setdefault("LITELLM_ENABLE_CODEX_AGENT", "1")
    env.setdefault("CODEX_AGENT_API_BASE", f"http://{host}:{port}")
    env.setdefault("MINI_AGENT_OPENAI_SHIM_MODE", "echo")
    env.setdefault("LITELLM_DEFAULT_CODE_MODEL", "ollama_chat/qwen2.5-coder:3b")
    env.setdefault("LITELLM_DEFAULT_TEXT_MODEL", "ollama_chat/glm4:latest")
    env.setdefault("OLLAMA_MODEL", "glm4:latest")

    try:
        # Run every smoke directory. Optional tests will self-skip if deps are missing.
        targets=["tests/smoke","tests/smoke_optional","tests/ndsmoke","tests/ndsmoke_e2e"]
        cmd_parts=["pytest","-q"]+targets
        if importlib.util.find_spec("xdist") and os.environ.get("NO_XDIST") != "1":
            workers=os.environ.get("PYTEST_XDIST_AUTO_NUM_WORKERS") or "auto"
            cmd_parts += ["-n", workers]
        extra = shlex.split(os.environ.get("PYTEST_ADDOPTS","")) if os.environ.get("PYTEST_ADDOPTS") else []
        cmd_parts += extra
        rc = _run(" ".join(cmd_parts), env=env)
        return rc
    finally:
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


if __name__ == "__main__":
    sys.exit(main())
