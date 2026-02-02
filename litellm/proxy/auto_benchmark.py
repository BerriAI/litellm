# ruff: noqa: T201
"""
Start auto-benchmark from the proxy app lifespan so the server binds and prints
"Uvicorn running on ..." before any benchmark thread runs. This avoids blocking
uvicorn/gunicorn init in the main process.
"""
import os
import subprocess
import sys
import threading
import urllib.parse as urlparse


def _get_mock_chat_endpoint() -> tuple:
    """Return (base_url, port) for mock chat API from LITELLM_MOCK_CHAT_ENDPOINT."""
    endpoint = os.getenv("LITELLM_MOCK_CHAT_ENDPOINT", "http://localhost:8090").strip().rstrip("/")
    if endpoint.isdigit():
        port = int(endpoint)
        return f"http://localhost:{port}", port
    parsed = urlparse.urlparse(endpoint)
    port = parsed.port if parsed.port is not None else 8090
    base = endpoint if parsed.scheme else f"http://localhost:{port}"
    return base, port


def _try_acquire_single_process_lock(lock_path: str) -> bool:
    """Try to acquire an exclusive lock so only one worker starts the benchmark. Returns True if we got it."""
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.close(fd)
        return True
    except FileExistsError:
        return False


def start_auto_benchmark_from_app(host: str, port: int) -> None:
    """
    Called from the proxy app lifespan. Schedules the benchmark to run after
    stabilize_seconds; only one process/worker actually starts it (via file lock).
    Server is already up, so we do not poll for health.
    """
    if os.getenv("LITELLM_AUTO_BENCHMARK", "").strip().lower() not in ("1", "true", "yes"):
        return

    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _litellm_root = os.path.dirname(_this_dir)
    _repo_root = os.path.dirname(_litellm_root)

    rps_suite = os.getenv("LITELLM_AUTO_BENCHMARK_RPS_SUITE", "").strip().lower() in ("1", "true", "yes")
    if rps_suite:
        script_path = os.path.join(_repo_root, "scripts", "run_benchmark_rps_suite.py")
    else:
        script_path = os.path.join(_repo_root, "scripts", "run_benchmark_after_proxy.py")
    if not os.path.isfile(script_path):
        return

    lock_path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "litellm_auto_benchmark.lock")
    if sys.platform == "win32":
        lock_path = os.path.join(os.environ.get("TEMP", os.environ.get("TMP", ".")), "litellm_auto_benchmark.lock")
    if not _try_acquire_single_process_lock(lock_path):
        return

    output = os.getenv("LITELLM_AUTO_BENCHMARK_OUTPUT", "").strip()
    output_dir = os.getenv("LITELLM_AUTO_BENCHMARK_OUTPUT_DIR", "").strip()
    extra_args_str = os.getenv("LITELLM_AUTO_BENCHMARK_ARGS", "--rps-control 10000 --requests 500000").strip()
    extra_args = extra_args_str.split() if extra_args_str else ["--rps-control", "10000", "--requests", "500000"]

    health_host = "127.0.0.1" if host == "0.0.0.0" else host
    proxy_base_url = f"http://{health_host}:{port}"
    mock_base, _ = _get_mock_chat_endpoint()
    provider_url_default = f"{mock_base.rstrip('/')}/chat/completions"
    env = os.environ.copy()
    env["LITELLM_PROXY_URL"] = f"{proxy_base_url}/chat/completions"
    env.setdefault("PROVIDER_URL", provider_url_default)

    popen_kwargs: dict = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        popen_kwargs["start_new_session"] = True

    def _run() -> None:
        import time
        from datetime import datetime

        stabilize_seconds = float(os.getenv("LITELLM_AUTO_BENCHMARK_STABILIZE_SECONDS", "300"))
        if stabilize_seconds > 0:
            time.sleep(stabilize_seconds)

        if rps_suite:
            cmd = [
                sys.executable, script_path, "--no-wait",
                "--proxy-url", env["LITELLM_PROXY_URL"],
                "--provider-url", env["PROVIDER_URL"],
            ]
            if output_dir:
                cmd.extend(["--output-dir", output_dir if os.path.isabs(output_dir) else os.path.join(_repo_root, output_dir)])
            requests_env = os.getenv("LITELLM_AUTO_BENCHMARK_REQUESTS", "").strip()
            if requests_env.isdigit():
                cmd.extend(["--requests", requests_env])
            try:
                print("LiteLLM: Auto-benchmark RPS suite starting in background (1k–10k RPS, 10 runs)", flush=True)
                subprocess.Popen(cmd, env=env, cwd=_repo_root, **popen_kwargs)
            except Exception as e:
                print(f"LiteLLM: Auto-benchmark RPS suite failed: {e}", file=sys.stderr)
        else:
            if not output:
                out_path = os.path.join(_repo_root, f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            else:
                out_path = output if os.path.isabs(output) else os.path.join(_repo_root, output)
            cmd = [
                sys.executable, script_path, "--no-wait", "--output", out_path,
                "--proxy-url", env["LITELLM_PROXY_URL"],
                "--provider-url", env["PROVIDER_URL"],
            ] + extra_args
            try:
                print(f"LiteLLM: Auto-benchmark starting in background, output -> {out_path}", flush=True)
                subprocess.Popen(cmd, env=env, cwd=_repo_root, **popen_kwargs)
            except Exception as e:
                print(f"LiteLLM: Auto-benchmark failed: {e}", file=sys.stderr)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    print("\033[1;32mLiteLLM: Auto-benchmark enabled; will run after stabilize and save output to .txt\033[0m\n")
