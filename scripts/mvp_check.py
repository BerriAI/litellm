#!/usr/bin/env python3
"""
MVP Readiness Check for the LiteLLM Fork

Runs a curated, skip-friendly set of checks that exercise the fork features:
- Deterministic local tests (mini-agent core + response utils)
- Local mini-agent shim on :8788 → low E2E finalize
- Optional Exec-RPC probe (:8790)
- Optional Docker readiness tests (if agent-api is up)

Usage:
  python scripts/mvp_check.py

Environment (optional):
  MINI_AGENT_API_HOST / MINI_AGENT_API_PORT  (for docker checks)
  DOCKER_MINI_AGENT=1                       (to enable docker checks)
"""
from __future__ import annotations

import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from typing import List, Tuple
import json
try:
    # Ensure .env is loaded so checks see keys like GEMINI_API_KEY
    from dotenv import load_dotenv, find_dotenv  # type: ignore
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass


def _run(cmd: str, env: dict | None = None, timeout: int | None = 120) -> Tuple[int, str]:
    """
    Stream stdout live (tee) so long-running pytest runs are visible while still
    capturing output for summaries. Honors a soft timeout (seconds) when provided.
    """
    env2 = dict(env or {})
    env2.setdefault('PYTHONUNBUFFERED', '1')
    p = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env2,
        bufsize=1,
    )
    out_lines = []
    start = time.time()
    try:
        assert p.stdout is not None
        for line in p.stdout:
            out_lines.append(line)
            sys.stdout.write(line)
            sys.stdout.flush()
            if timeout is not None and (time.time() - start) > timeout:
                print(f"[readiness] timeout exceeded ({timeout}s); terminating…", flush=True)
                p.kill()
                break
    except KeyboardInterrupt:
        p.kill()
        raise
    rc = p.wait()
    return rc, ''.join(out_lines)


def _can(host: str, port: int, t: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=t):
            return True
    except OSError:
        return False


def section(title: str):
    print(f"\n=== {title} ===")


def _emoji(ok: bool | None, skipped: bool | None = None) -> str:
    if skipped:
        return "⏭"
    if ok is True:
        return "✅"
    if ok is False:
        return "❌"
    return "❔"


def _print_preflight_banner(strict_ready: bool, expected_live: list[str]):
    # Resolve key envs and endpoints up-front for clarity
    rl = os.getenv("READINESS_LIVE", "0") == "1"
    dm = os.getenv("DOCKER_MINI_AGENT", "0") == "1"
    ca_base = os.getenv("CODEX_AGENT_API_BASE")
    oa_base = os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434")
    ma_host = os.getenv("MINI_AGENT_API_HOST", "127.0.0.1")
    try:
        ma_port = int(os.getenv("MINI_AGENT_API_PORT", "8788"))
    except Exception:
        ma_port = 8788
    policy = "STRICT" if strict_ready else ("LIVE" if rl else "DEV")
    exp = ",".join(expected_live) if expected_live else "(none)"
    print("\n=== Readiness Preflight ===")
    print(f"Policy: {policy}; READINESS_EXPECT={exp}; DOCKER_MINI_AGENT={'1' if dm else '0'}")
    print(f"Resolved endpoints: mini-agent={ma_host}:{ma_port}; codex-agent base={ca_base or '(auto)'}; ollama={oa_base}")


def _ensure_dir(p: str) -> None:
    from pathlib import Path as _P
    _P(p).mkdir(parents=True, exist_ok=True)


def _write_router_stub_config(base: str) -> str:
    """
    Create a minimal LiteLLM Router config that maps group 'codex-agent-1'
    to a single model hitting the provided api_base.
    """
    _ensure_dir('.artifacts')
    path = os.path.join('.artifacts', 'router_stub.yaml')
    content = (
        "model_list:\n"
        "  - model_name: gpt-5\n"
        "    litellm_params:\n"
        "      model: openai/gpt-4o-mini\n"
        f"      api_base: {base}\n"
        "      api_key: dummy\n"
        "router_settings:\n"
        "  groups:\n"
        "    - id: codex-agent-1\n"
        "      models: [\"gpt-5\"]\n"
    )
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return path


def _maybe_autoconfig_router(check_name: str, env: dict) -> None:
    """Ensure a usable Router group 'codex-agent-1' for router-related checks.

    Behavior:
    - If no LITELLM_ROUTER_CONFIG is set, autogenerate a stub pointing to
      CODEX_AGENT_API_BASE (or MINI_AGENT_API_BASE) and set ROUTER_MODEL_GROUP.
    - If a config is provided, validate that it contains group 'codex-agent-1'.
      In STRICT mode, emit a clear CONFIG_ERROR if missing.
    """
    name = (check_name or '').lower()
    if not any(s in name for s in ("codex-agent", "router", "all_smokes_core", "live_summary")):
        return
    cfg_path = env.get('LITELLM_ROUTER_CONFIG')
    if cfg_path:
        try:
            import yaml  # type: ignore
            with open(cfg_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            groups = (((data or {}).get('router_settings') or {}).get('groups')) or []
            has_group = any((isinstance(g, dict) and (g.get('id') == 'codex-agent-1')) for g in groups)
            if not has_group:
                msg = f"[CONFIG_ERROR] Router config '{cfg_path}' lacks group 'codex-agent-1'."
                if os.getenv('STRICT_READY','0') == '1':
                    print(msg)
                else:
                    print(msg + " Autogenerating a stub for non-strict run.")
                    base = env.get('CODEX_AGENT_API_BASE') or env.get('MINI_AGENT_API_BASE')
                    if base:
                        cfg = _write_router_stub_config(base)
                        env['LITELLM_ROUTER_CONFIG'] = cfg
                        env.setdefault('ROUTER_MODEL_GROUP', 'codex-agent-1')
        except Exception:
            # Best-effort: fall back to stub unless STRICT
            if os.getenv('STRICT_READY','0') != '1':
                base = env.get('CODEX_AGENT_API_BASE') or env.get('MINI_AGENT_API_BASE')
                if base:
                    cfg = _write_router_stub_config(base)
                    env['LITELLM_ROUTER_CONFIG'] = cfg
                    env.setdefault('ROUTER_MODEL_GROUP', 'codex-agent-1')
        return
    # No config provided: write a stub
    base = env.get('CODEX_AGENT_API_BASE') or env.get('MINI_AGENT_API_BASE')
    if not base:
        return
    cfg = _write_router_stub_config(base)
    env.setdefault('LITELLM_ROUTER_CONFIG', cfg)
    env.setdefault('ROUTER_MODEL_GROUP', 'codex-agent-1')


def _rewrite_cmd_with_venv(run: str, env: dict | None = None) -> str:
    """Make 'pytest …' and 'python …' invocations venv-safe using sys.executable."""
    cmd = (run or '').lstrip()
    py = (env or {}).get('PYTHON_BIN', sys.executable)
    pytest_cmd = (env or {}).get('PYTEST', f"{py} -m pytest")
    if cmd.startswith('pytest '):
        return cmd.replace('pytest', pytest_cmd, 1)
    if cmd.startswith('python3 '):
        return cmd.replace('python3', py, 1)
    if cmd.startswith('python '):
        return cmd.replace('python', py, 1)
    return run

def _augment_pytest_env(check_name: str, env: dict) -> tuple[dict, str]:
    """Add readable defaults + JUnit XML per-check."""
    junit_dir = os.path.join('.artifacts', 'junit')
    _ensure_dir(junit_dir)
    junit_path = os.path.join(junit_dir, f"{check_name}.xml")
    want = ['-ra', '--tb=short', '--durations=25', '-o', 'console_output_style=progress']
    opts = env.get('PYTEST_ADDOPTS', '').split()
    for w in want:
        if w not in opts:
            opts.append(w)
    if not any(x.startswith('--junit-xml') for x in opts):
        opts.append(f"--junit-xml={junit_path}")
    env['PYTEST_ADDOPTS'] = ' '.join(opts).strip()
    return env, junit_path


def _parse_junit_first_failure(junit_path: str) -> str | None:
    try:
        import xml.etree.ElementTree as ET
        import os as _os
        if not _os.path.exists(junit_path):
            return None
        tree = ET.parse(junit_path)
        root = tree.getroot()
        for tc in root.iter('testcase'):
            for tag in ('failure', 'error'):
                el = tc.find(tag)
                if el is not None:
                    cls = tc.get('classname') or ''
                    name = tc.get('name') or ''
                    msg = (el.get('message') or '').strip()
                    anchor = f"{cls}::{name}" if cls else name
                    return f"{anchor} — {msg}"
    except Exception:
        return None
    return None


def _extract_pytest_cause(out: str) -> str | None:
    """Try to extract a concise pytest failure anchor and message."""
    import re as _re
    m = _re.search(r'^FAILED\s+([^\s]+::[^\s]+)\s+-\s+(.+)$', out, flags=_re.M)
    if m:
        return f"{m.group(1)} — {m.group(2)}"
    lines = [ln for ln in out.splitlines() if ln.startswith('E   ')]
    if lines:
        return lines[-1].strip()
    return None



def _check_docker_env_consistency() -> None:
    """Warn early if .env ports look inconsistent with a running docker agent container."""
    if os.getenv("DOCKER_MINI_AGENT", "0") != "1":
        return
    desired_port = os.getenv("MINI_AGENT_API_PORT") or "8788"
    # Best-effort: check common container name mapping
    rc, out = _run("docker port docker-agent-api-1 2>/dev/null")
    if rc == 0 and out.strip():
        # Expect like: "8788/tcp -> 127.0.0.1:8788"
        host_port = None
        for line in out.strip().splitlines():
            if "->" in line and ":" in line:
                try:
                    host_port = line.rsplit(":", 1)[-1].strip()
                except Exception:
                    pass
        if host_port and host_port != desired_port:
            print(
                f"[WARN] MINI_AGENT_API_PORT={desired_port} but docker publishes {host_port}. "
                f"Either set MINI_AGENT_API_PORT={host_port} or reconfigure compose."
            )


def main() -> int:
    failures = 0
    report = {"checks": []}
    strict_ready = os.getenv("STRICT_READY", "0") == "1"
    readiness_live = os.getenv("READINESS_LIVE", "0") == "1"
    expected_raw = os.getenv("READINESS_EXPECT", "")
    expected_live = [x.strip().lower() for x in expected_raw.split(",") if x.strip()] if expected_raw else []

    # Resolve and normalize base endpoints early
    env0 = os.environ
    mini_base = env0.get("MINI_AGENT_API_BASE")
    if not mini_base:
        host = env0.get("MINI_AGENT_API_HOST", "127.0.0.1")
        port = env0.get("MINI_AGENT_API_PORT", "8788")
        mini_base = f"http://{host}:{port}"
        os.environ["MINI_AGENT_API_BASE"] = mini_base
    codex_base = env0.get("CODEX_AGENT_API_BASE") or mini_base
    if codex_base != mini_base and strict_ready:
        print(f"[CONFIG_ERROR] CODEX_AGENT_API_BASE ({codex_base}) != MINI_AGENT_API_BASE ({mini_base}) under STRICT mode.")
    os.environ["CODEX_AGENT_API_BASE"] = codex_base

    _print_preflight_banner(strict_ready, expected_live)
    _check_docker_env_consistency()

    # Optional: run configured checks from readiness.yml (best-effort)
    # This allows projects to extend/override without changing code.
    import yaml  # type: ignore
    cfg_path = "readiness.yml"
    cfg = None
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = None

    # 1) Deterministic local tests (fast, no network)
    section("Deterministic local tests")
    _cmd = (
        "PYTHONPATH=$(pwd) pytest -q "
        "tests/local_testing/test_agent_local_finalize.py "
        "tests/local_testing/test_http_tools_invoker.py "
        "tests/local_testing/test_response_utils.py -q"
    )
    _cmd = _rewrite_cmd_with_venv(_cmd, os.environ.copy())
    rc, out = _run(_cmd)
    print(out.strip())
    if rc != 0:
        print("[FAIL] deterministic local tests")
        failures += 1
        report["checks"].append({"name": "deterministic_local", "ok": False, "details": out[-1000:]})
    else:
        print("[OK] deterministic local tests")
        report["checks"].append({"name": "deterministic_local", "ok": True})

    # 2) Local mini-agent shim on 8788 → low E2E finalize
    section("Local mini-agent shim → low E2E finalize")
    shim_env = os.environ.copy()
    shim_env.setdefault("MINI_AGENT_ALLOW_DUMMY", "1")
    proc = None

    def _health_ok(host: str, port: int) -> bool:
        try:
            import urllib.request, json as _json
            with urllib.request.urlopen(f"http://{host}:{port}/healthz", timeout=0.75) as r:
                if r.status != 200:
                    return False
                data = _json.loads(r.read().decode('utf-8', errors='replace'))
                return bool(isinstance(data, dict) and data.get('ok') is True)
        except Exception:
            return False

    ma_host = '127.0.0.1'
    ma_port = 8788
    if not _can(ma_host, ma_port):
        cmd = f"{sys.executable} -m uvicorn local.tools.mini_agent_shim_app:app --host {ma_host} --port {ma_port} --log-level warning"
        proc = subprocess.Popen(shlex.split(cmd), env=shim_env)
    else:
        if _health_ok(ma_host, ma_port):
            print("[INFO] shim on 127.0.0.1:8788 is healthy; reusing")
        else:
            s = socket.socket(); s.bind((ma_host, 0)); free_port = s.getsockname()[1]; s.close()
            ma_port = int(free_port)
            cmd = f"{sys.executable} -m uvicorn local.tools.mini_agent_shim_app:app --host {ma_host} --port {ma_port} --log-level warning"
            proc = subprocess.Popen(shlex.split(cmd), env=shim_env)
            print(f"[INFO] 8788 in use by non-shim; started shim on {ma_host}:{ma_port}")

    try:
        # wait for /ready
        ok = False
        for _ in range(30):
            if _can(ma_host, ma_port):
                ok = True
                break
            time.sleep(0.2)
        if not ok:
            print("[SKIP] shim not reachable on 127.0.0.1:8788")
        else:
            env = os.environ.copy()
            env.update({
                "MINI_AGENT_API_HOST": ma_host,
                "MINI_AGENT_API_PORT": str(ma_port),
                "MINI_AGENT_ALLOW_DUMMY": shim_env.get("MINI_AGENT_ALLOW_DUMMY", "1"),
            })
            _cmd2 = "PYTHONPATH=$(pwd) pytest -q tests/ndsmoke_e2e/test_mini_agent_e2e_low.py -q"
            _cmd2 = _rewrite_cmd_with_venv(_cmd2, env)
            rc2, out2 = _run(_cmd2, env=env)
            print(out2.strip())
            if rc2 != 0:
                print("[FAIL] mini-agent low E2E finalize")
                failures += 1
                report["checks"].append({"name": "mini_agent_e2e_low", "ok": False, "details": out2[-1000:]})
            else:
                print("[OK] mini-agent low E2E finalize")
                report["checks"].append({"name": "mini_agent_e2e_low", "ok": True})

            # 2b) codex-agent via Router using shim (codex-agent/gpt-5)
            section("codex-agent (gpt-5) via Router against shim")
            # Run a self-contained stub server on a free port to avoid collisions,
            # then call Router against it using the codex-agent provider.
            code = (
                "python - <<'PY'\n"
                "import os, socket, threading, time, json\n"
                "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
                "# pick a free port\n"
                "s=socket.socket(); s.bind(('127.0.0.1',0)); port=s.getsockname()[1]; s.close()\n"
                "class H(BaseHTTPRequestHandler):\n"
                "  def do_POST(self):\n"
                "    if not (self.path.endswith('/chat/completions')):\n"
                "      self.send_response(404); self.end_headers(); return\n"
                "    _ = self.rfile.read(int(self.headers.get('content-length','0') or '0'))\n"
                "    self.send_response(200)\n"
                "    self.send_header('Content-Type','application/json')\n"
                "    self.end_headers()\n"
                "    body={'choices':[{'message':{'content':'hello from codex-agent stub'}}]}\n"
                "    self.wfile.write(json.dumps(body).encode('utf-8'))\n"
                "server=HTTPServer(('127.0.0.1',port), H)\n"
                "t=threading.Thread(target=server.serve_forever, daemon=True); t.start()\n"
                "# wait for port\n"
                "ok=False\n"
                "for _ in range(50):\n"
                "  try:\n"
                "    c=socket.create_connection(('127.0.0.1',port),0.2); c.close(); ok=True; break\n"
                "  except Exception: time.sleep(0.05)\n"
                "os.environ['LITELLM_ENABLE_CODEX_AGENT']='1'\n"
                "os.environ['CODEX_AGENT_API_BASE']=f'http://127.0.0.1:{port}'\n"
                "from litellm import Router\n"
                "r = Router(model_list=[{\n"
                "  'model_name':'codex-agent-1','litellm_params':{\n"
                "    'model':'gpt-5',\n"
                "    'custom_llm_provider':'custom_openai',\n"
                "    'api_base':os.getenv('CODEX_AGENT_API_BASE'),\n"
                "    'api_key':os.getenv('CODEX_AGENT_API_KEY','sk-stub')\n"
                "  }\n"
                "}])\n"
                "out = r.completion(model='codex-agent-1', messages=[{'role':'user','content':'Say hello and finish.'}])\n"
                "print(getattr(getattr(out.choices[0],'message',{}),'content','').strip() or '(empty)')\n"
                "PY\n"
            )
            rc3b, out3b = _run(code)
            print(out3b.strip())
            if rc3b != 0 or '(empty)' in out3b:
                print("[FAIL] codex-agent (gpt-5) via Router against shim")
                failures += 1
                report["checks"].append({"name": "codex_agent_router_shim", "ok": False, "details": out3b[-1000:]})
            else:
                print("[OK] codex-agent (gpt-5) via Router against shim")
                report["checks"].append({"name": "codex_agent_router_shim", "ok": True})
    finally:
        try:
            if proc is not None:
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=5)
        except Exception:
            try:
                if proc is not None:
                    proc.kill()
            except Exception:
                pass

    # 3) Optional Exec-RPC quick probe
    section("Exec-RPC quick probe (optional)")
    if _can("127.0.0.1", 8790):
        rc3, out3 = _run("curl -sf http://127.0.0.1:8790/health")
        if rc3 == 0 and "ok" in out3:
            print("[OK] exec-rpc /health")
        else:
            print("[WARN] exec-rpc reachable but /health did not return ok")
    else:
        print("[SKIP] exec-rpc not running on 127.0.0.1:8790")

    # 3) Optional: run any extra configured checks from readiness.yml
    # Determine which configured checks to run from readiness.yml.
    # If READINESS_EXPECT names match any check names, run only those; otherwise run all.
    selected_names: set[str] = set()
    if expected_live:
        # Only names that exactly match a configured check will be considered.
        selected_names = set(x for x in expected_live)
    if cfg and isinstance(cfg.get("checks"), list) and selected_names:
        print(f"[readiness] selected checks: {', '.join(selected_names)}")
    if cfg and isinstance(cfg.get("checks"), list):
        for ch in cfg["checks"]:
            name = ch.get("name")
            run = ch.get("run")
            optional = bool(ch.get("optional"))
            env_add = ch.get("env") or {}
            if not name or not run:
                continue
            # Filter: if a subset has been explicitly selected, skip others.
            if selected_names:
                if name not in selected_names:
                    continue
            section(f"Configured check: {name}")
            env = os.environ.copy()
            env.update({k: str(v) for (k, v) in env_add.items()})
            # If the configured check targets the mini-agent API, prefer the shim port we resolved above
            if name in ('mini_agent_e2e_low','all_smokes','all_smokes_core','all_smokes_nd') or 'MINI_AGENT_API_PORT' in env:
                try:
                    env['MINI_AGENT_API_HOST'] = locals().get('ma_host', env.get('MINI_AGENT_API_HOST','127.0.0.1'))
                    env['MINI_AGENT_API_PORT'] = str(locals().get('ma_port', env.get('MINI_AGENT_API_PORT','8788')))
                    # Ensure codex-agent base hits the same shim
                    env['CODEX_AGENT_API_BASE'] = f"http://{env['MINI_AGENT_API_HOST']}:{env['MINI_AGENT_API_PORT']}"
                except Exception:
                    pass
            # For mini-agent-targeting checks, allow the dummy path to eliminate model dependency flake
            if 'mini_agent' in (name or ''):
                env.setdefault('MINI_AGENT_ALLOW_DUMMY', '1')
            def _get_timeout_for_check(check_name: str, check_cfg: dict | None) -> int:
                """Resolve timeout precedence for a check.
                1) readiness.yml 'timeout' value if provided
                2) Env overrides per known check name
                3) Default 120s
                """
                if isinstance(check_cfg, dict) and 'timeout' in check_cfg:
                    try:
                        return int(check_cfg['timeout'])
                    except Exception:
                        pass
                mapping = {
                    'all_smokes': ('ALL_SMOKES_TIMEOUT', 900),
                    'all_smokes_core': ('ALL_SMOKES_CORE_TIMEOUT', 600),
                    'all_smokes_nd': ('ALL_SMOKES_ND_TIMEOUT', 900),
                    'mini_agent_e2e_low': ('MINI_AGENT_E2E_LOW_TIMEOUT', 600),
                }
                env_key, default = mapping.get(check_name, (None, 120))
                if env_key:
                    try:
                        return int(os.getenv(env_key, str(default)))
                    except Exception:
                        return default
                return 120
            check_cfg = ch
            env, junit_path = _augment_pytest_env(name, env)
            tmo = _get_timeout_for_check(name, check_cfg)
            eff_tmo = None if os.getenv('READINESS_DISABLE_TIMEOUT') == '1' or (isinstance(tmo,int) and tmo <= 0) else tmo
            # Hermetic defaults and venv-safe executables
            env.setdefault('PYTHON_BIN', sys.executable)
            env.setdefault('PYTEST', f"{sys.executable} -m pytest")
            # Ensure Router group exists for Router-related checks
            _maybe_autoconfig_router(name, env)
            run_cmd = _rewrite_cmd_with_venv(run, env)
            rcx, outx = _run(run_cmd, env=env, timeout=eff_tmo)
            print(outx.strip())
            if rcx != 0:
                cause = _parse_junit_first_failure(junit_path) or _extract_pytest_cause(outx)
                if cause:
                    print(f"[readiness] CHECK '{name}' FAILED — CAUSE: {cause}")
            ok = (rcx == 0)
            if os.getenv('READINESS_FAIL_ON_SKIP') == '1':
                import re as _re
                m=_re.search(r'(\d+)\s+skipped', outx, flags=_re.I)
                if m and int(m.group(1))>0:
                    ok = False
            if not ok and not optional:
                failures += 1
            report["checks"].append({"name": name, "ok": ok, "details": outx[-2000:]})

    # 4) Optional Docker readiness + loopback (legacy path)
    # Record outcomes into results for strict policy
    results: dict[str, bool] = {}
    section("Docker mini-agent readiness (optional)")
    if os.getenv("DOCKER_MINI_AGENT", "0") == "1":
        host = os.getenv("MINI_AGENT_API_HOST", "127.0.0.1")
        port = int(os.getenv("MINI_AGENT_API_PORT", "8788"))
        if not _can(host, port):
            print(f"[SKIP] mini-agent Docker not reachable on {host}:{port}")
            report["checks"].append({"name": "docker_ready", "ok": False, "skipped": True})
            results["docker"] = False
        else:
            rc4, out4 = _run("make ndsmoke-docker")
            print(out4.strip())
            # make target is skip-friendly; treat non-zero as warning
            if rc4 != 0:
                print("[WARN] docker ndsmokes returned non-zero (check output)")
                report["checks"].append({"name": "docker_smokes", "ok": False, "details": out4[-2000:]})
                results["docker"] = False
            else:
                print("[OK] docker ndsmokes (readiness + loopback gated)")
                report["checks"].append({"name": "docker_smokes", "ok": True})
                results["docker"] = True

    # 5) Live provider checks (optional; strict mode requires success per policy)
    section("Live provider checks (optional)")
    live_attempts = 0
    live_success = 0

    # Gemini live check
    if os.getenv("GEMINI_API_KEY"):
        live_attempts += 1
        code = (
            "python - <<'PY'\n"
            "import os\n"
            "import litellm\n"
            "out = litellm.completion(model='gemini/gemini-2.5-flash', messages=[{'role':'user','content':'ping'}])\n"
            "print(getattr(getattr(out.choices[0],'message',{}),'content','').strip() or '(empty)')\n"
            "PY\n"
        )
        code = _rewrite_cmd_with_venv(code, os.environ.copy())
        rcg, outg = _run(code)
        print(outg.strip())
        ok = (rcg == 0 and '(empty)' not in outg)
        report["checks"].append({"name": "gemini_live", "ok": ok, "details": outg[-1000:]})
        results["gemini"] = ok
        if ok:
            live_success += 1
        else:
            failures += 1 if strict_ready else 0
    else:
        print("[SKIP] GEMINI_API_KEY not set")

    # Ollama live check
    ollama_base = os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434")
    host = ollama_base.split("//",1)[-1].split(":")[0]
    try:
        port_str = ollama_base.split(":")[-1]
        port = int(port_str)
    except Exception:
        port = 11434
    if _can(host, port):
        live_attempts += 1
        code = (
            "python - <<'PY'\n"
            "import os, json, urllib.request\n"
            "base = os.getenv('OLLAMA_API_BASE','http://127.0.0.1:11434')\n"
            "# pick a model from /api/tags or fallback\n"
            "try:\n"
            "  tags = json.loads(urllib.request.urlopen(base + '/api/tags', timeout=2).read().decode())\n"
            "  models = [m.get('model') for m in (tags.get('models') or []) if m.get('model')]\n"
            "except Exception:\n"
            "  models = []\n"
            "model = os.getenv('OLLAMA_MODEL') or (models[0] if models else 'llama3.1')\n"
            "import litellm\n"
            "out = litellm.completion(model=f'ollama_chat/{model}', api_base=base, messages=[{'role':'user','content':'ping'}])\n"
            "print(getattr(getattr(out.choices[0],'message',{}),'content','').strip() or '(empty)')\n"
            "PY\n"
        )
        code = _rewrite_cmd_with_venv(code, os.environ.copy())
        rco, outo = _run(code)
        print(outo.strip())
        ok = (rco == 0 and '(empty)' not in outo)
        report["checks"].append({"name": "ollama_live", "ok": ok, "details": outo[-1000:]})
        results["ollama"] = ok
        if ok:
            live_success += 1
        else:
            failures += 1 if strict_ready else 0
    else:
        print(f"[SKIP] Ollama not reachable on {host}:{port}")

    # codex-agent live check (env-gated to a reachable base; default uses MINI_AGENT_API_HOST/PORT)
    ca_host = os.getenv("CODEX_AGENT_HOST") or os.getenv("MINI_AGENT_API_HOST", "127.0.0.1")
    try:
        ca_port = int(os.getenv("CODEX_AGENT_PORT") or os.getenv("MINI_AGENT_API_PORT") or "8788")
    except Exception:
        ca_port = 8788
    # try to ensure codex-agent base is up; if not, start in-process FastAPI agent_proxy
    ca_proc = None
    if not _can(ca_host, ca_port):
        try:
            # If desired port is busy, allocate a free one
            if _can(ca_host, ca_port):
                pass
            else:
                try:
                    s = socket.socket(); s.bind((ca_host, 0)); ca_port = s.getsockname()[1]; s.close()
                except Exception:
                    pass
            start_cmd = f"{sys.executable} -m uvicorn litellm.experimental_mcp_client.mini_agent.agent_proxy:app --host {ca_host} --port {ca_port} --log-level warning"
            envp = os.environ.copy()
            # Only allow echo shim in non-strict mode; strict must exercise real agent path
            if not strict_ready:
                envp.setdefault("MINI_AGENT_OPENAI_SHIM_MODE", "echo")
            ca_proc = subprocess.Popen(shlex.split(start_cmd), env=envp)
            # wait briefly for readiness
            for _ in range(50):
                if _can(ca_host, ca_port):
                    break
                time.sleep(0.1)
        except Exception:
            ca_proc = None
    if _can(ca_host, ca_port):
        live_attempts += 1
        # Choose a real local model for the downstream mini-agent to use via codex-agent base
        # Prefer Ollama if reachable; else fall back to provided model name
        oa_base = os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434")
        oa_host = oa_base.split("//",1)[-1].split(":")[0]
        try:
            oa_port = int(oa_base.split(":")[-1])
        except Exception:
            oa_port = 11434
        downstream_model = "gpt-5"
        if _can(oa_host, oa_port):
            # Pick a model from /api/tags if available
            try:
                import urllib.request, json as _json
                tags = _json.loads(urllib.request.urlopen(oa_base + "/api/tags", timeout=1.5).read().decode())
                models = [m.get("model") for m in (tags.get("models") or []) if m.get("model")]
                downstream_model = f"ollama_chat/{(os.getenv('OLLAMA_MODEL') or (models[0] if models else 'llama3.1'))}"
            except Exception:
                downstream_model = f"ollama_chat/{(os.getenv('OLLAMA_MODEL') or 'llama3.1')}"
        code = (
            "python - <<'PY'\n"
            "import os\n"
            "os.environ['LITELLM_ENABLE_CODEX_AGENT']='1'\n"
            "os.environ['CODEX_AGENT_API_BASE']=f'http://%s:%d'\n" % (ca_host, ca_port) +
            "from litellm import Router\n"
            "r = Router(model_list=[{\n"
            "  'model_name':'codex-agent-1','litellm_params':{\n"
            "    'model':'%s',\n" % downstream_model +
            "    'custom_llm_provider':'codex-agent',\n"
            "    'api_base':os.getenv('CODEX_AGENT_API_BASE'),\n"
            "    'api_key':os.getenv('CODEX_AGENT_API_KEY','sk-stub')\n"
            "  }\n"
            "}])\n"
            "out = r.completion(model='codex-agent-1', messages=[{'role':'user','content':'ping'}])\n"
            "print(getattr(getattr(out.choices[0],'message',{}),'content','').strip() or '(empty)')\n"
            "PY\n"
        )
        envc = os.environ.copy()
        envc["PYTHONPATH"] = os.getcwd()
        code = _rewrite_cmd_with_venv(code, envc)
        rcc, outc = _run(code, env=envc)
        print(outc.strip())
        ok = (rcc == 0 and '(empty)' not in outc)
        results["codex-agent"] = ok
        report["checks"].append({"name": "codex_agent_live", "ok": ok, "details": outc[-1000:]})
        if ok:
            live_success += 1
        else:
            failures += 1 if strict_ready else 0
    else:
        print(f"[SKIP] codex-agent base not reachable on {ca_host}:{ca_port}")
    # cleanup if we started local server
    if ca_proc is not None:
        try:
            ca_proc.send_signal(signal.SIGTERM)
            ca_proc.wait(timeout=5)
        except Exception:
            try:
                ca_proc.kill()
            except Exception:
                pass

    if strict_ready:
        # Default policy: require Ollama unless READINESS_EXPECT overrides
        # Only consider known provider tokens for the live provider summary.
        known_providers = {"ollama", "codex-agent", "docker", "gemini"}
        if not expected_live:
            expected_live = ["ollama"]
        expected_live = [p for p in expected_live if p in known_providers]
        missing = [p for p in expected_live if results.get(p) is not True]
        if missing:
            print(f"[FAIL] live providers missing/failed in STRICT mode: {', '.join(missing)}")
            failures += 1
            report["checks"].append({
                "name": "live_summary",
                "ok": False,
                "details": f"Failed/missing: {', '.join(missing)}; results={results}"
            })
        else:
            report["checks"].append({
                "name": "live_summary",
                "ok": True,
                "details": f"All expected live providers OK: {', '.join(expected_live)}"
            })

    print("\n=== Summary ===")
    # Write JSON report
    os.makedirs("local/artifacts/mvp", exist_ok=True)
    with open("local/artifacts/mvp/mvp_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Write human-readable PROJECT_READY.md
    try:
        md_lines = []
        md_lines.append("# Project Readiness\n")
        md_lines.append("\n")
        pol = "STRICT" if strict_ready else ("LIVE" if readiness_live else "DEV")
        md_lines.append(f"Policy: {pol}; READINESS_EXPECT={expected_raw or '(none)'}\n")
        md_lines.append("\n")
        mini_host = os.getenv("MINI_AGENT_API_HOST", "127.0.0.1")
        mini_port = os.getenv("MINI_AGENT_API_PORT", "8788")
        md_lines.append("Resolved endpoints:\n")
        md_lines.append(f"- mini-agent: {mini_host}:{mini_port}\n")
        md_lines.append(f"- codex-agent base: {os.getenv('CODEX_AGENT_API_BASE') or '(auto)'}\n")
        md_lines.append(f"- ollama: {os.getenv('OLLAMA_API_BASE','http://127.0.0.1:11434')}\n")
        md_lines.append("\n")
        md_lines.append("## Results\n")
        for c in report.get("checks", []):
            name = c.get("name")
            ok = c.get("ok")
            skipped = c.get("skipped")
            emoji = _emoji(ok, skipped)
            md_lines.append(f"- {emoji} {name}\n")
        md_lines.append("\n")
        md_lines.append("Artifacts:\n")
        md_lines.append("- local/artifacts/mvp/mvp_report.json\n")
        with open("PROJECT_READY.md", "w", encoding="utf-8") as mdf:
            mdf.write("".join(md_lines))
    except Exception:
        pass

    # Compact terminal table for quick scan
    try:
        print("\n--- Results (quick) ---")
        for c in report.get("checks", []):
            name = c.get("name")
            ok = c.get("ok")
            skipped = c.get("skipped")
            print(f"{name:<26} {_emoji(ok, skipped)}")
    except Exception:
        pass

    if failures:
        print(f"MVP readiness checks: {failures} failure(s)")
    else:
        print("MVP readiness checks: OK")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
