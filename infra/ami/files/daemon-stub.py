#!/usr/bin/env python3.13
"""
litellm-agent-runtime daemon — STUB.

This stub exists so Epic B (VM provisioning) can land a working AMI before
Epic C (the real agent runtime) ships. The real daemon will live in a
sibling repo / package and replace this file in Epic C.

Behaviour:
- Reads runtime config from `/etc/litellm-agent/runtime.env` (loaded into the
  process environment by systemd's EnvironmentFile)
- Two boot modes selected via `LITELLM_AGENT_MODE`:
  * `session` — call /v1/internal/sessions/{sid}/bootstrap, then heartbeat
  * `warm`    — long-poll /v1/internal/warm-pool/{sid}/hydrate (B2)
- Heartbeat every 30s. Exits 0 on session end (when proxy returns 410 Gone).

The stub uses only `requests` (already installed by the AMI builder) and the
standard library so it has zero non-system deps.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

LOG = logging.getLogger("litellm-agent-runtime")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

HEARTBEAT_INTERVAL_SECONDS = 30
BOOTSTRAP_RETRY_INTERVAL_SECONDS = 5
BOOTSTRAP_MAX_ATTEMPTS = 30  # ~150s with 5s backoff

# HTTP status code returned by the proxy when the session is ended.
SESSION_ENDED_STATUS = 410

CONFIG_DIR = Path("/etc/litellm-agent")
REPOS_FILE = CONFIG_DIR / "repos.json"
ENV_FILE = CONFIG_DIR / "env.json"

# Warm-pool hydrate (LIT-2890): the proxy pushes a JSON payload here via
# SSM RunCommand and signals the daemon with SIGUSR1.
HYDRATE_FILE = Path("/var/run/litellm-agent/hydrate.json")
HYDRATE_POLL_INTERVAL_SECONDS = 0.05  # 50ms — tight loop after signal received
HYDRATE_TIMEOUT_SECONDS = 10  # max wait between SIGUSR1 and file appearing


def _redact(value: Optional[str]) -> str:
    if not value:
        return "<unset>"
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def _load_runtime_env() -> Dict[str, str]:
    """Read the systemd EnvironmentFile fields out of os.environ."""
    return {
        "session_id": os.environ.get("LITELLM_SESSION_ID", ""),
        "team_id": os.environ.get("LITELLM_TEAM_ID", ""),
        "agent_id": os.environ.get("LITELLM_AGENT_ID", ""),
        "base_url": os.environ.get("LITELLM_BASE_URL", "").rstrip("/"),
        "mode": os.environ.get("LITELLM_AGENT_MODE", "session"),
        "jwt": os.environ.get("LITELLM_DAEMON_JWT", ""),
    }


def _load_repos() -> list:
    if REPOS_FILE.exists():
        try:
            return json.loads(REPOS_FILE.read_text())
        except (ValueError, OSError) as e:
            LOG.warning("repos.json unreadable: %s", e)
    return []


def _load_env_overrides() -> Dict[str, str]:
    if ENV_FILE.exists():
        try:
            return json.loads(ENV_FILE.read_text())
        except (ValueError, OSError) as e:
            LOG.warning("env.json unreadable: %s", e)
    return {}


def _post_with_jwt(
    url: str, jwt: str, payload: Optional[Dict[str, Any]] = None, timeout: float = 10.0
) -> requests.Response:
    return requests.post(
        url,
        json=payload or {},
        headers={
            "Authorization": f"Bearer {jwt}",
            "Content-Type": "application/json",
        },
        timeout=timeout,
    )


def _bootstrap(env: Dict[str, str]) -> None:
    """Cold-boot path: tell the proxy this VM is ready to receive runs."""
    bootstrap_url = (
        f"{env['base_url']}/v1/internal/sessions/{env['session_id']}/bootstrap"
    )
    payload = {
        "session_id": env["session_id"],
        "team_id": env["team_id"],
        "agent_id": env["agent_id"],
        "repos": _load_repos(),
        "env_keys": sorted(_load_env_overrides().keys()),
    }
    last_err: Optional[str] = None
    for attempt in range(1, BOOTSTRAP_MAX_ATTEMPTS + 1):
        try:
            r = _post_with_jwt(bootstrap_url, env["jwt"], payload, timeout=10.0)
            if r.ok:
                LOG.info("bootstrap ok session=%s", env["session_id"])
                return
            last_err = f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.RequestException as e:
            last_err = f"{type(e).__name__}: {e}"
        LOG.warning(
            "bootstrap attempt %d/%d failed: %s",
            attempt,
            BOOTSTRAP_MAX_ATTEMPTS,
            last_err,
        )
        time.sleep(BOOTSTRAP_RETRY_INTERVAL_SECONDS)
    LOG.error(
        "bootstrap failed after %d attempts: %s", BOOTSTRAP_MAX_ATTEMPTS, last_err
    )
    sys.exit(1)


def _heartbeat_loop(env: Dict[str, str]) -> None:
    """Heartbeat every 30s until the proxy says the session is over."""
    heartbeat_url = (
        f"{env['base_url']}/v1/internal/sessions/{env['session_id']}/heartbeat"
    )
    while True:
        try:
            r = _post_with_jwt(heartbeat_url, env["jwt"], {}, timeout=10.0)
            if r.status_code == SESSION_ENDED_STATUS:
                LOG.info("session ended (HTTP 410); exiting cleanly.")
                return
            if not r.ok:
                LOG.warning("heartbeat HTTP %s: %s", r.status_code, r.text[:200])
        except requests.RequestException as e:
            LOG.warning("heartbeat error: %s", e)
        time.sleep(HEARTBEAT_INTERVAL_SECONDS)


def _wait_for_hydrate() -> Optional[Dict[str, Any]]:
    """Block until SIGUSR1 lands AND the hydrate file appears.

    The proxy's SSM RunCommand script writes the JSON, then signals the
    daemon. Returns the parsed payload, or ``None`` if the file never
    appears within ``HYDRATE_TIMEOUT_SECONDS`` of the signal (retry-able).
    """
    signal_received = {"flag": False}

    def _on_sigusr1(_signum, _frame):  # noqa: ANN001
        signal_received["flag"] = True

    signal.signal(signal.SIGUSR1, _on_sigusr1)

    LOG.info("warm-pool: waiting for SIGUSR1 hydrate signal...")
    while not signal_received["flag"]:
        # `signal.pause` is unavailable on some platforms; sleep is portable.
        time.sleep(1)

    LOG.info("warm-pool: SIGUSR1 received; reading hydrate file...")
    deadline = time.monotonic() + HYDRATE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if HYDRATE_FILE.exists():
            try:
                payload = json.loads(HYDRATE_FILE.read_text())
                LOG.info(
                    "warm-pool: hydrate ok session=%s",
                    payload.get("session_id", "<unknown>"),
                )
                return payload
            except (ValueError, OSError) as e:
                LOG.warning("hydrate file unreadable: %s", e)
                return None
        time.sleep(HYDRATE_POLL_INTERVAL_SECONDS)

    LOG.error("warm-pool: hydrate file never appeared after SIGUSR1")
    return None


def _apply_hydrate(payload: Dict[str, Any]) -> Dict[str, str]:
    """Materialize payload onto disk and return the new runtime env.

    Writes:
      - env_vars  -> /etc/litellm-agent/env       (mode 0644)
      - secrets   -> /etc/litellm-agent/secrets.env (mode 0600 root-only)
      - repos     -> /etc/litellm-agent/repos.json  (overwrite)
    Network policy (iptables) is left to the production daemon — out of scope
    for this stub.
    """
    env_vars = payload.get("env_vars") or {}
    secrets = payload.get("secrets") or {}
    repos = payload.get("repos") or []

    env_path = CONFIG_DIR / "env"
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(
        "".join(f"{k}={v}\n" for k, v in env_vars.items()),
    )
    os.chmod(env_path, 0o644)

    secrets_path = CONFIG_DIR / "secrets.env"
    # Restrict ownership BEFORE we write — chmod after open is racy.
    fd = os.open(secrets_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        for k, v in secrets.items():
            f.write(f"{k}={v}\n")
    os.chmod(secrets_path, 0o600)

    REPOS_FILE.write_text(json.dumps(repos))
    os.chmod(REPOS_FILE, 0o600)

    return {
        "session_id": str(payload.get("session_id", "")),
        "agent_id": str(payload.get("agent_id", "")),
        "team_id": "",
        "base_url": os.environ.get("LITELLM_BASE_URL", "").rstrip("/"),
        "mode": "session",
        "jwt": str(payload.get("jwt", "")),
    }


def _warm_idle(env: Dict[str, str]) -> None:
    """Warm-pool path (LIT-2890): wait for hydrate, then drop into session mode.

    Flow:
      1. Block on SIGUSR1 — signal raised by SSM RunCommand from the proxy
      2. Read /var/run/litellm-agent/hydrate.json (written by the same
         RunCommand script)
      3. Materialize env/secrets/repos onto disk
      4. Switch to session mode and run the normal bootstrap+heartbeat loop
    """
    LOG.info("warm-pool mode: idling for SIGUSR1 hydrate.")

    payload = _wait_for_hydrate()
    if payload is None:
        LOG.error("warm-pool: hydrate failed; exiting so AWS terminates the VM.")
        sys.exit(3)

    new_env = _apply_hydrate(payload)
    if not new_env["jwt"] or not new_env["session_id"]:
        LOG.error("warm-pool: hydrate payload missing jwt or session_id; aborting.")
        sys.exit(4)

    LOG.info(
        "warm-pool: hydrate applied session=%s; entering session mode",
        new_env["session_id"],
    )
    _bootstrap(new_env)
    _heartbeat_loop(new_env)


def _install_signal_handlers() -> None:
    def _shutdown(signum, _frame):
        LOG.info("received signal %s; exiting.", signum)
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _shutdown)


def main() -> None:
    _install_signal_handlers()
    env = _load_runtime_env()
    LOG.info(
        "litellm-agent-runtime starting session=%s team=%s mode=%s base_url=%s jwt=%s",
        env["session_id"],
        env["team_id"],
        env["mode"],
        env["base_url"] or "<unset>",
        _redact(env["jwt"]),
    )

    if env["mode"] == "warm":
        # In warm mode the JWT and session_id arrive at hydrate time, not
        # boot time. Only base_url is required so the daemon can call back.
        if not env["base_url"]:
            LOG.error("missing required runtime env (LITELLM_BASE_URL) in warm mode")
            sys.exit(2)
        _warm_idle(env)
        return

    if not env["session_id"] or not env["base_url"] or not env["jwt"]:
        LOG.error(
            "missing required runtime env "
            "(LITELLM_SESSION_ID / LITELLM_BASE_URL / LITELLM_DAEMON_JWT)"
        )
        sys.exit(2)

    _bootstrap(env)
    _heartbeat_loop(env)


if __name__ == "__main__":
    main()
