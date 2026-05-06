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
    LOG.error("bootstrap failed after %d attempts: %s", BOOTSTRAP_MAX_ATTEMPTS, last_err)
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


def _warm_idle(env: Dict[str, str]) -> None:
    """Warm-pool path: idle until B2's hydrate flow lands. Stub just heartbeats."""
    LOG.info("warm-pool mode (stub) — idling. Real implementation lands in B2.")
    while True:
        time.sleep(HEARTBEAT_INTERVAL_SECONDS)


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

    if not env["session_id"] or not env["base_url"] or not env["jwt"]:
        LOG.error(
            "missing required runtime env "
            "(LITELLM_SESSION_ID / LITELLM_BASE_URL / LITELLM_DAEMON_JWT)"
        )
        sys.exit(2)

    if env["mode"] == "warm":
        _warm_idle(env)
        return

    _bootstrap(env)
    _heartbeat_loop(env)


if __name__ == "__main__":
    main()
