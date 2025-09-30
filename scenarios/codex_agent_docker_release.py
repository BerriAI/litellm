#!/usr/bin/env python3
"""Validate codex-agent sidecar exposed via Docker is reachable."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List

import httpx
from dotenv import find_dotenv, load_dotenv
from litellm import Router

load_dotenv(find_dotenv())


def _require_flag(flag: str) -> None:
    if os.getenv(flag) != "1":
        print(f"Set {flag}=1 before running this scenario.")
        sys.exit(1)


def _container_name() -> str:
    value = os.getenv("CODEX_AGENT_DOCKER_CONTAINER", "litellm-codex-agent")
    if not value:
        print(
            "Provide CODEX_AGENT_DOCKER_CONTAINER or ensure the sidecar is reachable via CODEX_AGENT_API_BASE."
        )
        sys.exit(1)
    return value


def _ensure_container_running(container: str) -> None:
    if os.getenv("CODEX_AGENT_API_BASE"):
        return
    try:
        proc = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        print("Docker CLI not found; install Docker or export CODEX_AGENT_API_BASE pointing at a running endpoint.")
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        msg = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        print(msg)
        print(f"Container '{container}' is not accessible. Launch the stack via docker compose.")
        sys.exit(1)

    if proc.stdout.strip().lower() != "true":
        print(f"Container '{container}' is not running (inspect -> {proc.stdout.strip()!r}).")
        sys.exit(1)


def _resolve_api_base() -> str:
    base = os.getenv("CODEX_AGENT_API_BASE")
    if base:
        return base.rstrip("/")
    host = os.getenv("CODEX_AGENT_DOCKER_HOST", "127.0.0.1")
    port = os.getenv("CODEX_AGENT_DOCKER_PORT", "8077")
    return f"http://{host}:{port}".rstrip("/")


def _wait_for_health(api_base: str, timeout: float = 15.0) -> None:
    url = f"{api_base}/healthz"
    deadline = time.time() + timeout
    last_error = None
    with httpx.Client(timeout=2.0) as client:
        while time.time() < deadline:
            try:
                resp = client.get(url)
                if resp.status_code == 200:
                    return
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except Exception as exc:  # noqa: BLE001 - diagnostics only
                last_error = str(exc)
            time.sleep(0.5)
    print(f"codex-agent sidecar at {api_base} failed health check: {last_error}")
    sys.exit(1)


def _router_config(api_base: str) -> Router:
    params: Dict[str, Any] = {
        "model": os.getenv("CODEX_AGENT_MODEL", "codex-agent/mini"),
        "custom_llm_provider": "codex-agent",
        "api_base": api_base,
    }
    api_key = os.getenv("CODEX_AGENT_API_KEY")
    if api_key:
        params["api_key"] = api_key
    return Router(model_list=[{"model_name": "codex-agent-docker", "litellm_params": params}])


PROMPT: List[Dict[str, Any]] = [
    {
        "role": "system",
        "content": (
            "You are a planning agent. Reply with a JSON object containing keys 'plan' and 'final'."
        ),
    },
    {
        "role": "user",
        "content": "Outline two checks we should run against the mini-agent Docker stack and stop.",
    },
]


async def main_async() -> None:
    _require_flag("LITELLM_ENABLE_CODEX_AGENT")
    container = _container_name()
    _ensure_container_running(container)
    api_base = _resolve_api_base()
    _wait_for_health(api_base)

    router = _router_config(api_base)
    start = time.perf_counter()
    response = await router.acompletion(model="codex-agent-docker", messages=PROMPT)
    elapsed = time.perf_counter() - start

    if hasattr(response, "model_dump"):
        payload = response.model_dump()
    else:
        payload = json.loads(json.dumps(response))

    print(
        json.dumps(
            {
                "request": PROMPT,
                "response": payload,
                "api_base": api_base,
                "container": container,
                "elapsed_s": round(elapsed, 2),
            },
            indent=2,
        )
    )

    choices = (payload.get("choices") or []) if isinstance(payload, dict) else []
    content = None
    if choices:
        content = ((choices[0] or {}).get("message") or {}).get("content")
    if not content:
        print("codex-agent docker scenario returned empty content; treat as failure.")
        sys.exit(1)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
