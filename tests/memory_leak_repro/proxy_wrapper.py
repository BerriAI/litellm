"""
Wrapper that imports the litellm proxy app and patches it to simulate the
production memory leak (spend_log_transactions growing without bound).

Run with: uvicorn tests.memory_leak_repro.proxy_wrapper:app --workers 2 --port 4001
"""

import asyncio
import os
import resource
import sys
import copy

# Ensure workspace root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Set config before importing the proxy
os.environ.setdefault("LITELLM_MASTER_KEY", "sk-test-master-key")
os.environ.setdefault("LITELLM_LOG", "ERROR")

# CRITICAL: Ensure no DATABASE_URL is set so the proxy starts without Prisma.
# (The leak middleware will simulate the spend_log_transactions accumulation)
if "DATABASE_URL" in os.environ:
    del os.environ["DATABASE_URL"]

# Point proxy to our config file via env var
_config_path = os.environ.get("CONFIG_FILE_PATH", "")
if not _config_path:
    _default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proxy_config.yaml")
    if os.path.exists(_default_config):
        os.environ["CONFIG_FILE_PATH"] = _default_config

# Apply memory cap BEFORE importing anything heavy
_cap_mb = int(os.environ.get("_REPRO_WORKER_MEM_CAP_MB", "0"))
if _cap_mb > 0:
    _cap_bytes = _cap_mb * 1024 * 1024
    try:
        _soft, _hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (_cap_bytes, _hard))
        print(f"[proxy_wrapper] RLIMIT_AS set to {_cap_mb}MB (pid={os.getpid()})")
    except Exception as e:
        print(f"[proxy_wrapper] Failed to set RLIMIT_AS: {e}")


# Now import the proxy app
from litellm.proxy.proxy_server import app  # noqa: E402


# ---------------------------------------------------------------------------
# Monkey-patch: after each request, simulate the spend_log_transactions leak
# by appending a ~3KB payload to a global list that is NEVER drained.
# This is exactly what happens in production when the DB is unreachable.
# ---------------------------------------------------------------------------
_leaked_payloads = []
_leak_lock = asyncio.Lock()

# Approximate size of a real SpendLogsPayload (from production observation)
# Each entry simulates a real SpendLogsPayload dict.
# In production with store_prompts_in_spend_logs=true, messages + response can be
# tens of KB. Even without, the metadata dict has many fields.
# We make each entry ~8KB to match a realistic production payload.
_PADDING = "x" * 4096  # Simulates response/metadata content
_FAKE_SPEND_LOG_ENTRY = {
    "request_id": "req-00000000-0000-0000-0000-000000000000",
    "call_type": "acompletion",
    "api_key": "hashed_sk_test_1234567890abcdef1234567890abcdef",
    "spend": 0.00042,
    "total_tokens": 18,
    "prompt_tokens": 10,
    "completion_tokens": 8,
    "startTime": "2026-02-26T03:47:24.123456+00:00",
    "endTime": "2026-02-26T03:47:24.234567+00:00",
    "completionStartTime": "2026-02-26T03:47:24.200000+00:00",
    "model": "gpt-3.5-turbo",
    "model_id": "model-abc123",
    "model_group": "gpt-3.5-turbo",
    "api_base": "http://127.0.0.1:18080/v1",
    "user": "user-test-123",
    "metadata": '{"user_api_key":"sk-test","user_api_key_user_id":"user-123","user_api_key_team_id":"team-456","additional_usage_values":{},"status":"success","extra_padding":"' + _PADDING + '"}',
    "cache_hit": "False",
    "cache_key": "Cache OFF",
    "request_tags": "[]",
    "team_id": "team-456",
    "end_user": "",
    "requester_ip_address": None,
    "messages": '{"content": "' + _PADDING + '"}',
    "response": '{"content": "' + _PADDING + '"}',
    "proxy_server_request": None,
    "custom_llm_provider": "openai",
}


from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


# Use a simple ASGI middleware instead of BaseHTTPMiddleware to avoid overhead
from starlette.types import ASGIApp, Receive, Scope, Send


class LeakMiddleware:
    """
    ASGI middleware that simulates the spend_log_transactions leak.
    On every HTTP request, appends a deepcopy of a spend log payload to a list
    that is NEVER drained — exactly like production when DB is unreachable.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        await self.app(scope, receive, send)

        if scope["type"] == "http" and scope.get("path", "").startswith("/chat/"):
            # Simulate the spend log leak: deepcopy a payload and append to list.
            # In production, each spend log payload is a dict with metadata, request
            # data, response data, etc. We deepcopy it to match the real code path
            # in _insert_spend_log_to_db (db_spend_update_writer.py:683).
            # Each real payload is roughly 2-5KB in Python memory after deepcopy.
            payload = copy.deepcopy(_FAKE_SPEND_LOG_ENTRY)
            _leaked_payloads.append(payload)

            if len(_leaked_payloads) % 2000 == 0:
                # Estimate leaked memory (each dict payload ~2-3KB in CPython)
                est_mb = len(_leaked_payloads) * 3 / 1024
                print(
                    f"[leak] pid={os.getpid()} spend_log_transactions={len(_leaked_payloads)} "
                    f"(~{est_mb:.0f}MB leaked)"
                )


# Install the leak middleware
app.add_middleware(LeakMiddleware)

print(f"[proxy_wrapper] LeakMiddleware installed (pid={os.getpid()})")
