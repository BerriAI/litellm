"""Regression tests for the canned OpenAI mock that decouples the suite from a
hosted endpoint (see ``tests/fake_openai_endpoint.py`` and
``tests/_fake_openai_endpoint_server.py``).

Before this, router/completion/triton tests hardcoded a shared Railway mock as
their ``api_base``; when that single deployment went down, unrelated CI jobs
failed with ``404 Application not found``. These tests pin the two behaviors
those tests now rely on from the local stand-in (the Triton embeddings shape and
the ``slow-endpoint`` delay) and guard against the dead host creeping back in.
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx
import pytest

from tests.fake_openai_endpoint import (
    _LOCAL_DEFAULT,
    _resolve_base,
    ensure_fake_openai_endpoint,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

_MIGRATED_FILES = (
    "tests/llm_translation/test_triton.py",
    "tests/local_testing/test_router.py",
    "tests/local_testing/test_router_custom_routing.py",
    "tests/local_testing/test_router_fallback_handlers.py",
    "tests/local_testing/test_router_fallbacks.py",
    "tests/local_testing/test_secret_detect_hook.py",
    "tests/local_testing/test_lowest_latency_routing.py",
    "tests/local_testing/test_completion.py",
)

# A live ``railway.app/`` or ``railway.app"`` api_base reappearing in a migrated
# file would re-couple CI to an external service's uptime. The deliberately
# broken ``...railway.appzzzzz`` fallback URL is not a live host, so it does not
# match.
_LIVE_HOSTED_MOCK = re.compile(r"railway\.app(?:/|\")")


def test_chat_completion_shape():
    base = ensure_fake_openai_endpoint()
    response = httpx.post(
        f"{base}/v1/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
        timeout=10,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"]
    assert body["usage"]["total_tokens"] == 40


def test_triton_embeddings_route():
    base = ensure_fake_openai_endpoint()
    response = httpx.post(f"{base}/triton/embeddings", json={"inputs": []}, timeout=10)
    assert response.status_code == 200
    output = response.json()["outputs"][0]
    assert output["shape"] == [1, 2]
    assert output["data"] == [0.1, 0.2]


def test_slow_model_blocks_past_client_timeout():
    base = ensure_fake_openai_endpoint()
    with pytest.raises(httpx.TimeoutException):
        httpx.post(
            f"{base}/v1/chat/completions",
            json={
                "model": "slow-endpoint",
                "messages": [{"role": "user", "content": "hi"}],
            },
            timeout=0.5,
        )


def test_remote_env_base_resolves_to_local(monkeypatch):
    monkeypatch.setenv(
        "FAKE_OPENAI_API_BASE",
        "https://exampleopenaiendpoint-production.up.railway.app",
    )
    assert _resolve_base() == _LOCAL_DEFAULT


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "[::1]"])
def test_loopback_env_base_is_honored(monkeypatch, host):
    base = f"http://{host}:9191"
    monkeypatch.setenv("FAKE_OPENAI_API_BASE", base)
    assert _resolve_base() == base


@pytest.mark.parametrize("relative_path", _MIGRATED_FILES)
def test_migrated_files_have_no_live_hosted_mock(relative_path):
    source = (_REPO_ROOT / relative_path).read_text()
    assert not _LIVE_HOSTED_MOCK.search(source), (
        f"{relative_path} references the hosted Railway mock; point api_base at "
        "FAKE_OPENAI_API_BASE so CI does not depend on an external service"
    )
