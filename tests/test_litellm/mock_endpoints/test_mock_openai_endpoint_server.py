"""End-to-end smoke tests for the vendored mock OpenAI endpoint.

These tests boot the mock server in-process (via the session-scoped fixture
in ``tests/mock_endpoints/conftest.py``) and verify that the key endpoints
that production tests rely on actually respond correctly.

If this test passes in CI we know the local mock works as a drop-in
replacement for the Railway deployment for chat-completions, embeddings, and
fine-tuning routes — which is the whole point of vendoring it.
"""

from __future__ import annotations

import json
import urllib.request

import pytest

pytest_plugins = ("tests.mock_endpoints.conftest",)


def _post_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer sk-test",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


@pytest.mark.usefixtures("mock_openai_endpoint_server")
def test_should_serve_openai_chat_completion(mock_openai_endpoint_server: str) -> None:
    body = _post_json(
        f"{mock_openai_endpoint_server}/chat/completions",
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert body["object"] == "chat.completion"
    assert body["model"] == "gpt-4o-mini"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"]


def test_should_serve_embeddings(mock_openai_endpoint_server: str) -> None:
    body = _post_json(
        f"{mock_openai_endpoint_server}/v1/embeddings",
        {"model": "text-embedding-3-small", "input": "hello world"},
    )
    assert body["object"] == "list"
    assert body["model"] == "text-embedding-3-small"
    assert len(body["data"][0]["embedding"]) > 0


def test_should_serve_fine_tuning_jobs_list(
    mock_openai_endpoint_server: str,
) -> None:
    body = _get_json(f"{mock_openai_endpoint_server}/openai/fine_tuning/jobs")
    assert body["object"] == "list"
    assert isinstance(body["data"], list)
