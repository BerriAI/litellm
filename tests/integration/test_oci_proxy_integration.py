"""
OCI GenAI — end-to-end **proxy** integration tests.

Spins up the LiteLLM proxy (`litellm --config oci_proxy_test_config.yaml`) as a
subprocess, then sends OpenAI-shaped HTTP requests at it for the OCI models
declared in the test config:

  - oci-cohere-command   (oci/cohere.command-latest)
  - oci-llama            (oci/meta.llama-3.3-70b-instruct)
  - oci-gemini           (oci/google.gemini-2.5-flash)
  - oci-grok             (oci/xai.grok-3-mini)
  - oci-embed            (oci/cohere.embed-v4.0)

Skipped unless:
  - ~/.oci/config exists
  - The `oci` SDK is installed (handled by ``pytest.importorskip``)

Environment variables honoured (passed through to the proxy subprocess):
  OCI_CONFIG_PROFILE   profile inside ~/.oci/config (default: DEFAULT)
  OCI_REGION           overrides region from the profile (default: us-chicago-1)

Run with::

    OCI_CONFIG_PROFILE=LUIGI_FRA_API OCI_REGION=us-chicago-1 \
        uv run pytest tests/integration/test_oci_proxy_integration.py -v -s

The tests open a real socket on a free TCP port — no port collision with a
locally-running proxy.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

import httpx
import pytest

# ---------------------------------------------------------------------------
# Skip gate
# ---------------------------------------------------------------------------
OCI_CONFIG_FILE = os.path.expanduser("~/.oci/config")
pytestmark = pytest.mark.skipif(
    not os.path.isfile(OCI_CONFIG_FILE),
    reason="~/.oci/config not found — skipping OCI proxy integration tests",
)


CONFIG_PATH = Path(__file__).parent / "oci_proxy_test_config.yaml"
MASTER_KEY = "sk-1234"
STARTUP_TIMEOUT_S = 90.0
REQUEST_TIMEOUT_S = 120.0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(base_url: str, proc: subprocess.Popen, deadline: float) -> None:
    """Poll /health/liveliness until the proxy answers or the deadline expires."""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            output = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(
                f"litellm proxy exited early with code {proc.returncode}\n--- proxy output ---\n{output}"
            )
        try:
            r = httpx.get(f"{base_url}/health/liveliness", timeout=2.0)
            if r.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise RuntimeError(
        f"litellm proxy did not become ready within {STARTUP_TIMEOUT_S}s"
    )


def _oci_env_from_profile() -> dict[str, str]:
    """Translate the active OCI profile into the OCI_* env vars the litellm
    OCI provider expects. Only API-key profiles are supported; session-token
    profiles would need an in-process signer and so are skipped here.
    """
    oci = pytest.importorskip("oci")
    profile = os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT")
    cfg = oci.config.from_file(profile_name=profile)
    if "security_token_file" in cfg:
        pytest.skip(
            f"OCI profile {profile!r} uses session-token auth; "
            "litellm's OCI provider needs an API-key profile for env-driven config"
        )
    region = os.environ.get("OCI_REGION") or cfg.get("region") or "us-chicago-1"
    return {
        "OCI_USER": cfg["user"],
        "OCI_FINGERPRINT": cfg["fingerprint"],
        "OCI_TENANCY": cfg["tenancy"],
        "OCI_COMPARTMENT_ID": os.environ.get("OCI_COMPARTMENT_ID", cfg["tenancy"]),
        "OCI_KEY_FILE": os.path.expanduser(cfg["key_file"]),
        "OCI_REGION": region,
    }


def _serve(config_path: str) -> Iterator[str]:
    """Boot the litellm proxy with the given config and yield its base URL."""
    env = os.environ.copy()
    env.update(_oci_env_from_profile())
    # Avoid pulling in DB-backed features for this lightweight smoke run.
    env.pop("DATABASE_URL", None)
    env["STORE_MODEL_IN_DB"] = "False"

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"

    # Prefer the `litellm` console script that lives next to the active
    # Python so we inherit the test virtualenv. Fall back to PATH.
    cli = Path(sys.executable).parent / "litellm"
    if not cli.exists():
        cli = "litellm"

    proc = subprocess.Popen(
        [
            str(cli),
            "--config",
            config_path,
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
            "--num_workers",
            "1",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_health(base_url, proc, time.monotonic() + STARTUP_TIMEOUT_S)
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


@pytest.fixture(scope="module")
def proxy_url() -> Iterator[str]:
    yield from _serve(str(CONFIG_PATH))


@pytest.fixture(scope="module")
def proxy_url_no_drop_params(tmp_path_factory) -> Iterator[str]:
    """A proxy WITHOUT drop_params, to prove benign params the proxy injects
    (e.g. max_retries) don't break OCI calls."""
    cfg = tmp_path_factory.mktemp("oci_nodrop") / "config.yaml"
    cfg.write_text(
        "model_list:\n"
        "  - model_name: oci-cohere-command\n"
        "    litellm_params:\n"
        "      model: oci/cohere.command-latest\n"
        "general_settings:\n"
        f"  master_key: {MASTER_KEY}\n"
    )
    yield from _serve(str(cfg))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auth_headers() -> dict:
    return {
        "Authorization": f"Bearer {MASTER_KEY}",
        "Content-Type": "application/json",
    }


def _chat_payload(model: str, *, stream: bool = False) -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "user", "content": "Reply with only the single word: pong"}
        ],
        "max_tokens": 64,
        "stream": stream,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


CHAT_MODELS = ["oci-cohere-command", "oci-llama", "oci-gemini", "oci-grok"]


@pytest.mark.parametrize("model", CHAT_MODELS)
def test_chat_completion_via_proxy(proxy_url: str, model: str) -> None:
    """Non-streaming chat completion returns a well-formed OpenAI response."""
    r = httpx.post(
        f"{proxy_url}/v1/chat/completions",
        headers=_auth_headers(),
        json=_chat_payload(model),
        timeout=REQUEST_TIMEOUT_S,
    )
    assert r.status_code == 200, f"{model} -> {r.status_code}: {r.text}"
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == model
    choices = body["choices"]
    assert len(choices) >= 1
    msg = choices[0]["message"]
    assert msg["role"] == "assistant"
    # Reasoning models may return empty content if their budget covers only
    # the thinking turn — accept either text or a non-empty reasoning field.
    has_content = bool(msg.get("content"))
    has_reasoning = bool(msg.get("reasoning_content")) or bool(msg.get("reasoning"))
    assert has_content or has_reasoning, f"empty assistant message for {model}: {msg}"
    usage = body.get("usage") or {}
    assert usage.get("total_tokens", 0) > 0


@pytest.mark.parametrize("model", CHAT_MODELS)
def test_chat_completion_streaming_via_proxy(proxy_url: str, model: str) -> None:
    """Streaming chat completion yields at least one data: chunk and a [DONE]."""
    saw_chunk = False
    saw_done = False
    with httpx.stream(
        "POST",
        f"{proxy_url}/v1/chat/completions",
        headers=_auth_headers(),
        json=_chat_payload(model, stream=True),
        timeout=REQUEST_TIMEOUT_S,
    ) as r:
        assert r.status_code == 200, f"{model} stream -> {r.status_code}: {r.read()!r}"
        for line in r.iter_lines():
            if not line:
                continue
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                saw_done = True
                break
            saw_chunk = True
    assert saw_chunk, f"no streamed chunks for {model}"
    assert saw_done, f"no [DONE] sentinel for {model}"


def test_embedding_via_proxy(proxy_url: str) -> None:
    """OCI Cohere embedding endpoint returns a non-empty vector via the proxy."""
    r = httpx.post(
        f"{proxy_url}/v1/embeddings",
        headers=_auth_headers(),
        json={"model": "oci-embed", "input": ["hello from the litellm proxy"]},
        timeout=REQUEST_TIMEOUT_S,
    )
    assert r.status_code == 200, f"embed -> {r.status_code}: {r.text}"
    body = r.json()
    assert body["object"] == "list"
    assert body["model"] == "oci-embed"
    data = body["data"]
    assert len(data) == 1
    embedding = data[0]["embedding"]
    assert isinstance(embedding, list)
    assert len(embedding) >= 64
    assert all(isinstance(x, (int, float)) for x in embedding)

def test_model_list_advertises_oci_models(proxy_url: str) -> None:
    """The /v1/models registry advertises every OCI alias from the config."""
    r = httpx.get(
        f"{proxy_url}/v1/models",
        headers=_auth_headers(),
        timeout=REQUEST_TIMEOUT_S,
    )
    assert r.status_code == 200, r.text
    advertised = {row["id"] for row in r.json()["data"]}
    for expected in CHAT_MODELS + ["oci-embed"]:
        assert expected in advertised, f"{expected} missing from /v1/models: {advertised}"


def test_chat_completion_no_drop_params(proxy_url_no_drop_params: str) -> None:
    """A plain chat completion succeeds through a proxy without drop_params.

    Regression for the HTTP 500 ``param `max_retries` is not supported on OCI``:
    the proxy injects max_retries on every request, so without this fix any OCI
    call through the proxy failed unless drop_params was set.
    """
    r = httpx.post(
        f"{proxy_url_no_drop_params}/v1/chat/completions",
        headers=_auth_headers(),
        json=_chat_payload("oci-cohere-command"),
        timeout=REQUEST_TIMEOUT_S,
    )
    assert r.status_code == 200, f"no-drop_params -> {r.status_code}: {r.text}"
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"].get("content") is not None


def test_cohere_default_n_via_proxy(proxy_url: str) -> None:
    """A Cohere request carrying the default n=1 succeeds through the gateway.

    Regression for the HTTP 500 ``param `n` is not supported on OCI`` that
    rejected every client which always sends n=1 (e.g. the MLflow gateway),
    since OCI Cohere has no numGenerations field.
    """
    payload = {**_chat_payload("oci-cohere-command"), "n": 1}
    r = httpx.post(
        f"{proxy_url}/v1/chat/completions",
        headers=_auth_headers(),
        json=payload,
        timeout=REQUEST_TIMEOUT_S,
    )
    assert r.status_code == 200, f"n=1 -> {r.status_code}: {r.text}"
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"].get("content") is not None


@pytest.mark.parametrize("model", ["oci-cohere-command", "oci-llama"])
def test_response_format_json_schema_via_proxy(proxy_url: str, model: str) -> None:
    """A response_format json_schema succeeds through the gateway for both a
    Cohere and a generic OCI model.
    Regression for the HTTP 400 ``Please pass in correct format of request``
    that rejected every json_schema request (which MLflow LLM judges always
    send): generic models choke on OpenAI's ``strict`` key, and Cohere has no
    JSON_SCHEMA type.
    """
    r = httpx.post(
        f"{proxy_url}/v1/chat/completions",
        headers=_auth_headers(),
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Rate the answer 4 to 2+2. Give an integer score and a short rationale.",
                }
            ],
            "max_tokens": 200,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "judgment",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "score": {"type": "integer"},
                            "rationale": {"type": "string"},
                        },
                        "required": ["score", "rationale"],
                        "additionalProperties": False,
                    },
                },
            },
        },
        timeout=REQUEST_TIMEOUT_S,
    )
    assert r.status_code == 200, f"{model} json_schema -> {r.status_code}: {r.text}"
    content = r.json()["choices"][0]["message"]["content"]
    assert content is not None
    assert "score" in json.loads(content)


def test_omitted_max_tokens_not_truncated(proxy_url: str) -> None:
    """A request that omits max_tokens completes instead of being cut off.
    Regression for OCI's tiny server-side maxTokens default (~20 tokens): without
    an injected default, a request that doesn't set max_tokens came back with
    finish_reason "length" after ~19 tokens, so structured outputs (e.g. MLflow
    judge JSON) arrived as unterminated strings. The OCI provider now injects a
    sane default when the caller omits one.
    """
    r = httpx.post(
        f"{proxy_url}/v1/chat/completions",
        headers=_auth_headers(),
        json={
            "model": "oci-cohere-command",
            "messages": [
                {
                    "role": "user",
                    "content": "In four or five complete sentences, explain why the sky appears blue.",
                }
            ],
        },
        timeout=REQUEST_TIMEOUT_S,
    )
    assert r.status_code == 200, f"omitted max_tokens -> {r.status_code}: {r.text}"
    body = r.json()
    choice = body["choices"][0]
    assert (
        choice["finish_reason"] != "length"
    ), f"response truncated by token cap: {choice}"
    assert choice["finish_reason"] == "stop"
    content = choice["message"].get("content") or ""
    assert content.strip(), f"empty content: {choice}"
    # The ~20-token server default truncated well before this; a complete
    # four-to-five sentence answer comfortably exceeds it.
    assert body["usage"]["completion_tokens"] > 50, body["usage"]
