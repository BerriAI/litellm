import json
import threading
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final

import pytest
from pydantic import ValidationError

import litellm
from litellm.llms.custom_httpx.upstream_proxy import (
    UPSTREAM_PROXY_REGISTRY,
    NamedUpstreamProxy,
    current_upstream_proxy,
    upstream_proxy_scope,
)
from litellm.types.router import LiteLLM_Params
from litellm.types.utils import CredentialItem

_COMPLETION: Final = json.dumps(
    {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-test",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
).encode()


@dataclass
class _Recorder:
    requests: list[Mapping[str, object]] = field(default_factory=list)  # mutable-ok: collects served requests

    def record(self, handler: BaseHTTPRequestHandler, body: bytes) -> None:
        self.requests.append(
            {
                "path": handler.path,
                "body": json.loads(body) if body else {},
                "proxy_authorization": handler.headers.get("proxy-authorization"),
            }
        )


def _handler_for(recorder: _Recorder) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length: Final = int(self.headers.get("content-length") or 0)
            recorder.record(self, self.rfile.read(length))
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(_COMPLETION)))
            self.end_headers()
            self.wfile.write(_COMPLETION)

        def log_message(self, format: str, *args: object) -> None:  # matches the stdlib signature
            return None

    return Handler


@contextmanager
def _serving() -> Generator[tuple[str, _Recorder]]:
    recorder: Final = _Recorder()
    server: Final = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(recorder))
    thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", recorder
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture(autouse=True)
def _no_environment_proxy(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]  # autouse fixture
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("DISABLE_AIOHTTP_TRANSPORT", raising=False)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", False)
    UPSTREAM_PROXY_REGISTRY.replace(())


def _complete(api_base: str, **params: object) -> None:
    litellm.completion(
        model="openai/gpt-test",
        api_base=f"{api_base}/v1",
        api_key="sk-test",
        messages=[{"role": "user", "content": "hi"}],
        max_retries=0,
        **params,
    )


async def _acomplete(api_base: str, model: str = "openai/gpt-test", **params: object) -> None:
    await litellm.acompletion(
        model=model,
        api_base=f"{api_base}/v1",
        api_key="sk-test",
        messages=[{"role": "user", "content": "hi"}],
        max_retries=0,
        **params,
    )


@pytest.mark.asyncio
async def test_model_upstream_proxy_replaces_the_environment_proxy_for_that_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _serving() as (environment_proxy, environment_hits), _serving() as (model_proxy, model_hits):
        monkeypatch.setenv("HTTP_PROXY", environment_proxy)
        monkeypatch.setenv("HTTPS_PROXY", environment_proxy)

        await _acomplete("http://upstream.invalid", upstream_proxy=model_proxy)

    assert [request["path"] for request in model_hits.requests] == ["http://upstream.invalid/v1/chat/completions"]
    assert environment_hits.requests == []


@pytest.mark.asyncio
async def test_a_provider_with_its_own_aiohttp_send_path_uses_the_deployment_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """aiohttp_openai posts from its own ClientSession instead of the aiohttp transport,
    so the transport-level hook never sees the request."""
    with _serving() as (environment_proxy, environment_hits), _serving() as (model_proxy, model_hits):
        monkeypatch.setenv("HTTP_PROXY", environment_proxy)
        monkeypatch.setenv("HTTPS_PROXY", environment_proxy)

        await _acomplete("http://upstream.invalid", model="aiohttp_openai/gpt-test", upstream_proxy=model_proxy)

    assert [request["path"] for request in model_hits.requests] == ["http://upstream.invalid/v1/chat/completions"]
    assert environment_hits.requests == []
    assert all("upstream_proxy" not in str(request["body"]) for request in model_hits.requests)


@pytest.mark.asyncio
async def test_leaving_a_deployment_without_upstream_proxy_falls_back_to_the_environment_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _serving() as (environment_proxy, environment_hits), _serving() as (model_proxy, model_hits):
        monkeypatch.setenv("HTTP_PROXY", environment_proxy)
        monkeypatch.setenv("HTTPS_PROXY", environment_proxy)

        await _acomplete("http://upstream.invalid", upstream_proxy=model_proxy)
        await _acomplete("http://upstream.invalid")

    assert len(model_hits.requests) == 1
    assert len(environment_hits.requests) == 1
    assert current_upstream_proxy() is None


@pytest.mark.asyncio
async def test_credential_upstream_proxy_applies_and_the_model_overrides_it(monkeypatch: pytest.MonkeyPatch) -> None:
    credential: Final = CredentialItem(
        credential_name="jp-account",
        credential_values={"api_key": "sk-test", "upstream_proxy": "http://credential.invalid:3128"},
        credential_info={},
    )
    monkeypatch.setattr(litellm, "credential_list", [credential])

    with _serving() as (credential_proxy, credential_hits), _serving() as (model_proxy, model_hits):
        monkeypatch.setattr(
            litellm,
            "credential_list",
            [
                credential.model_copy(
                    update={"credential_values": {"api_key": "sk-test", "upstream_proxy": credential_proxy}}
                )
            ],
        )
        await _acomplete("http://upstream.invalid", litellm_credential_name="jp-account")
        await _acomplete("http://upstream.invalid", litellm_credential_name="jp-account", upstream_proxy=model_proxy)

    assert len(credential_hits.requests) == 1
    assert len(model_hits.requests) == 1


@pytest.mark.asyncio
async def test_a_proxy_referenced_by_name_is_used_with_its_credentials() -> None:
    with _serving() as (proxy_url, proxy_hits):
        port: Final = int(proxy_url.rsplit(":", 1)[1])
        UPSTREAM_PROXY_REGISTRY.replace(
            (
                NamedUpstreamProxy(
                    name="jp", protocol="http", host="127.0.0.1", port=port, username="u", password="p@ss"
                ),
            )
        )

        await _acomplete("http://upstream.invalid", upstream_proxy="jp")

    assert [request["proxy_authorization"] for request in proxy_hits.requests] == ["Basic dTpwQHNz"]


@pytest.mark.asyncio
async def test_a_name_with_no_configured_proxy_fails_the_call_instead_of_going_out_unproxied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _serving() as (environment_proxy, environment_hits):
        monkeypatch.setenv("HTTP_PROXY", environment_proxy)

        with pytest.raises(Exception, match="names no proxy configured"):
            await _acomplete("http://upstream.invalid", upstream_proxy="missing")

    assert environment_hits.requests == []


def test_sync_call_with_upstream_proxy_fails_instead_of_silently_using_the_environment() -> None:
    with pytest.raises(ValueError, match="upstream_proxy only applies to async calls"):
        _complete("http://upstream.invalid", upstream_proxy="http://10.80.0.64:3128")


def test_nested_scope_without_a_proxy_keeps_the_enclosing_one() -> None:
    with upstream_proxy_scope("http://jp.invalid:3128", lambda: True):
        with upstream_proxy_scope(None, lambda: True):
            assert current_upstream_proxy() == "http://jp.invalid:3128"
        with upstream_proxy_scope("http://eu.invalid:3128", lambda: True):
            assert current_upstream_proxy() == "http://eu.invalid:3128"
        assert current_upstream_proxy() == "http://jp.invalid:3128"
    assert current_upstream_proxy() is None


@pytest.mark.parametrize(
    "configured",
    ("socks5://10.80.0.64:1080", "http://", "10.80.0.64:3128", "http://user:secret@10.80.0.64:3128", "jp proxy"),
)
def test_deployment_rejects_proxy_urls_the_transport_cannot_use(configured: str) -> None:
    with pytest.raises(ValidationError, match="upstream_proxy"):
        LiteLLM_Params(model="openai/gpt-test", upstream_proxy=configured)


def test_deployment_accepts_http_proxy_proxy_name_and_environment_reference() -> None:
    assert LiteLLM_Params(model="openai/gpt-test", upstream_proxy="jp").upstream_proxy == "jp"
    assert (
        LiteLLM_Params(model="openai/gpt-test", upstream_proxy="http://10.80.0.64:3128").upstream_proxy
        == "http://10.80.0.64:3128"
    )
    assert (
        LiteLLM_Params(model="openai/gpt-test", upstream_proxy="os.environ/JP_PROXY_URL").upstream_proxy
        == "os.environ/JP_PROXY_URL"
    )
