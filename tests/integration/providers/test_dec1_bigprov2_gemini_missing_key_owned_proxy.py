"""A Gemini deployment with no key anywhere: the missing key error, on an owned proxy without Google keys.

`gemini/` deployments take their key from the deployment `api_key`, then `GOOGLE_API_KEY`, `GEMINI_API_KEY`
and `PALM_API_KEY` (`litellm/main.py:3937`). When none is set the URL builder refuses before any request with
`ValueError("Missing Gemini API key. Set the GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")`, raised
on the custom `api_base` branch (`litellm/llms/vertex_ai/vertex_llm_base.py:817`) and on the default host branch
(`:882`). The shared proxy has a Gemini key in its environment, so this module boots its own proxy with all
three variables removed (`index/test_dec1_gemini_.md`, "Not covered"). The important contract for a port: the
call is refused without any request to Google (no request with an empty `x-goog-api-key`), and the caller sees
an error naming both variables. UNVERIFIED: the status. A bare `ValueError` with no status is mapped to 500 by
the generic exception path; a port might reasonably answer 401, and that change should be deliberate.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import JSON_OBJECT, gateway_from_environment, object_value
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers.dec1_bigprov2_support import marker

pytestmark: Final = [pytest.mark.timeout(2 * graceful_stop_seconds() + 180)]

_MESSAGE: Final = "Missing Gemini API key. Set the GEMINI_API_KEY or GOOGLE_API_KEY environment variable."


def _answer(request: Request) -> Reply:
    if request.method != "POST":
        return Reply(body=b'{"models": []}')
    return Reply(status=500, body=b'{"error": {"message": "dec1 peer must not be called"}}')


@dataclass(frozen=True, slots=True)
class _Rig:
    proxy: OwnedProxy
    peer: Wire

    def posts(self) -> tuple[Request, ...]:
        return tuple(request for request in self.peer.drain() if request.method == "POST")


@pytest.fixture(scope="module")
def keyless(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("dec1-bigprov2-gemini-keyless")
    with wire_server(_answer) as peer:
        base: Final = JSON_OBJECT.validate_python(
            yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        )
        config: Final = directory / f"dec1-bigprov2-keyless-{uuid.uuid4().hex}.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    **base,
                    "model_list": [
                        {
                            "model_name": "dec1-keyless-with-base",
                            "litellm_params": {
                                "model": "gemini/gemini-2.5-flash",
                                "api_base": f"{peer.url.rstrip('/')}/v1beta",
                            },
                        },
                        {
                            "model_name": "dec1-keyless-keyed",
                            "litellm_params": {
                                "model": "gemini/gemini-2.5-flash",
                                "api_base": f"{peer.url.rstrip('/')}/v1beta",
                                "api_key": "dec1-deployment-key",
                            },
                        },
                    ],
                    "litellm_settings": {**object_value(base["litellm_settings"]), "cache": False},
                    "router_settings": {**object_value(base["router_settings"]), "num_retries": 0},
                }
            )
        )
        with (
            gateway_from_environment() as gateway,
            owned_proxy_process(
                gateway,
                directory,
                {},
                config=config,
                remove_environment=("GEMINI_API_KEY", "GOOGLE_API_KEY", "PALM_API_KEY", "GEMINI_API_BASE"),
                workers=1,
            ) as proxy,
        ):
            yield _Rig(proxy, peer)


def _chat(rig: _Rig, model: str, **fields: object) -> tuple[int, str]:
    response: Final = rig.proxy.gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": marker()}], **fields},  # pyright: ignore[reportArgumentType]
    )
    return response.status_code, response.text


def test_a_keyless_gemini_chat_is_refused_with_the_missing_key_message(keyless: _Rig) -> None:
    """A `gemini/gemini-2.5-flash` deployment with a custom `api_base` and no key, on a proxy with no Google
    key variable. The chat fails, the error text contains the exact missing key message, and the peer receives
    no POST at all."""
    status, text = _chat(keyless, "dec1-keyless-with-base")
    assert status >= 400, text
    assert _MESSAGE in text
    assert keyless.posts() == ()


def test_the_missing_key_refusal_is_a_500_today(keyless: _Rig) -> None:
    """The same refusal is answered with HTTP 500. UNVERIFIED: the `ValueError` carries no status and goes through
    the generic exception mapping; pinned so that a port changing it to 400 or 401 does so on purpose."""
    status, text = _chat(keyless, "dec1-keyless-with-base")
    assert status == 500, text


def test_a_keyless_streamed_gemini_chat_is_refused_before_any_stream(keyless: _Rig) -> None:
    """With `stream: true` the same deployment fails before a stream opens: a JSON error (not `text/event-stream`)
    containing the missing key message, and no request reaches the peer."""
    response: Final = keyless.proxy.gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": "dec1-keyless-with-base", "messages": [{"role": "user", "content": marker()}], "stream": True},
    )
    assert response.status_code >= 400, response.text
    assert response.headers.get("content-type", "").startswith("application/json"), response.headers
    assert _MESSAGE in response.text
    assert keyless.posts() == ()


def test_a_keyless_gemini_embedding_never_sends_an_empty_key(keyless: _Rig) -> None:
    """`/v1/embeddings` on a keyless `gemini/gemini-embedding-001` deployment registered at runtime. The call
    fails and the peer receives no request carrying an empty or `None` `x-goog-api-key`. UNVERIFIED: the
    embedding path's own message; only the absence of an upstream call is asserted."""
    with keyless.proxy.gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="gemini/gemini-embedding-001", api_base=f"{keyless.peer.url.rstrip('/')}/v1beta", api_key=None
        )
        response: Final = keyless.proxy.gateway.request("POST", "/v1/embeddings", {"model": model, "input": marker()})
        assert response.status_code >= 400, response.text
        assert all(request.headers.get("x-goog-api-key") not in ("", "None") for request in keyless.posts())


def test_a_deployment_key_still_serves_on_the_keyless_proxy(keyless: _Rig) -> None:
    """Control: a deployment with its own `api_key` on the same proxy reaches the peer with
    `x-goog-api-key: dec1-deployment-key`, so the refusal above is about the missing key, not the proxy."""
    _chat(keyless, "dec1-keyless-keyed")
    (call,) = keyless.posts()
    assert call.headers.get("x-goog-api-key") == "dec1-deployment-key"
