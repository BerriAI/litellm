"""The native Google AI Studio pass-through route, `/gemini/{endpoint}`, on an owned proxy.

`gemini_proxy_route` (`litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`) lets a Google
GenAI SDK client talk to the proxy as if it were `generativelanguage.googleapis.com`. Its contract is the
mirror image of the `gemini/` chat route:

The CALLER authenticates the way the Google SDK does, with its LiteLLM virtual key in `?key=` or in the
`x-goog-api-key` header (the route reads `request.query_params["key"]` first, then the header, and
runs normal key auth on it). The proxy then forwards to `GEMINI_API_BASE` (default
`https://generativelanguage.googleapis.com`) plus the same path, and puts the REAL AI Studio key in the
upstream `?key=` query parameter (`merged_params.update({"key": gemini_api_key})`), replacing the
caller's. So on this route the upstream key travels as a query parameter, while on `/v1/chat/completions`
it only ever travels as `x-goog-api-key`. The key comes from a configured `gemini/` deployment, or
from `GEMINI_API_KEY`. A path containing `stream` is relayed as a stream. The pass-through logging
handler prices `generateContent` answers from `usageMetadata` with the model taken from the URL and
writes a `pass_through_endpoint` spend row.

This file boots one proxy for the module with `GEMINI_API_BASE` pointed at an owned peer and a synthetic
`GEMINI_API_KEY`, the same pattern as `test_gemini_passthrough_batch_wire.py`. The Rust front door must
keep both the caller key spellings and the key-in-query upstream form, or every Google SDK user breaks.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers.dec1_gemini_support import (
    FLASH,
    body_of,
    gemini_json,
    google_error,
    headers_of,
    marker,
    path_of,
    usage_metadata,
)
from pydantic import JsonValue, TypeAdapter

from litellm.proxy.utils import hash_token

pytestmark = pytest.mark.timeout(300)

_PROVIDER_KEY: Final = "dec1-passthrough-provider-key"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_ERROR_MODEL: Final = "gemini-dec1-error"
_SSE_FRAMES: Final = (
    {"candidates": [{"content": {"role": "model", "parts": [{"text": "hel"}]}, "index": 0}]},
    {
        "candidates": [{"content": {"role": "model", "parts": [{"text": "lo"}]}, "index": 0, "finishReason": "STOP"}],
        "usageMetadata": usage_metadata(11, 7),
    },
)


def _respond(request: Request) -> Reply:
    path: Final = path_of(request)
    if f"/models/{_ERROR_MODEL}:" in path:
        return google_error(429, "RESOURCE_EXHAUSTED", "Quota exceeded for quota metric dec1")
    if path.endswith(":streamGenerateContent"):
        return Reply(
            content_type="text/event-stream",
            chunks=tuple(f"data: {json.dumps(frame)}\r\n\r\n".encode() for frame in _SSE_FRAMES),
        )
    if path.endswith(":generateContent"):
        return Reply(body=json.dumps(gemini_json(text="native pong")).encode())
    if request.method == "GET" and path == "/v1beta/models":
        return Reply(body=json.dumps({"models": [{"name": f"models/{FLASH}"}]}).encode())
    return Reply(body=json.dumps(gemini_json()).encode())


@pytest.fixture(scope="module")
def rig() -> Iterator[Gateway]:
    with gateway_from_environment() as gateway:
        yield gateway


@pytest.fixture(scope="module")
def wire() -> Iterator[Wire]:
    with wire_server(_respond) as served:
        yield served


def _config(directory: Path) -> Path:
    path: Final = directory / "dec1-gemini-passthrough.yaml"
    path.write_text(
        "model_list:\n"
        f"  - model_name: {FLASH}\n"
        "    litellm_params:\n"
        f"      model: gemini/{FLASH}\n"
        "      api_key: os.environ/GEMINI_API_KEY\n"
        "general_settings:\n"
        "  master_key: os.environ/LITELLM_MASTER_KEY\n"
        "  database_url: os.environ/DATABASE_URL\n"
        "router_settings:\n"
        "  disable_cooldowns: true\n"
    )
    return path


@pytest.fixture(scope="module")
def proxy(rig: Gateway, wire: Wire, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("dec1-gemini-passthrough")
    overrides: Final = {"GEMINI_API_BASE": wire.url, "GEMINI_API_KEY": _PROVIDER_KEY}
    with owned_proxy(
        rig, directory, overrides, config=_config(directory), remove_environment=("GOOGLE_API_KEY",)
    ) as owned:
        yield owned


def _base(proxy: Gateway) -> str:
    return str(proxy.client.base_url).rstrip("/")


def _generate_body() -> dict[str, JsonValue]:
    return {"contents": [{"role": "user", "parts": [{"text": f"native {marker()}"}]}]}


def _post(
    proxy: Gateway, path: str, *, key: str, via: str = "query", body: dict[str, JsonValue] | None = None
) -> httpx.Response:
    params: Final = {"key": key} if via == "query" else {}
    headers: Final = {"x-goog-api-key": key} if via == "header" else {}
    return httpx.post(
        f"{_base(proxy)}/gemini/{path}", params=params, headers=headers, json=body or _generate_body(), timeout=30
    )


def _generate_calls(wire: Wire) -> tuple[Request, ...]:
    return tuple(request for request in wire.drain() if request.method == "POST")


@pytest.mark.parametrize("via", ["query", "header"])
def test_a_caller_key_in_query_or_header_is_accepted(proxy: Gateway, wire: Wire, via: str) -> None:
    """A virtual key sent as `?key=<sk>` or as `x-goog-api-key: <sk>` both authenticate
    `POST /gemini/v1beta/models/gemini-2.5-flash:generateContent`, which answers 200 with Google's body."""
    wire.drain()
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        response: Final = _post(proxy, f"v1beta/models/{FLASH}:generateContent", key=key, via=via)
        assert response.status_code == 200, response.text
        assert "native pong" in response.text


def test_the_request_is_forwarded_to_the_same_path_under_gemini_api_base(proxy: Gateway, wire: Wire) -> None:
    """The peer (set as `GEMINI_API_BASE`) receives exactly `POST /v1beta/models/gemini-2.5-flash:generateContent`;
    the `/gemini` prefix is removed and nothing else in the path changes."""
    wire.drain()
    with proxy.scenario() as scenario:
        assert _post(proxy, f"v1beta/models/{FLASH}:generateContent", key=scenario.key()).status_code == 200
        calls: Final = _generate_calls(wire)
        assert [path_of(call) for call in calls] == [f"/v1beta/models/{FLASH}:generateContent"]


def test_the_upstream_key_travels_as_the_key_query_parameter(proxy: Gateway, wire: Wire) -> None:
    """Upstream, the query parameter `key` is the provider key `dec1-passthrough-provider-key`: unlike the
    chat route, the pass-through sends the AI Studio key in the URL, as the Google REST API allows."""
    wire.drain()
    with proxy.scenario() as scenario:
        assert _post(proxy, f"v1beta/models/{FLASH}:generateContent", key=scenario.key()).status_code == 200
        call: Final = _generate_calls(wire)[0]
        assert f"key={_PROVIDER_KEY}" in call.target


@pytest.mark.parametrize("via", ["query", "header"])
def test_the_callers_virtual_key_never_reaches_google(proxy: Gateway, wire: Wire, via: str) -> None:
    """Whichever spelling the caller used, its virtual key is not in the upstream target, header values or
    body; the caller's `?key=` is overwritten rather than appended."""
    wire.drain()
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        assert _post(proxy, f"v1beta/models/{FLASH}:generateContent", key=key, via=via).status_code == 200
        call: Final = _generate_calls(wire)[0]
        assert key not in call.target
        assert key not in call.headers.values()
        assert key.encode() not in call.body
        assert call.target.count("key=") == 1


def test_the_native_body_is_forwarded_unchanged(proxy: Gateway, wire: Wire) -> None:
    """A native body with `contents`, `generationConfig.temperature: 0.2` and `safetySettings` reaches the
    peer with the same JSON, no OpenAI translation and no added keys."""
    wire.drain()
    body: Final[dict[str, JsonValue]] = {
        **_generate_body(),
        "generationConfig": {"temperature": 0.2},
        "safetySettings": [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}],
    }
    with proxy.scenario() as scenario:
        assert _post(proxy, f"v1beta/models/{FLASH}:generateContent", key=scenario.key(), body=body).status_code == 200
        assert body_of(_generate_calls(wire)[0]) == body


def test_a_wrong_key_is_refused_before_google(proxy: Gateway, wire: Wire) -> None:
    """`?key=sk-not-a-real-key` is refused with 401 by key auth and nothing is sent to the peer."""
    wire.drain()
    response: Final = _post(proxy, f"v1beta/models/{FLASH}:generateContent", key="sk-not-a-real-key")
    assert response.status_code == 401, response.text
    assert _generate_calls(wire) == ()


def test_a_request_with_no_key_is_refused(proxy: Gateway, wire: Wire) -> None:
    """With neither `?key=` nor `x-goog-api-key` the route authenticates `Bearer None` and answers 401; it
    must not fall back to the master key or the provider key."""
    wire.drain()
    response: Final = httpx.post(
        f"{_base(proxy)}/gemini/v1beta/models/{FLASH}:generateContent", json=_generate_body(), timeout=30
    )
    assert response.status_code == 401, response.text
    assert _generate_calls(wire) == ()


def test_a_stream_path_is_relayed_as_sse_with_alt_sse_kept(proxy: Gateway, wire: Wire) -> None:
    """`POST /gemini/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse` is relayed as a stream:
    the caller gets `text/event-stream` with Google's two `data:` frames byte for byte, and the upstream
    query keeps `alt=sse` next to the provider `key`."""
    wire.drain()
    with proxy.scenario() as scenario:
        response: Final = httpx.post(
            f"{_base(proxy)}/gemini/v1beta/models/{FLASH}:streamGenerateContent",
            params={"alt": "sse", "key": scenario.key()},
            json=_generate_body(),
            timeout=30,
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.text.count("data: ") == 2
        call: Final = _generate_calls(wire)[0]
        assert "alt=sse" in call.target and f"key={_PROVIDER_KEY}" in call.target


def test_a_google_error_is_relayed_with_its_status_and_body(proxy: Gateway, wire: Wire) -> None:
    """A Google 429 `RESOURCE_EXHAUSTED` is returned to the caller as 429 with Google's own JSON error,
    not an OpenAI style envelope; the pass-through does not map errors."""
    wire.drain()
    with proxy.scenario() as scenario:
        response: Final = _post(proxy, f"v1beta/models/{_ERROR_MODEL}:generateContent", key=scenario.key())
        assert response.status_code == 429, response.text
        error: Final = _JSON_OBJECT.validate_json(response.content)["error"]
        assert isinstance(error, dict) and error["status"] == "RESOURCE_EXHAUSTED"


def test_a_get_list_models_is_passed_through(proxy: Gateway, wire: Wire) -> None:
    """`GET /gemini/v1beta/models?key=<sk>` is forwarded as `GET /v1beta/models` and Google's model list is
    returned as is."""
    wire.drain()
    with proxy.scenario() as scenario:
        response: Final = httpx.get(f"{_base(proxy)}/gemini/v1beta/models", params={"key": scenario.key()}, timeout=30)
        assert response.status_code == 200, response.text
        assert f"models/{FLASH}" in response.text
        assert any(request.method == "GET" and path_of(request) == "/v1beta/models" for request in wire.drain())


def test_a_generate_content_pass_through_writes_a_priced_spend_row(proxy: Gateway, wire: Wire) -> None:
    """One pass-through `generateContent` answered with 11 prompt and 7 candidate tokens writes one SpendLogs
    row for the caller's key with `call_type = "pass_through_endpoint"`, a `model` ending in
    `gemini-2.5-flash` (taken from the URL) and `spend = 11 * 3e-07 + 7 * 2.5e-06`."""
    wire.drain()
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        assert _post(proxy, f"v1beta/models/{FLASH}:generateContent", key=key).status_code == 200
        rows: Final = eventually(
            lambda: proxy.rows(
                'SELECT model, spend, call_type, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" '
                "WHERE api_key = %s",
                (hash_token(key),),
            ),
            lambda found: len(found) >= 1,
            seconds=90,
        )
        assert len(rows) == 1, rows
        row: Final = rows[0]
        assert row["call_type"] == "pass_through_endpoint"
        assert str(row["model"]).endswith(FLASH)
        assert (row["prompt_tokens"], row["completion_tokens"]) == (11, 7)
        assert float(str(row["spend"])) == pytest.approx(11 * 3e-07 + 7 * 2.5e-06)


def test_a_chat_on_a_deployment_without_api_base_uses_gemini_api_base(proxy: Gateway, wire: Wire) -> None:
    """The configured `gemini/gemini-2.5-flash` deployment has no `api_base`, so `/v1/chat/completions`
    goes to `GEMINI_API_BASE` with `x-goog-api-key: dec1-passthrough-provider-key` (from
    `os.environ/GEMINI_API_KEY`), and still never in the query. UNVERIFIED: whether the env base is joined
    as a custom base (`/models/...`, no version) or the default `v1beta` path is kept; the test accepts
    either and pins only the method suffix."""
    wire.drain()
    response: Final = proxy.request(
        "POST", "/v1/chat/completions", {"model": FLASH, "messages": [{"role": "user", "content": marker()}]}
    )
    assert response.status_code == 200, response.text
    calls: Final = _generate_calls(wire)
    assert len(calls) == 1
    assert path_of(calls[0]).endswith(f"models/{FLASH}:generateContent")
    assert headers_of(calls[0]).get("x-goog-api-key") == _PROVIDER_KEY
    assert "key=" not in calls[0].target
