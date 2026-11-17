"""Requests that take unusual paths through the limiter: no_log, client disconnects, provider stream aborts,
callback opt-outs, pass-through routes and image generation.

The contract:

- ``no_log: true`` turns off logging callbacks but not the limiter. ``Logging.should_run_callback`` keeps every
  callback whose class name contains ``_PROXY_`` (the limiter is ``_PROXY_MaxParallelRequestsHandler_v3``), so a
  no_log request is still counted, still settles its tokens and still releases its slot. The Rust port registers
  the limiter as a CustomLoggerV2 callback (#51537), whose selection can honour no_log
  (``custom_logger_v2/selection.py``): that must stay off for the limiter or no_log becomes a limit bypass.
- A client that disconnects mid-stream gives its ``max_parallel_requests`` slot back
  (``async_release_max_parallel_requests_on_disconnect``), as does a stream the provider aborts.
- ``x-litellm-disable-callbacks`` cannot switch the limiter off.
- Pass-through endpoints run the proxy pre-call hook, so a key's request window counts them; image generation is
  counted like any other generation route.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Final

import httpx
import pytest

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.wire import Reply, Request, wire_server
from tests.integration.authorization.dec1_limits_support import (
    assert_limited,
    chat,
    hashed,
    limits_proxy,
    posts,
    probe,
    provider,
    scope_headers,
)
from tests.integration.authorization.dec1_ratelim_support import (
    chat_with,
    in_flight,
    redis_counters,
    tokens_counter,
)


def test_a_no_log_request_still_uses_the_keys_request_window(gateway: Gateway) -> None:
    """A key with rpm_limit=1 sends a request with ``no_log: true``; it must be served with
    ``x-ratelimit-api_key-remaining-requests: 0`` and the next request (with or without no_log) refused."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=1)
        served: Final = chat(gateway, model, key, no_log=True)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key")["remaining-requests"] == "0"
        assert_limited(chat(gateway, model, key, no_log=True), "api_key", hashed(key), "requests", 1)


def test_a_no_log_request_settles_its_tokens_to_the_reported_usage(gateway: Gateway) -> None:
    """A key with tpm_limit=10000 sends a no_log request reserving 10 + 40 tokens; the provider reports 11 + 7. The
    Redis counter ``{api_key:<hash>}:tokens`` must settle to 18, proving the settlement callback ran despite
    no_log."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario, redis_counters() as redis:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=10000)
        assert chat(gateway, model, key, max_tokens=40, no_log=True).status_code == 200
        assert eventually(lambda: tokens_counter(redis, "api_key", hashed(key)), lambda t: t == 18, seconds=20) == 18


def test_a_no_log_request_gives_its_parallel_slot_back(gateway: Gateway) -> None:
    """A key with max_parallel_requests=1 sends a no_log request and then a second one. Both must be served and the
    sorted set ``{api_key:<hash>}:max_parallel_requests`` must end empty: the release ran for the no_log call."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario, redis_counters() as redis:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], max_parallel_requests=1)
        assert chat(gateway, model, key, no_log=True).status_code == 200
        eventually(lambda: in_flight(redis, "api_key", hashed(key)), lambda members: members == 0)
        assert chat(gateway, model, key, no_log=True).status_code == 200


def test_a_streamed_no_log_request_settles_from_its_final_usage_chunk(gateway: Gateway) -> None:
    """A key with tpm_limit=10000 streams a no_log request whose last chunk reports 11 + 7 tokens. The token
    counter must settle to 18 and the slot (max_parallel_requests=2) must be released."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario, redis_counters() as redis:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=10000, max_parallel_requests=2)
        streamed: Final = chat(
            gateway, model, key, max_tokens=40, no_log=True, stream=True, stream_options={"include_usage": True}
        )
        assert streamed.status_code == 200, streamed.text
        assert eventually(lambda: tokens_counter(redis, "api_key", hashed(key)), lambda t: t == 18, seconds=20) == 18
        eventually(lambda: in_flight(redis, "api_key", hashed(key)), lambda members: members == 0)


def test_a_client_that_disconnects_mid_stream_gives_its_slot_back(gateway: Gateway) -> None:
    """A key with max_parallel_requests=1 opens a stream that the provider holds after its first frame; the client
    reads the first bytes and closes the connection. The gauge must drop to 0 members and a new request must be
    served, without waiting for the slot TTL."""
    gate: Final = threading.Event()
    with wire_server(provider(gate_stream=gate)) as wire, gateway.scenario() as scenario, redis_counters() as redis:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], max_parallel_requests=1)
        body: Final = {"model": model, "messages": [{"role": "user", "content": probe()}], "stream": True}
        with gateway.client.stream(
            "POST", "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"}
        ) as opened:
            assert opened.status_code == 200
            next(opened.iter_bytes())
            assert in_flight(redis, "api_key", hashed(key)) == 1
        try:
            eventually(lambda: in_flight(redis, "api_key", hashed(key)), lambda members: members == 0, seconds=20)
        finally:
            gate.set()
        assert chat(gateway, model, key).status_code == 200


def test_a_stream_the_provider_aborts_gives_its_slot_back(gateway: Gateway) -> None:
    """A key with max_parallel_requests=1 streams from a provider that drops the connection after one frame. Once
    the client has the (failed) stream, the gauge must be empty and the next request served."""

    def aborting(request: Request) -> Reply:
        frame: Final = b'data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"a"}}]}\n\n'
        return Reply(content_type="text/event-stream", chunks=(frame, frame), abort_after=1)

    with wire_server(aborting) as wire, gateway.scenario() as scenario, redis_counters() as redis:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], max_parallel_requests=1)
        chat(gateway, model, key, stream=True)
        eventually(lambda: in_flight(redis, "api_key", hashed(key)), lambda members: members == 0, seconds=20)
        assert len(posts(wire)) == 1


def test_the_disable_callbacks_header_cannot_switch_the_limiter_off(gateway: Gateway) -> None:
    """A key with rpm_limit=1 sends two requests with ``x-litellm-disable-callbacks:
    parallel_request_limiter,_PROXY_MaxParallelRequestsHandler_v3``. The second must still be refused for
    ``api_key``. UNVERIFIED: the dynamic opt-out lives in enterprise callback controls and is assumed not to
    reach proxy hooks."""
    header: Final = {"x-litellm-disable-callbacks": "parallel_request_limiter,_PROXY_MaxParallelRequestsHandler_v3"}
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=1)
        assert chat_with(gateway, model, key, headers=header).status_code == 200
        assert_limited(chat_with(gateway, model, key, headers=header), "api_key", hashed(key), "requests", 1)


def _image(request: Request) -> Reply:
    assert request.target.split("?", 1)[0].endswith("/images/generations"), request.target
    return Reply(body=json.dumps({"created": 1, "data": [{"b64_json": "aW1hZ2U="}]}).encode())


def test_image_generation_uses_the_keys_request_window(gateway: Gateway) -> None:
    """A key with rpm_limit=1 calls ``/v1/images/generations`` twice. The first must be served and the second
    refused for ``api_key`` requests. No spend is asserted (image prices are lost today, #52103)."""
    with wire_server(_image) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1", model="openai/dall-e-3")
        key: Final = scenario.key(models=[model], rpm_limit=1)

        def generate() -> httpx.Response:
            return gateway.request("POST", "/v1/images/generations", {"model": model, "prompt": probe()}, key=key)

        first: Final = generate()
        assert first.status_code == 200, first.text
        assert_limited(generate(), "api_key", hashed(key), "requests", 1)


@pytest.fixture(scope="module")
def passthrough(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory):  # noqa: ANN201
    """A proxy of our own whose config declares an authenticated pass-through route ``/ratelim-passthrough`` to a
    scripted provider, on the shared database and Redis. UNVERIFIED: ``general_settings.pass_through_endpoints``
    entry shape (``path``, ``target``, ``auth``) read from the pass-through docs."""
    directory: Final[Path] = tmp_path_factory.mktemp("ratelim-passthrough")
    with (
        wire_server(provider()) as wire,
        limits_proxy(
            session_gateway,
            directory,
            general_settings={
                "pass_through_endpoints": [
                    {"path": "/ratelim-passthrough", "target": f"{wire.url}/v1/chat/completions", "auth": True}
                ]
            },
            shared=True,
        ) as candidate,
    ):
        yield candidate


def test_a_pass_through_call_uses_the_keys_request_window(gateway: Gateway, passthrough: Gateway) -> None:
    """A key with rpm_limit=1 calls the authenticated pass-through route twice. The first must be forwarded and the
    second refused with 429 for ``api_key``: pass-through runs the same pre-call hook."""
    with gateway.scenario() as scenario:
        key: Final = scenario.key(rpm_limit=1)
        body: Final = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": probe()}]}
        first: Final = passthrough.request("POST", "/ratelim-passthrough", body, key=key)
        assert first.status_code == 200, first.text
        assert_limited(
            passthrough.request("POST", "/ratelim-passthrough", body, key=key), "api_key", hashed(key), "requests", 1
        )


def test_a_pass_through_call_is_held_to_a_per_model_limit(gateway: Gateway, passthrough: Gateway) -> None:
    """A key with ``model_rpm_limit={"gpt-4o-mini": 1}`` calls the pass-through route three times with
    ``model: gpt-4o-mini`` in the body. Only the first is forwarded and the next two get 429: the pass-through route
    runs the pre-call hook on the parsed body (``pass_through_endpoints.py:1619``) and the v3 limiter builds a
    ``model_per_key`` descriptor from the body's ``model`` whether or not the router serves it
    (``parallel_request_limiter_v3.py:2890``, falling back to the requested name at ``:2753``)."""
    with gateway.scenario() as scenario:
        key: Final = scenario.key(model_rpm_limit={"gpt-4o-mini": 1})
        body: Final = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": probe()}]}
        codes: Final = [
            passthrough.request("POST", "/ratelim-passthrough", body, key=key).status_code for _ in range(3)
        ]
        assert codes == [200, 429, 429]
