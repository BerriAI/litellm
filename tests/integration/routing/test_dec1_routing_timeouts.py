"""`router_settings.timeout` and `stream_timeout`, and the deployment and request values that override them.

One owned proxy runs with `num_retries: 0`, cooldowns disabled, `timeout: 2` and `stream_timeout: 1`. The
scripted upstream answers after 4 seconds for `slow-*` models and after 1.5 seconds for `mid-*` models.

The contract: a non-streaming call uses the first of the request's own timeout, the deployment's
`litellm_params.timeout` and the router's `timeout`. A streaming call uses the first of the request's stream
timeout, the deployment's `stream_timeout` and the router's `stream_timeout`. A call that exceeds its limit
returns HTTP 408 with a message starting `litellm.Timeout: `. UNVERIFIED for the two router-level values:
nothing else in the suite sets them through `router_settings`, and the precedence is read from
`non_stream_timeout` and `stream_timeout` in `litellm/router_rust/invoke/shaping.py`.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from typing import Final

import pytest
from integration.routing.dec1_routing_support import (
    MID_SECONDS,
    SLOW_SECONDS,
    Rig,
    chat,
    content_of,
    deployment,
    error_message,
    error_of,
    routing_rig,
    stream_chat,
)
from pydantic import JsonValue

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

ROUTER_TIMEOUT_SECONDS: Final = 2
ROUTER_STREAM_TIMEOUT_SECONDS: Final = 1
ROUTER_SETTINGS: Final[dict[str, JsonValue]] = {
    "num_retries": 0,
    "disable_cooldowns": True,
    "timeout": ROUTER_TIMEOUT_SECONDS,
    "stream_timeout": ROUTER_STREAM_TIMEOUT_SECONDS,
}


def _models(api_base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        deployment("timeout-router", "slow-timeout-router", api_base),
        deployment("timeout-deployment-longer", "slow-timeout-longer", api_base, timeout=8),
        deployment("timeout-deployment-shorter", "mid-timeout-shorter", api_base, timeout=1),
        deployment("timeout-header", "slow-timeout-header", api_base),
        deployment("timeout-stream", "mid-timeout-stream", api_base),
        deployment("timeout-stream-deployment", "mid-timeout-stream-own", api_base, stream_timeout=5),
        deployment("timeout-stream-header", "mid-timeout-stream-header", api_base),
    )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with routing_rig(tmp_path_factory.mktemp("dec1-routing-timeouts"), _models, ROUTER_SETTINGS) as value:
        yield value


def test_the_router_timeout_ends_a_slow_call_with_408(rig: Rig) -> None:
    """`router_settings.timeout: 2`; the deployment sets no timeout and its upstream takes 4 seconds.

    The caller gets HTTP 408 with `code` `"408"` and a message starting `litellm.Timeout: `, after at least 2
    seconds and before the upstream's 4 seconds are up. The upstream received exactly one request.
    """
    started: Final = time.monotonic()
    response: Final = chat(rig.gateway, "timeout-router")
    elapsed: Final = time.monotonic() - started
    assert response.status_code == 408, response.text
    assert error_of(response)["code"] == "408"
    assert error_message(response).startswith("litellm.Timeout: "), response.text
    assert ROUTER_TIMEOUT_SECONDS <= elapsed < SLOW_SECONDS, elapsed
    assert rig.upstream.count("slow-timeout-router") == 1


def test_a_longer_deployment_timeout_wins_over_the_router_timeout(rig: Rig) -> None:
    """`router_settings.timeout: 2`; the deployment sets `litellm_params.timeout: 8` and its upstream takes 4
    seconds.

    The caller gets 200 with the deployment's content: the deployment's own timeout replaces the router's.
    """
    response: Final = chat(rig.gateway, "timeout-deployment-longer")
    assert response.status_code == 200, response.text
    assert content_of(response) == "served slow-timeout-longer"


def test_a_shorter_deployment_timeout_wins_over_the_router_timeout(rig: Rig) -> None:
    """`router_settings.timeout: 2`; the deployment sets `litellm_params.timeout: 1` and its upstream takes 1.5
    seconds.

    The router's limit would let the call finish, the deployment's does not: the caller gets HTTP 408 and the
    response header `x-litellm-timeout` reads `1.0`.
    """
    response: Final = chat(rig.gateway, "timeout-deployment-shorter")
    assert response.status_code == 408, response.text
    assert response.headers["x-litellm-timeout"] == "1.0"


def test_the_x_litellm_timeout_request_header_wins_over_the_router_timeout(rig: Rig) -> None:
    """`router_settings.timeout: 2`; the caller sends `x-litellm-timeout: 8` to a deployment whose upstream
    takes 4 seconds.

    The caller gets 200 with the deployment's content.
    """
    response: Final = chat(rig.gateway, "timeout-header", headers={"x-litellm-timeout": "8"})
    assert response.status_code == 200, response.text
    assert content_of(response) == "served slow-timeout-header"


def test_the_router_stream_timeout_applies_to_streaming_calls_only(rig: Rig) -> None:
    """`timeout: 2` and `stream_timeout: 1`; the upstream takes 1.5 seconds to start answering.

    A non-streaming request to the group returns 200, because 1.5 seconds is inside the 2 second timeout. A
    request with `stream: true` returns HTTP 408 with a body containing `litellm.Timeout`, because 1.5 seconds
    is past the 1 second stream timeout.
    """
    plain: Final = chat(rig.gateway, "timeout-stream")
    assert plain.status_code == 200, plain.text
    assert ROUTER_STREAM_TIMEOUT_SECONDS < MID_SECONDS < ROUTER_TIMEOUT_SECONDS
    status, text = stream_chat(rig.gateway, "timeout-stream")
    assert status == 408, text
    assert "litellm.Timeout" in text


def test_a_deployment_stream_timeout_wins_over_the_router_stream_timeout(rig: Rig) -> None:
    """`router_settings.stream_timeout: 1`; the deployment sets `litellm_params.stream_timeout: 5` and its
    upstream takes 1.5 seconds to start answering.

    A request with `stream: true` returns HTTP 200 and a stream containing the deployment's text.
    """
    status, text = stream_chat(rig.gateway, "timeout-stream-deployment")
    assert status == 200, text
    assert "served mid-timeout-stream-own" in text


def test_the_x_litellm_stream_timeout_request_header_wins_over_the_router_stream_timeout(rig: Rig) -> None:
    """`router_settings.stream_timeout: 1`; the caller sends `x-litellm-stream-timeout: 5` with `stream: true`
    to a deployment whose upstream takes 1.5 seconds to start answering.

    The caller gets HTTP 200 and a stream containing the deployment's text.
    """
    body: Final[dict[str, JsonValue]] = {
        "model": "timeout-stream-header",
        "stream": True,
        "messages": [{"role": "user", "content": "dec1 routing stream timeout header"}],
    }
    with rig.gateway.client.stream(
        "POST",
        "/v1/chat/completions",
        json=body,
        headers={"Authorization": f"Bearer {rig.gateway.key}", "x-litellm-stream-timeout": "5"},
    ) as response:
        text: Final = response.read().decode()
        assert response.status_code == 200, text
    assert "served mid-timeout-stream-header" in text
