"""WebSocket VCR wiring for the realtime suite.

This directory inherits the HTTP VCR machinery from
``tests/llm_translation/conftest.py`` (which only intercepts httpx/aiohttp and
is therefore a no-op for realtime WebSocket traffic). The autouse fixture below
adds the WebSocket layer: it patches ``websockets.connect`` for the duration of
each test so realtime frames are recorded to, or replayed from, the same
cassette Redis under a ``litellm:vcr:wscassette:`` prefix.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from tests._vcr_conftest_common import (  # noqa: E402
    vcr_disabled,
    vcr_outcome_logging_enabled,
)
from tests._ws_vcr import (  # noqa: E402
    WsVcrController,
    build_ws_cassette_client,
    load_ws_cassette,
    replay_timeout_seconds,
    save_ws_cassette,
    ws_redis_key_for,
)

_ws_cassette_client: Optional[object] = None


def _get_ws_cassette_client() -> Optional[object]:
    global _ws_cassette_client
    if _ws_cassette_client is None:
        _ws_cassette_client = build_ws_cassette_client()
    return _ws_cassette_client


def _emit_verdict(request: pytest.FixtureRequest, verdict: str) -> None:
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return
    reporter = request.config.pluginmanager.getplugin("terminalreporter")
    if reporter is None:
        return
    reporter.write_line(f"{verdict} :: {request.node.nodeid}")


@pytest.fixture(autouse=True)
def _ws_vcr(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    if vcr_disabled():
        yield
        return

    import websockets

    client = _get_ws_cassette_client()
    if client is None:
        yield
        return

    key = ws_redis_key_for(request.node.nodeid)
    cassette = load_ws_cassette(client, key)
    controller = WsVcrController(
        original_connect=websockets.connect,
        cassette=cassette,
        timeout=replay_timeout_seconds(),
    )
    monkeypatch.setattr(websockets, "connect", controller.connect)

    yield

    rep_call = getattr(request.node, "rep_call", None)
    passed = bool(rep_call and rep_call.passed)

    if controller.recorded:
        built = controller.built_cassette()
        if built is not None:
            save_ws_cassette(client, key, built, passed=passed)

    if vcr_outcome_logging_enabled():
        _emit_verdict(request, controller.verdict())

    if controller.errors and passed:
        raise controller.errors[0]
