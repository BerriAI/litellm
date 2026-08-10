"""
Tests for loading the standalone fallback generalizations file: remote wins when it is
usable, the bundled copy covers every failure, and the local-only env flags skip HTTP.
"""

import json
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm.litellm_core_utils.fallback_generalizations import (
    get_fallback_generalization_rules,
    match_routing_generalization,
    set_fallback_generalizations,
)
from litellm.litellm_core_utils.get_fallback_generalizations import (
    get_fallback_generalizations,
    install_fallback_generalizations,
    load_local_fallback_generalizations,
)

_URL = "https://example.invalid/fallback_generalizations.json"

_REMOTE_RULES = [
    {
        "name": "remote-only",
        "pattern": r"^widget-",
        "model_info": {"litellm_provider": "openai"},
    }
]


@pytest.fixture(autouse=True)
def _no_local_only_env(monkeypatch):
    """CI exports LITELLM_LOCAL_MODEL_COST_MAP=True; clear both flags so fetching is deterministic."""
    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
    monkeypatch.delenv("LITELLM_LOCAL_FALLBACK_GENERALIZATIONS", raising=False)


@pytest.fixture
def restore_generalizations():
    previous = list(get_fallback_generalization_rules())
    try:
        yield
    finally:
        set_fallback_generalizations(previous)


def _serve(monkeypatch, response: httpx.Response | Exception) -> dict:
    """Point httpx.get at a canned response and count the calls made to it."""
    calls = {"count": 0}

    def fake_get(url, timeout=None):
        calls["count"] += 1
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(
        "litellm.litellm_core_utils.get_fallback_generalizations.httpx.get", fake_get
    )
    return calls


def test_remote_rules_are_used_when_the_fetch_succeeds(monkeypatch):
    _serve(monkeypatch, httpx.Response(200, json={"rules": _REMOTE_RULES}, request=httpx.Request("GET", _URL)))
    assert get_fallback_generalizations(url=_URL) == _REMOTE_RULES


def test_network_failure_falls_back_to_the_bundled_file(monkeypatch):
    """A rules fetch must never take routing down: an unreachable host keeps the shipped rules."""
    _serve(monkeypatch, httpx.ConnectError("connection refused"))
    assert get_fallback_generalizations(url=_URL) == load_local_fallback_generalizations()


def test_http_error_falls_back_to_the_bundled_file(monkeypatch):
    """The file is 404 on older branches until this lands, so a 404 must not wipe the rules."""
    _serve(monkeypatch, httpx.Response(404, request=httpx.Request("GET", _URL)))
    assert get_fallback_generalizations(url=_URL) == load_local_fallback_generalizations()


@pytest.mark.parametrize(
    "payload",
    [{"rules": {}}, {}, []],
    ids=["rules_not_a_list", "no_rules_key", "not_an_object"],
)
def test_malformed_remote_payload_falls_back_to_the_bundled_file(monkeypatch, payload):
    """A truncated or reshaped remote file must not silently disable every rule."""
    _serve(monkeypatch, httpx.Response(200, json=payload, request=httpx.Request("GET", _URL)))
    assert get_fallback_generalizations(url=_URL) == load_local_fallback_generalizations()


@pytest.mark.parametrize(
    "env_name",
    ["LITELLM_LOCAL_FALLBACK_GENERALIZATIONS", "LITELLM_LOCAL_MODEL_COST_MAP"],
)
def test_local_only_env_skips_the_fetch(monkeypatch, env_name):
    """Both flags mean "no registry network calls", so neither may issue an HTTP request."""
    monkeypatch.setenv(env_name, "True")
    calls = _serve(monkeypatch, httpx.Response(200, json={"rules": _REMOTE_RULES}, request=httpx.Request("GET", _URL)))

    assert get_fallback_generalizations(url=_URL) == load_local_fallback_generalizations()
    assert calls["count"] == 0


def test_install_uses_the_configured_url_and_compiles_the_rules(
    monkeypatch, restore_generalizations
):
    monkeypatch.setattr(litellm, "fallback_generalizations_url", _URL)
    _serve(monkeypatch, httpx.Response(200, json={"rules": _REMOTE_RULES}, request=httpx.Request("GET", _URL)))

    assert install_fallback_generalizations() == _REMOTE_RULES
    assert match_routing_generalization("widget-9") == "openai"


def test_bundled_file_is_the_only_home_of_the_rules():
    """The registry must no longer carry the block, and the standalone file must carry it."""
    root = os.path.join(os.path.dirname(__file__), "../../..")
    for name in [
        "model_prices_and_context_window.json",
        "litellm/model_prices_and_context_window_backup.json",
    ]:
        with open(os.path.join(root, name)) as f:
            assert "fallback_generalizations" not in json.load(f), name

    assert load_local_fallback_generalizations()
