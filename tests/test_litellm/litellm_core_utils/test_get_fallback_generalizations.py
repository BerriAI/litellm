"""
Tests for loading the standalone fallback generalizations file: the remote copy wins when
it is usable, the bundled copy covers every failure, and the local-only env flags skip HTTP.
"""

import json
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../.."))

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

_REMOTE_RULES = (
    {
        "name": "remote-only",
        "pattern": r"^widget-",
        "model_info": {"litellm_provider": "openai"},
    },
)


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


def _serving(payload: object):
    """A fetcher returning ``payload``, or raising it when it is an exception."""

    def fetch(url: str) -> object:
        if isinstance(payload, Exception):
            raise payload
        return payload

    return fetch


def _never_called(url: str) -> object:
    raise AssertionError(f"fetched {url} while in local-only mode")


def test_remote_rules_are_used_when_the_fetch_succeeds():
    rules = get_fallback_generalizations(url=_URL, fetch=_serving({"rules": list(_REMOTE_RULES)}))
    assert rules == _REMOTE_RULES


def test_network_failure_falls_back_to_the_bundled_file():
    """A rules fetch must never take routing down: an unreachable host keeps the shipped rules."""
    rules = get_fallback_generalizations(url=_URL, fetch=_serving(httpx.ConnectError("connection refused")))
    assert rules == load_local_fallback_generalizations()


def test_http_error_falls_back_to_the_bundled_file():
    """The URL 404s on every branch that predates this file, so a 404 must not wipe the rules."""
    error = httpx.HTTPStatusError(
        "404", request=httpx.Request("GET", _URL), response=httpx.Response(404)
    )
    rules = get_fallback_generalizations(url=_URL, fetch=_serving(error))
    assert rules == load_local_fallback_generalizations()


def test_unparseable_remote_body_falls_back_to_the_bundled_file():
    rules = get_fallback_generalizations(url=_URL, fetch=_serving(json.JSONDecodeError("boom", "", 0)))
    assert rules == load_local_fallback_generalizations()


@pytest.mark.parametrize(
    "payload",
    [{"rules": {}}, {"rules": []}, {}, [], "rules"],
    ids=["rules_not_a_list", "rules_empty", "no_rules_key", "not_an_object", "not_json_object"],
)
def test_malformed_remote_payload_falls_back_to_the_bundled_file(payload):
    """A truncated or reshaped remote file must not silently disable every rule."""
    assert get_fallback_generalizations(url=_URL, fetch=_serving(payload)) == load_local_fallback_generalizations()


def test_non_object_rules_are_skipped_without_dropping_the_valid_ones():
    payload = {"rules": ["nonsense", *_REMOTE_RULES]}
    assert get_fallback_generalizations(url=_URL, fetch=_serving(payload)) == _REMOTE_RULES


@pytest.mark.parametrize(
    "env_name",
    ["LITELLM_LOCAL_FALLBACK_GENERALIZATIONS", "LITELLM_LOCAL_MODEL_COST_MAP"],
)
def test_local_only_env_skips_the_fetch(monkeypatch, env_name):
    """Both flags mean "no registry network calls", so neither may issue a request."""
    monkeypatch.setenv(env_name, "True")
    assert get_fallback_generalizations(url=_URL, fetch=_never_called) == load_local_fallback_generalizations()


def test_install_compiles_the_loaded_rules(restore_generalizations):
    installed = install_fallback_generalizations(url=_URL, fetch=_serving({"rules": list(_REMOTE_RULES)}))

    assert installed == _REMOTE_RULES
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
