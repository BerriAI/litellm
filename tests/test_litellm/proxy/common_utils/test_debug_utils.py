import json
import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm._logging import (
    verbose_logger,
    verbose_proxy_logger,
    verbose_router_logger,
)
from litellm.proxy.common_utils.debug_utils import init_verbose_loggers

_ALL_VERBOSE_LOGGERS = (verbose_logger, verbose_router_logger, verbose_proxy_logger)


@pytest.fixture(autouse=True)
def _reset_verbose_logger_levels():
    """init_verbose_loggers mutates process-global logger levels; snapshot and
    restore so tests can't leak state into each other."""
    saved = [lg.level for lg in _ALL_VERBOSE_LOGGERS]
    for lg in _ALL_VERBOSE_LOGGERS:
        lg.setLevel(logging.NOTSET)
    try:
        yield
    finally:
        for lg, level in zip(_ALL_VERBOSE_LOGGERS, saved):
            lg.setLevel(level)


def _effective_levels():
    return {lg.name: lg.getEffectiveLevel() for lg in _ALL_VERBOSE_LOGGERS}


def _worker_config_json():
    """The shape the litellm CLI writes: a JSON blob with debug flags off."""
    return json.dumps(
        {"model": None, "debug": False, "detailed_debug": False, "config": "/etc/litellm/config.yaml"}
    )


def test_litellm_log_debug_json_worker_config_sets_all_loggers(monkeypatch):
    monkeypatch.setenv("LITELLM_LOG", "DEBUG")
    monkeypatch.setenv("WORKER_CONFIG", _worker_config_json())

    init_verbose_loggers()

    assert _effective_levels() == {
        "LiteLLM": logging.DEBUG,
        "LiteLLM Router": logging.DEBUG,
        "LiteLLM Proxy": logging.DEBUG,
    }


def test_litellm_log_debug_file_worker_config_sets_all_loggers(monkeypatch, tmp_path):
    """K8s/Helm hand the proxy a config file path via WORKER_CONFIG. That shape
    used to short-circuit init_verbose_loggers before LITELLM_LOG was read, so
    LITELLM_LOG=DEBUG produced no debug logs at all."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("model_list: []\n")
    monkeypatch.setenv("LITELLM_LOG", "DEBUG")
    monkeypatch.setenv("WORKER_CONFIG", str(config_file))

    init_verbose_loggers()

    assert _effective_levels() == {
        "LiteLLM": logging.DEBUG,
        "LiteLLM Router": logging.DEBUG,
        "LiteLLM Proxy": logging.DEBUG,
    }


def test_litellm_log_debug_without_worker_config_sets_all_loggers(monkeypatch):
    monkeypatch.setenv("LITELLM_LOG", "DEBUG")
    monkeypatch.delenv("WORKER_CONFIG", raising=False)

    init_verbose_loggers()

    assert _effective_levels() == {
        "LiteLLM": logging.DEBUG,
        "LiteLLM Router": logging.DEBUG,
        "LiteLLM Proxy": logging.DEBUG,
    }


def test_litellm_log_debug_turns_on_core_package_logger(monkeypatch):
    """Regression for the omitted core logger: LITELLM_LOG=DEBUG used to raise
    only the router/proxy loggers, leaving the "LiteLLM" package logger (which
    emits raw request/response, cost tracking, provider calls) at WARNING."""
    monkeypatch.setenv("LITELLM_LOG", "DEBUG")
    monkeypatch.setenv("WORKER_CONFIG", _worker_config_json())

    init_verbose_loggers()

    assert verbose_logger.isEnabledFor(logging.DEBUG)


def test_litellm_log_info_sets_all_loggers_to_info(monkeypatch):
    monkeypatch.setenv("LITELLM_LOG", "INFO")
    monkeypatch.setenv("WORKER_CONFIG", _worker_config_json())

    init_verbose_loggers()

    assert _effective_levels() == {
        "LiteLLM": logging.INFO,
        "LiteLLM Router": logging.INFO,
        "LiteLLM Proxy": logging.INFO,
    }


def test_detailed_debug_flag_wins_over_env(monkeypatch):
    monkeypatch.setenv("LITELLM_LOG", "INFO")
    monkeypatch.setenv(
        "WORKER_CONFIG", json.dumps({"debug": False, "detailed_debug": True})
    )

    init_verbose_loggers()

    assert _effective_levels() == {
        "LiteLLM": logging.DEBUG,
        "LiteLLM Router": logging.DEBUG,
        "LiteLLM Proxy": logging.DEBUG,
    }


def test_debug_flag_sets_info(monkeypatch):
    monkeypatch.delenv("LITELLM_LOG", raising=False)
    monkeypatch.setenv(
        "WORKER_CONFIG", json.dumps({"debug": True, "detailed_debug": False})
    )

    init_verbose_loggers()

    assert _effective_levels() == {
        "LiteLLM": logging.INFO,
        "LiteLLM Router": logging.INFO,
        "LiteLLM Proxy": logging.INFO,
    }


def test_no_flags_and_no_env_leaves_loggers_untouched(monkeypatch):
    monkeypatch.delenv("LITELLM_LOG", raising=False)
    monkeypatch.delenv("WORKER_CONFIG", raising=False)

    init_verbose_loggers()

    assert _effective_levels() == {
        "LiteLLM": logging.WARNING,
        "LiteLLM Router": logging.WARNING,
        "LiteLLM Proxy": logging.WARNING,
    }
