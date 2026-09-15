import importlib
import logging

import pytest

import litellm_proxy_extras._logging as extras_logging


def test_litellm_log_error_silences_extras_info_lines(monkeypatch):
    saved_handlers = logging.getLogger("litellm_proxy_extras").handlers[:]
    monkeypatch.setenv("LITELLM_LOG", "ERROR")
    logging.getLogger("litellm_proxy_extras").handlers[:] = []
    try:
        reloaded = importlib.reload(extras_logging).logger
        assert reloaded.isEnabledFor(logging.INFO) is False
        assert reloaded.isEnabledFor(logging.ERROR) is True
    finally:
        logging.getLogger("litellm_proxy_extras").handlers[:] = saved_handlers
        logging.getLogger("litellm_proxy_extras").setLevel(logging.INFO)


@pytest.mark.parametrize("litellm_log", [None, "info", "DEBUG"])
def test_unset_or_verbose_litellm_log_keeps_extras_info_lines(monkeypatch, litellm_log):
    saved_handlers = logging.getLogger("litellm_proxy_extras").handlers[:]
    if litellm_log is None:
        monkeypatch.delenv("LITELLM_LOG", raising=False)
    else:
        monkeypatch.setenv("LITELLM_LOG", litellm_log)
    logging.getLogger("litellm_proxy_extras").handlers[:] = []
    try:
        assert importlib.reload(extras_logging).logger.isEnabledFor(logging.INFO) is True
    finally:
        logging.getLogger("litellm_proxy_extras").handlers[:] = saved_handlers
        logging.getLogger("litellm_proxy_extras").setLevel(logging.INFO)
