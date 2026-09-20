import os
from typing import Final

import litellm
import pytest
from pytest_socket import enable_socket, socket_allow_hosts

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

LOOPBACK_HOSTS: Final = ["127.0.0.1", "::1"]


@pytest.fixture
def local_model_cost_map(monkeypatch):
    original_model_cost = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost = original_model_cost
        litellm.get_model_info.cache_clear()


def _allow_loopback_only() -> None:
    socket_allow_hosts(LOOPBACK_HOSTS, allow_unix_socket=True)


_allow_loopback_only()


@pytest.hookimpl(trylast=True)
def pytest_runtest_setup() -> None:
    _allow_loopback_only()


def pytest_sessionfinish() -> None:
    enable_socket()
