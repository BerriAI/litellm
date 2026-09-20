from collections.abc import Iterator
from typing import Final

import pytest
from pytest_socket import enable_socket, socket_allow_hosts

import litellm
import litellm.router as litellm_router_module
import litellm.utils as litellm_utils_module

LOOPBACK_HOSTS: Final = ["127.0.0.1", "::1"]


def _allow_loopback_only() -> None:
    socket_allow_hosts(LOOPBACK_HOSTS, allow_unix_socket=True)


@pytest.fixture(autouse=True, scope="session")
def block_external_sockets() -> Iterator[None]:
    _allow_loopback_only()
    yield
    enable_socket()


@pytest.hookimpl(trylast=True)
def pytest_runtest_setup() -> None:
    _allow_loopback_only()


@pytest.fixture(autouse=True)
def isolate_router_model_cost_state() -> Iterator[None]:
    original_live_routers: Final = frozenset(litellm_router_module._live_routers)
    original_runtime_registered_model_cost: Final = {
        model_key: dict(model_value)
        for model_key, model_value in litellm_utils_module._runtime_registered_model_cost.items()
    }
    yield
    for router in tuple(litellm_router_module._live_routers):
        litellm_router_module._live_routers.discard(router)
    for router in original_live_routers:
        litellm_router_module._live_routers.add(router)
    litellm_utils_module._runtime_registered_model_cost.clear()
    litellm_utils_module._runtime_registered_model_cost.update(original_runtime_registered_model_cost)
    litellm_utils_module._invalidate_model_cost_lowercase_map()
    litellm.get_model_info.cache_clear()


@pytest.fixture
def local_model_cost_map(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()
