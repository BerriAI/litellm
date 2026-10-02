import sys
from collections.abc import Iterator
from typing import Final

import pytest

import litellm
from litellm import Router
from litellm import router as litellm_router_module
from litellm import utils as litellm_utils_module
from litellm.caching.caching import DualCache
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from tests.unit import conftest as unit_harness

CANARY_MODEL = "conftest-isolation-canary-model"


class _CanaryRouterHolder:
    router: Router | None = None


def test_register_model_ledger_entry_is_scoped_to_this_test():
    litellm.register_model({CANARY_MODEL: {"litellm_provider": "openai", "input_cost_per_token": 0.001}})
    assert CANARY_MODEL in litellm_utils_module._runtime_registered_model_cost


def test_register_model_ledger_entry_was_rolled_back():
    assert CANARY_MODEL not in litellm_utils_module._runtime_registered_model_cost


def test_live_router_membership_is_scoped_to_this_test():
    _CanaryRouterHolder.router = Router(
        model_list=[
            {
                "model_name": "conftest-isolation-canary-router",
                "litellm_params": {"model": "openai/conftest-isolation-canary-backend", "api_key": "sk-canary"},
            }
        ]
    )
    assert _CanaryRouterHolder.router in litellm_router_module._live_routers


def test_live_router_membership_was_rolled_back():
    assert _CanaryRouterHolder.router is not None
    assert _CanaryRouterHolder.router not in litellm_router_module._live_routers


def test_aws_cache_reset_sees_new_handlers_and_replaced_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    module: Final = unit_harness.importlib.import_module("litellm.files.main")
    handler: Final = BaseAWSLLM()
    first_cache: Final = DualCache()
    handler.iam_cache = first_cache
    monkeypatch.setattr(module, "unit_reset_canary", handler, raising=False)
    first_cache.set_cache("unit-reset-key", "first-secret")
    unit_harness._reset_aws_auth_caches()
    assert first_cache.get_cache("unit-reset-key") is None
    second_cache: Final = DualCache()
    handler.iam_cache = second_cache
    second_cache.set_cache("unit-reset-key", "second-secret")
    unit_harness._reset_aws_auth_caches()
    assert second_cache.get_cache("unit-reset-key") is None


def test_aws_cache_reset_sees_a_replaced_module(monkeypatch: pytest.MonkeyPatch) -> None:
    name: Final = "litellm.files.main"
    original_module: Final = unit_harness.importlib.import_module(name)
    package: Final = unit_harness.importlib.import_module("litellm.files")
    monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(package, "main", original_module)
    module: Final = unit_harness.importlib.import_module(name)
    handler: Final = BaseAWSLLM()
    cache: Final = DualCache()
    handler.iam_cache = cache
    monkeypatch.setattr(module, "unit_reset_canary", handler, raising=False)
    cache.set_cache("unit-reset-key", "module-secret")
    unit_harness._reset_aws_auth_caches()
    assert cache.get_cache("unit-reset-key") is None


@pytest.fixture(scope="module", autouse=True)
def registered_collection_models() -> Iterator[None]:
    original_registrations: Final = dict(litellm_utils_module._runtime_registered_model_cost)
    original_routers: Final = frozenset(litellm_router_module._live_routers)
    litellm.register_model(
        {"collection-isolation-ledger": {"litellm_provider": "openai", "input_cost_per_token": 0.001}}
    )
    collection_router: Final = Router(
        model_list=[
            {
                "model_name": "collection-isolation-router",
                "litellm_params": {"model": "openai/collection-isolation-backend", "api_key": "sk-canary"},
            }
        ]
    )
    yield
    litellm_utils_module._runtime_registered_model_cost.clear()
    litellm_utils_module._runtime_registered_model_cost.update(original_registrations)
    litellm_router_module._live_routers.discard(collection_router)
    litellm_router_module._live_routers.update(original_routers)


def test_collection_model_metadata_is_not_replayed_into_a_test_catalog() -> None:
    from litellm.litellm_core_utils.get_model_cost_map import adopt_model_cost_map

    catalog: Final = {"unit-test-catalog-model": {"litellm_provider": "openai", "input_cost_per_token": 0.001}}
    adopt_model_cost_map(catalog)
    assert "collection-isolation-ledger" not in litellm.model_cost
    assert "collection-isolation-router" not in litellm.model_cost
    assert "unit-test-catalog-model" in litellm.model_cost
