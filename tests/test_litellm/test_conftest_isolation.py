import litellm
from litellm import Router
from litellm import router as litellm_router_module
from litellm import utils as litellm_utils_module

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
