import litellm
from litellm import utils as litellm_utils_module

CANARY_MODEL = "conftest-isolation-canary-model"


def test_register_model_ledger_entry_is_scoped_to_this_test():
    litellm.register_model({CANARY_MODEL: {"litellm_provider": "openai", "input_cost_per_token": 0.001}})
    assert CANARY_MODEL in litellm_utils_module._runtime_registered_model_cost


def test_register_model_ledger_entry_was_rolled_back():
    assert CANARY_MODEL not in litellm_utils_module._runtime_registered_model_cost
