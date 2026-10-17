import asyncio
import importlib.util
import os
import sys
from types import ModuleType
import unittest
from unittest.mock import MagicMock

# Mock base guardrail classes before importing module
if "litellm.integrations.custom_guardrail" not in sys.modules:
    cg_mod = ModuleType("litellm.integrations.custom_guardrail")
    class CustomGuardrail:
        def __init__(self, **kwargs): pass
    cg_mod.CustomGuardrail = CustomGuardrail
    cg_mod.log_guardrail_information = lambda f: f
    sys.modules["litellm.integrations.custom_guardrail"] = cg_mod

if "litellm._logging" not in sys.modules:
    log_mod = ModuleType("litellm._logging")
    log_mod.verbose_proxy_logger = MagicMock()
    sys.modules["litellm._logging"] = log_mod

if "litellm.caching.caching" not in sys.modules:
    cache_mod = ModuleType("litellm.caching.caching")
    cache_mod.DualCache = MagicMock
    sys.modules["litellm.caching.caching"] = cache_mod

if "litellm.proxy._types" not in sys.modules:
    types_mod = ModuleType("litellm.proxy._types")
    types_mod.UserAPIKeyAuth = MagicMock
    sys.modules["litellm.proxy._types"] = types_mod

# Direct module load
file_path = os.path.join(
    os.path.dirname(__file__),
    "../litellm/proxy/guardrails/guardrail_hooks/action_gate/action_gate.py",
)
spec = importlib.util.spec_from_file_location("action_gate_module", file_path)
action_gate_mod = importlib.util.module_from_spec(spec)
sys.modules["action_gate_module"] = action_gate_mod
spec.loader.exec_module(action_gate_mod)

ActionGateGuardrail = action_gate_mod.ActionGateGuardrail
GENESIS_HASH = action_gate_mod.GENESIS_HASH


class TestActionGateGuardrail(unittest.TestCase):
    def setUp(self):
        self.guardrail = ActionGateGuardrail(
            never_equate_intent_to_approval=True,
            enforce_action_boundary=True,
        )
        self.mock_user_auth = MagicMock()
        self.mock_user_auth.user_id = "test_user_001"
        self.mock_cache = MagicMock()

    def test_pre_call_hook_allows_normal_request(self):
        data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Analyze security logs"}],
            "tools": [{"type": "function", "function": {"name": "query_logs"}}],
        }
        res = asyncio.run(
            self.guardrail.async_pre_call_hook(
                user_api_key_dict=self.mock_user_auth,
                cache=self.mock_cache,
                data=data,
                call_type="completion",
            )
        )
        self.assertEqual(res, data)
        self.assertEqual(len(self.guardrail.get_ledger_entries()), 1)

    def test_post_call_success_hook_records_audit_ledger(self):
        data = {"model": "claude-3-5-sonnet", "messages": []}
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{"id": "call_1", "function": {"name": "execute_query"}}],
                    }
                }
            ]
        }
        res = asyncio.run(
            self.guardrail.async_post_call_success_hook(
                data=data,
                user_api_key_dict=self.mock_user_auth,
                response=response,
            )
        )
        self.assertEqual(res, response)
        entries = self.guardrail.get_ledger_entries()
        self.assertGreaterEqual(len(entries), 1)
        self.assertTrue(self.guardrail.verify_ledger_integrity())

    def test_hash_chain_integrity(self):
        # Record 3 events
        self.guardrail._record_audit_entry("e1", "m1", "u1", "ok", {"step": 1})
        self.guardrail._record_audit_entry("e2", "m2", "u2", "ok", {"step": 2})
        self.guardrail._record_audit_entry("e3", "m3", "u3", "ok", {"step": 3})

        entries = self.guardrail.get_ledger_entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertEqual(entries[2]["prev_hash"], entries[1]["curr_hash"])
        self.assertTrue(self.guardrail.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()
