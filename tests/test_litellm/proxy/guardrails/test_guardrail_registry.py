import pytest
from unittest.mock import patch

# Import your handler and types
from litellm.proxy.guardrails.guardrail_registry import InMemoryGuardrailHandler
from litellm.types.guardrails import LitellmParams, GuardrailEventHooks
from litellm.proxy.guardrails.guardrail_registry import (
    get_guardrail_initializer_from_hooks,
)


def test_get_guardrail_initializer_from_hooks():
    initializers = get_guardrail_initializer_from_hooks()
    print(f"initializers: {initializers}")
    assert "aim" in initializers


def test_guardrail_class_registry():
    from litellm.proxy.guardrails.guardrail_registry import guardrail_class_registry

    print(f"guardrail_class_registry: {guardrail_class_registry}")
    assert "aim" in guardrail_class_registry
    assert "aporia" in guardrail_class_registry
    
def test_initialize_custom_guardrail_with_api_base(tmp_path):
    """
    Verifies that 'api_base' from litellm_params is correctly passed
    to the custom guardrail's constructor.
    """
    # Dynamically create a guardrail class that stores api_base
    guardrail_content = """
class MyAPIGuardrail:
    def __init__(self, guardrail_name, event_hook, default_on, api_base=None):
        self.guardrail_name = guardrail_name
        self.event_hook = event_hook
        self.default_on = default_on
        self.api_base = api_base

    def async_log_success_event(self, *args, **kwargs): pass
    def async_log_failure_event(self, *args, **kwargs): pass
"""
    guardrail_file = tmp_path / "my_api_guardrail.py"
    guardrail_file.write_text(guardrail_content)

    handler = InMemoryGuardrailHandler()
    guardrail_config = {
        "guardrail_name": "api_base_test",
        "guardrail_type": "my_api_guardrail.MyAPIGuardrail"
    }
    litellm_params_with_api_base = LitellmParams(
        mode=GuardrailEventHooks.pre_call,
        default_on=True,
        api_base="https://my-custom-api.com/v1"
    )

    with patch("litellm.logging_callback_manager.add_litellm_callback") as mock_add_callback:
        initialized_guardrail = handler.initialize_custom_guardrail(
            guardrail=guardrail_config,
            guardrail_type="my_api_guardrail.MyAPIGuardrail",
            litellm_params=litellm_params_with_api_base,
            config_file_path=str(guardrail_file),
        )

    # --- Robust attribute access using getattr ---
    assert initialized_guardrail is not None
    api_base_val = getattr(initialized_guardrail, "api_base", None)
    assert api_base_val == "https://my-custom-api.com/v1"
    mock_add_callback.assert_called_once()

def test_initialize_custom_guardrail_without_api_base(tmp_path):
    """
    Verifies that the function works correctly when 'api_base' is
    not provided, ensuring no regressions.
    """
    guardrail_content = """
class MyRegularGuardrail:
    def __init__(self, guardrail_name, event_hook, default_on, api_base=None):
        self.guardrail_name = guardrail_name
        self.event_hook = event_hook
        self.default_on = default_on
        self.api_base = api_base

    def async_log_success_event(self, *args, **kwargs): pass
    def async_log_failure_event(self, *args, **kwargs): pass
"""
    guardrail_file = tmp_path / "my_regular_guardrail.py"
    guardrail_file.write_text(guardrail_content)

    handler = InMemoryGuardrailHandler()
    guardrail_config = {
        "guardrail_name": "regular_test",
        "guardrail_type": "my_regular_guardrail.MyRegularGuardrail"
    }
    litellm_params_without_api_base = LitellmParams(
        mode=GuardrailEventHooks.post_call,
        default_on=False
    )

    with patch("litellm.logging_callback_manager.add_litellm_callback") as mock_add_callback:
        initialized_guardrail = handler.initialize_custom_guardrail(
            guardrail=guardrail_config,
            guardrail_type="my_regular_guardrail.MyRegularGuardrail",
            litellm_params=litellm_params_without_api_base,
            config_file_path=str(guardrail_file),
        )

    # --- Robust attribute access using getattr ---
    assert initialized_guardrail is not None
    api_base_val = getattr(initialized_guardrail, "api_base", None)
    assert api_base_val is None
    mock_add_callback.assert_called_once()
