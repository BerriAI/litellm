"""
Tests for fix/guardrail-request-data-passthrough

Verifies that BaseTranslation.process_output_response() declares request_data
so implementing classes are type-consistent with the abstract interface.

Related issue: https://github.com/BerriAI/litellm/issues/22821
"""

import inspect
import pytest
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation


def test_base_translation_process_output_response_has_request_data_param():
    """
    All concrete implementations of process_output_response() accept a
    request_data keyword argument, but the abstract base did not declare it.
    This test ensures the abstract signature matches the implementations.
    """
    sig = inspect.signature(BaseTranslation.process_output_response)
    assert "request_data" in sig.parameters, (
        "BaseTranslation.process_output_response() must declare a 'request_data' "
        "parameter so all implementations are type-consistent with the interface."
    )
    param = sig.parameters["request_data"]
    assert param.default is None, (
        "'request_data' must default to None for backwards compatibility."
    )


def test_all_handler_implementations_accept_request_data():
    """
    Verify a representative set of concrete handler implementations also
    accept request_data (ensuring they conform to the updated abstract sig).
    """
    import importlib

    handlers = [
        "litellm.llms.openai.chat.guardrail_translation.handler",
        "litellm.llms.anthropic.chat.guardrail_translation.handler",
        "litellm.llms.openai.completion.guardrail_translation.handler",
    ]

    for module_path in handlers:
        try:
            mod = importlib.import_module(module_path)
        except ImportError:
            pytest.skip(f"Module {module_path} not available")

        # Find the class that implements process_output_response
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if not hasattr(obj, "process_output_response"):
                continue
            # Skip the abstract base itself
            if obj is BaseTranslation:
                continue
            # Only check classes from this module
            if obj.__module__ != module_path:
                continue

            sig = inspect.signature(obj.process_output_response)
            assert "request_data" in sig.parameters, (
                f"{module_path}.{name}.process_output_response() "
                f"must accept 'request_data' to match BaseTranslation."
            )
