import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))  # Adds the parent directory to the system path

from litellm.proxy.guardrails.guardrail_registry import InMemoryGuardrailHandler
from litellm.types.guardrails import SupportedGuardrailIntegrations


def test_initialize_presidio_guardrail():
    """
    Test that initialize_guardrail correctly uses registered initializers
    for presidio guardrail
    """
    # Setup test data for a non-custom guardrail (using Presidio as an example)
    test_guardrail = {
        "guardrail_name": "test_presidio_guardrail",
        "litellm_params": {
            "guardrail": SupportedGuardrailIntegrations.PRESIDIO.value,
            "mode": "pre_call",
            "presidio_analyzer_api_base": "https://fakelink.com/v1/presidio/analyze",
            "presidio_anonymizer_api_base": "https://fakelink.com/v1/presidio/anonymize",
        },
    }

    # Call the initialize_guardrail method
    guardrail_handler = InMemoryGuardrailHandler()
    result = guardrail_handler.initialize_guardrail(
        guardrail=test_guardrail,
    )

    assert result["guardrail_name"] == "test_presidio_guardrail"
    assert result["litellm_params"].guardrail == SupportedGuardrailIntegrations.PRESIDIO.value
    assert result["litellm_params"].mode == "pre_call"


def test_initialize_guardrail_preserves_guardrail_info():
    """
    Regression (LIT-2529): initialize_guardrail must carry guardrail_info into the
    stored in-memory Guardrail. Dropping it left the Guardrail Monitor's usage
    endpoints unable to render type/description for YAML-defined guardrails.
    """
    test_guardrail = {
        "guardrail_name": "test_presidio_with_info",
        "litellm_params": {
            "guardrail": SupportedGuardrailIntegrations.PRESIDIO.value,
            "mode": "pre_call",
            "presidio_analyzer_api_base": "https://fakelink.com/v1/presidio/analyze",
            "presidio_anonymizer_api_base": "https://fakelink.com/v1/presidio/anonymize",
        },
        "guardrail_info": {"type": "PII", "description": "masks PII"},
    }

    guardrail_handler = InMemoryGuardrailHandler()
    result = guardrail_handler.initialize_guardrail(guardrail=test_guardrail)

    assert result is not None
    assert result["guardrail_info"] == {"type": "PII", "description": "masks PII"}
    stored = guardrail_handler.IN_MEMORY_GUARDRAILS[result["guardrail_id"]]
    assert stored["guardrail_info"] == {"type": "PII", "description": "masks PII"}


@pytest.mark.parametrize(
    "config_value, expected",
    [(True, True), (False, False), (None, False)],
)
def test_initialize_guardrail_sets_run_in_parallel(config_value, expected):
    """run_in_parallel from litellm_params must reach the built guardrail instance."""
    litellm_params = {
        "guardrail": SupportedGuardrailIntegrations.PRESIDIO.value,
        "mode": "pre_call",
        "presidio_analyzer_api_base": "https://fakelink.com/v1/presidio/analyze",
        "presidio_anonymizer_api_base": "https://fakelink.com/v1/presidio/anonymize",
    }
    if config_value is not None:
        litellm_params["run_in_parallel"] = config_value

    guardrail_handler = InMemoryGuardrailHandler()
    result = guardrail_handler.initialize_guardrail(
        guardrail={"guardrail_name": "test_parallel_flag", "litellm_params": litellm_params},
    )

    custom_guardrail = guardrail_handler.guardrail_id_to_custom_guardrail[result["guardrail_id"]]
    assert custom_guardrail.run_in_parallel is expected
