from typing import Final

import pytest

from litellm.litellm_core_utils.custom_logger_registry import CustomLoggerRegistry
from litellm.proxy.guardrails.guardrail_registry import guardrail_initializer_registry
from litellm.types.guardrails import SupportedGuardrailIntegrations
from tests.test_litellm_rust.integrations import (
    DISCOVERED_ONLY_GUARDRAIL_NAMES,
    ENTERPRISE_LOGGER_NAMES,
    GUARDRAIL_NAMES,
    GUARDRAIL_OBLIGATIONS,
    LOGGER_OBLIGATIONS,
    OSS_LOGGER_NAMES,
    REQUIRED_GUARDRAIL_BEHAVIOR,
    REQUIRED_LOGGER_BEHAVIOR,
)

pytestmark = pytest.mark.requires_rust_extension


def test_logger_catalogue_matches_registered_integration_names() -> None:
    registered_names: Final = frozenset(CustomLoggerRegistry.CALLBACK_CLASS_STR_TO_CLASS_TYPE)
    catalogued_names: Final = frozenset(
        name for obligation in LOGGER_OBLIGATIONS.values() for name in obligation.registration_names
    )

    assert OSS_LOGGER_NAMES <= registered_names
    assert registered_names <= OSS_LOGGER_NAMES | ENTERPRISE_LOGGER_NAMES
    assert catalogued_names == OSS_LOGGER_NAMES | ENTERPRISE_LOGGER_NAMES


def test_guardrail_catalogue_matches_registered_integration_names() -> None:
    enum_names: Final = frozenset(integration.value for integration in SupportedGuardrailIntegrations)
    initializer_names: Final = frozenset(guardrail_initializer_registry)

    assert enum_names == GUARDRAIL_NAMES
    assert initializer_names == GUARDRAIL_NAMES | DISCOVERED_ONLY_GUARDRAIL_NAMES
    assert frozenset(GUARDRAIL_OBLIGATIONS) == initializer_names


def test_required_integrations_have_behavioral_case_labels() -> None:
    labelled_loggers: Final = frozenset(
        name
        for obligation in LOGGER_OBLIGATIONS.values()
        if obligation.behavioral_case_labels
        for name in obligation.registration_names
    )
    labelled_guardrails: Final = frozenset(
        name for name, obligation in GUARDRAIL_OBLIGATIONS.items() if obligation.behavioral_case_labels
    )

    assert REQUIRED_LOGGER_BEHAVIOR <= labelled_loggers
    assert REQUIRED_GUARDRAIL_BEHAVIOR <= labelled_guardrails
