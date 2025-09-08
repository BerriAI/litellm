import datetime
import json
import os
import sys
import unittest
from unittest.mock import ANY, MagicMock, patch

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.guardrails import GuardrailEventHooks, Mode


def test_custom_guardrail_with_mode(monkeypatch):
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.premium_user", True
    )  # Set premium_user to True
    cg = CustomGuardrail(
        guardrail_name="test_guardrail",
        supported_event_hooks=[
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.logging_only,
        ],
        event_hook=Mode(
            tags={"test_tag": "pre_call"},
            default="logging_only",
        ),
        default_on=True,
    )

    assert (
        cg.should_run_guardrail(
            data={
                "messages": [{"role": "user", "content": "test_message"}],
                "litellm_metadata": {"tags": ["test_tag"]},
            },
            event_type=GuardrailEventHooks.pre_call,
        )
        is True
    )

    assert (
        cg.should_run_guardrail(
            data={
                "messages": [{"role": "user", "content": "test_message"}],
            },
            event_type=GuardrailEventHooks.pre_call,
        )
        is False
    )

    assert (
        cg.should_run_guardrail(
            data={
                "messages": [{"role": "user", "content": "test_message"}],
                "litellm_metadata": {"tags": ["test_tag"]},
            },
            event_type=GuardrailEventHooks.logging_only,
        )
        is False
    )
