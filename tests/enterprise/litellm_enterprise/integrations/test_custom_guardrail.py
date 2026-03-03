import os
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.guardrails import GuardrailEventHooks, Mode


def test_custom_guardrail_with_mode_default_list(monkeypatch):
    """Test Mode with default as a list of modes (e.g. default: ["pre_call", "post_call"])"""
    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", True)
    cg = CustomGuardrail(
        guardrail_name="test_guardrail",
        supported_event_hooks=[
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.post_call,
            GuardrailEventHooks.logging_only,
        ],
        event_hook=Mode(
            tags={"test_tag": "logging_only"},
            default=["pre_call", "post_call"],
        ),
        default_on=True,
    )

    # No tag match → default fires for pre_call
    assert (
        cg.should_run_guardrail(
            data={"messages": [{"role": "user", "content": "test"}]},
            event_type=GuardrailEventHooks.pre_call,
        )
        is True
    )

    # No tag match → default fires for post_call
    assert (
        cg.should_run_guardrail(
            data={"messages": [{"role": "user", "content": "test"}]},
            event_type=GuardrailEventHooks.post_call,
        )
        is True
    )

    # No tag match → logging_only NOT in default list, should not fire
    assert (
        cg.should_run_guardrail(
            data={"messages": [{"role": "user", "content": "test"}]},
            event_type=GuardrailEventHooks.logging_only,
        )
        is False
    )

    # Tag matches → only logging_only should fire
    assert (
        cg.should_run_guardrail(
            data={
                "messages": [{"role": "user", "content": "test"}],
                "litellm_metadata": {"tags": ["test_tag"]},
            },
            event_type=GuardrailEventHooks.logging_only,
        )
        is True
    )

    # Tag matches → pre_call should NOT fire (tag says logging_only)
    assert (
        cg.should_run_guardrail(
            data={
                "messages": [{"role": "user", "content": "test"}],
                "litellm_metadata": {"tags": ["test_tag"]},
            },
            event_type=GuardrailEventHooks.pre_call,
        )
        is False
    )

    # Tag matches → post_call should NOT fire (tag says logging_only)
    assert (
        cg.should_run_guardrail(
            data={
                "messages": [{"role": "user", "content": "test"}],
                "litellm_metadata": {"tags": ["test_tag"]},
            },
            event_type=GuardrailEventHooks.post_call,
        )
        is False
    )


def test_custom_guardrail_with_mode_no_default(monkeypatch):
    """Test Mode with no default — guardrail only fires when tag matches"""
    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", True)
    cg = CustomGuardrail(
        guardrail_name="test_guardrail",
        supported_event_hooks=[
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.logging_only,
        ],
        event_hook=Mode(
            tags={"test_tag": "logging_only"},
        ),
        default_on=True,
    )

    # No tag, no default → nothing fires
    assert (
        cg.should_run_guardrail(
            data={"messages": [{"role": "user", "content": "test"}]},
            event_type=GuardrailEventHooks.pre_call,
        )
        is False
    )

    assert (
        cg.should_run_guardrail(
            data={"messages": [{"role": "user", "content": "test"}]},
            event_type=GuardrailEventHooks.logging_only,
        )
        is False
    )

    # Tag matches → only logging_only fires
    assert (
        cg.should_run_guardrail(
            data={
                "messages": [{"role": "user", "content": "test"}],
                "litellm_metadata": {"tags": ["test_tag"]},
            },
            event_type=GuardrailEventHooks.logging_only,
        )
        is True
    )


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
