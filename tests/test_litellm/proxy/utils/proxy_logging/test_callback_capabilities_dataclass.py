"""Pin the ``_CallbackCapabilities`` dataclass shape and defaults."""

from __future__ import annotations

import dataclasses

import pytest

from litellm.proxy.utils import _CallbackCapabilities


def test_callback_capabilities_default_values():
    caps = _CallbackCapabilities()
    snapshot = {
        "has_post_call_response_headers": caps.has_post_call_response_headers,
        "has_iterator_override": caps.has_iterator_override,
        "has_streaming_chunk_override": caps.has_streaming_chunk_override,
        "has_guardrail": caps.has_guardrail,
        "has_pre_call_override": caps.has_pre_call_override,
        "iterator_overrides": caps.iterator_overrides,
        "resolved_callbacks": caps.resolved_callbacks,
    }
    assert snapshot == {
        "has_post_call_response_headers": False,
        "has_iterator_override": False,
        "has_streaming_chunk_override": False,
        "has_guardrail": False,
        "has_pre_call_override": False,
        "iterator_overrides": (),
        "resolved_callbacks": (),
    }


def test_callback_capabilities_explicit_values_preserved():
    cb1 = object()
    cb2 = object()
    caps = _CallbackCapabilities(
        has_post_call_response_headers=True,
        has_iterator_override=True,
        has_streaming_chunk_override=False,
        has_guardrail=True,
        has_pre_call_override=False,
        iterator_overrides=((cb1, "override"), (cb2, "apply_guardrail")),
        resolved_callbacks=(cb1, cb2),
    )
    assert caps.has_post_call_response_headers is True
    assert caps.iterator_overrides == ((cb1, "override"), (cb2, "apply_guardrail"))
    assert caps.resolved_callbacks == (cb1, cb2)


def test_callback_capabilities_is_frozen_error_on_mutation_raises():
    caps = _CallbackCapabilities()
    with pytest.raises(dataclasses.FrozenInstanceError):
        caps.has_post_call_response_headers = True  # type: ignore[misc]


def test_callback_capabilities_invalid_field_error_raises():
    with pytest.raises(TypeError):
        _CallbackCapabilities(unknown_field=True)  # type: ignore[call-arg]
