from types import SimpleNamespace

import pytest

try:
    from litellm.proxy.guardrails import _rust  # noqa: F401
except ImportError:
    pytest.skip("litellm guardrails _rust engine not built", allow_module_level=True)

from litellm.proxy.guardrails.guardrail_hooks.guardrail_v2.guardrail_v2 import (
    GuardrailV2,
    build_v2_config,
)


def _params(**overrides):
    base = dict(
        optional_params=None,
        api_key="sk-test",
        api_base=None,
        model="omni-moderation-latest",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_openai_moderation_builds_rust_config():
    # openai_moderation always runs on the Rust engine, including streaming
    # configs (the streaming behaviour is honoured by forwarding attributes).
    cfg = build_v2_config("openai_moderation", _params())
    assert cfg is not None
    assert cfg["guardrail"] == "openai_moderation"


def test_streaming_overrides_are_forwarded_to_the_rust_guardrail():
    guardrail = GuardrailV2(
        engine_config={"guardrail": "openai_moderation"},
        streaming_end_of_stream_only=True,
        streaming_sampling_rate=9,
        guardrail_name="t",
    )
    # Read by UnifiedLLMGuardrails to drive streaming moderation while the
    # apply call still routes to Rust.
    assert guardrail.streaming_end_of_stream_only is True
    assert guardrail.streaming_sampling_rate == 9


def test_streaming_defaults_match_python_guardrail():
    guardrail = GuardrailV2(
        engine_config={"guardrail": "openai_moderation"}, guardrail_name="t"
    )
    assert guardrail.streaming_end_of_stream_only is False
    assert guardrail.streaming_sampling_rate == 5
