import pytest

from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail import (
    unified_guardrail as unified_module,
)


@pytest.fixture(autouse=True)
def _forget_unscanned_warnings():
    """The unscanned warning fires once per route and reason, so every guardrail test starts with nothing remembered."""
    unified_module._warn_left_unscanned_once.cache_clear()
    yield
    unified_module._warn_left_unscanned_once.cache_clear()
