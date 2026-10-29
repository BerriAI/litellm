from collections.abc import Iterator

import pytest

from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail import (
    unified_guardrail as unified_module,
)


@pytest.fixture(autouse=True)
def _forget_unscanned_warnings() -> Iterator[None]:
    unified_module._warn_left_unscanned_once.cache_clear()
    yield
    unified_module._warn_left_unscanned_once.cache_clear()
