from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.token_counter import token_counter
from litellm.rust_bridge import bindings
from litellm.rust_bridge import token_counter as bridge

pytestmark = pytest.mark.requires_rust_extension

ANTHROPIC_MODEL: Final = "claude-sonnet-4-5-20250929"
TEXTS: Final = (
    "Hello, how are you today?",
    "I'VE got 1234567 things; it's \"fine\"...\r\n\r\n  café مرحبا 😀 <|endoftext|>",
    "x " * 5_000,
    "",
)


@pytest.fixture
def native_bridge() -> None:
    bridge.TOKEN_COUNTER.reset()
    bridge._counter.cache_clear()  # pyright: ignore[reportPrivateUsage]  # the counter cache is keyed on the factory object
    assert bindings.get_native_bridge() is not None


@pytest.mark.parametrize("text", TEXTS)
def test_native_anthropic_text_count_matches_python(native_bridge: None, text: str) -> None:
    messages: Final = [{"role": "system", "content": "You are terse."}, {"role": "user", "name": "bob", "content": text}]

    litellm.rust(False)
    python_text: Final = token_counter(model=ANTHROPIC_MODEL, text=text)
    python_messages: Final = token_counter(model=ANTHROPIC_MODEL, messages=messages)
    litellm.rust(True)

    rust_count: Final = bridge.text_counter("anthropic")
    assert rust_count is not None
    assert rust_count(text) == python_text
    assert token_counter(model=ANTHROPIC_MODEL, text=text) == python_text
    assert token_counter(model=ANTHROPIC_MODEL, messages=messages) == python_messages
