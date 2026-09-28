import pytest

pytest.register_assert_rewrite(
    "tests.test_litellm_rust.support.callback_contract",
    "tests.test_litellm_rust.support.stream_callback_contract",
)
