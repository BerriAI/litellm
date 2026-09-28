import pytest

from tests.test_litellm_rust.support.callback_contract import (
    CallbackRoute,
    TestCallbackContract as TestCallbackContract,
)
from tests.test_litellm_rust.support.callback_routes import chat_contract

pytestmark = pytest.mark.requires_rust_extension


@pytest.fixture
def callback_route() -> CallbackRoute:
    return chat_contract()
