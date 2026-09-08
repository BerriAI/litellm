from types import ModuleType
from typing import Final

import pytest

from tests.test_litellm_rust.conftest import _isolated_list  # pyright: ignore[reportPrivateUsage]  # exercise fixture restoration without mutating SDK state

pytestmark = pytest.mark.requires_rust_extension


def test_callback_list_restores_identity_after_rebinding_and_failure() -> None:
    container: Final = ModuleType("callback_registry")
    callback: Final = object()
    original: Final = [callback]
    setattr(container, "callbacks", original)

    def failing_test() -> None:
        with _isolated_list(container, "callbacks"):
            assert getattr(container, "callbacks") is original
            assert original == []
            original.append(object())
            setattr(container, "callbacks", [object()])
            raise RuntimeError("test failed")

    with pytest.raises(RuntimeError, match="test failed"):
        failing_test()

    assert getattr(container, "callbacks") is original
    assert original == [callback]
