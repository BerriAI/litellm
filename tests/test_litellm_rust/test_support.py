from types import ModuleType
from typing import Final
from pathlib import Path

import pytest

from tests.test_litellm_rust import conftest
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


def test_rust_extension_gate_skips_only_rust_suite_items(monkeypatch: pytest.MonkeyPatch) -> None:
    class Item:
        def __init__(self, path: Path) -> None:
            self.path = path
            self.markers: list[object] = []

        def add_marker(self, marker: object) -> None:
            self.markers.append(marker)

    monkeypatch.setattr(conftest, "_parse_env_bool", lambda value: False)
    rust_item: Final = Item(Path("tests/test_litellm_rust/ocr/test_dispatch.py"))
    other_item: Final = Item(Path("tests/test_litellm/test_completion.py"))

    conftest.pytest_collection_modifyitems([rust_item, other_item])  # pyright: ignore[reportArgumentType]  # lightweight items isolate hook selection

    assert rust_item.markers
    assert not other_item.markers
