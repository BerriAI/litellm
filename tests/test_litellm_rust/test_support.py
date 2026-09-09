import asyncio
from pathlib import Path
from types import ModuleType
from typing import Final

import pytest

import litellm
from litellm.rust_bridge import configuration
from tests.test_litellm_rust import conftest
from tests.test_litellm_rust.conftest import _isolated_list  # pyright: ignore[reportPrivateUsage]  # exercise fixture restoration without mutating SDK state

pytestmark = pytest.mark.requires_rust_extension


@pytest.mark.asyncio
@pytest.mark.parametrize("override", (None, False, True))
@pytest.mark.parametrize("exception_type", (RuntimeError, asyncio.CancelledError))
async def test_backend_restores_override_after_nested_failure(
    monkeypatch: pytest.MonkeyPatch,
    override: bool | None,
    exception_type: type[RuntimeError] | type[asyncio.CancelledError],
) -> None:
    monkeypatch.setenv("LITELLM_RUST", "1")
    configuration.reset_rust_configuration()
    if override is not None:
        litellm.rust(override)

    async def fail_in_nested_backend() -> None:
        async with conftest.isolated_backend("python"):
            assert configuration.rust_enabled() is False
            with pytest.raises(exception_type, match="scope failed"):
                await fail_in_rust_backend()
            assert configuration.rust_enabled() is False
            raise exception_type("scope failed")

    async def fail_in_rust_backend() -> None:
        async with conftest.isolated_backend("rust"):
            assert configuration.rust_enabled() is True
            raise exception_type("scope failed")

    with pytest.raises(exception_type, match="scope failed"):
        await fail_in_nested_backend()

    assert configuration.rust_enabled() is (True if override is None else override)
    monkeypatch.setenv("LITELLM_RUST", "0")
    assert configuration.rust_enabled() is (False if override is None else override)


@pytest.mark.asyncio
async def test_backend_rejects_overlapping_tasks_without_changing_state() -> None:
    async def competing_scope() -> None:
        async with conftest.isolated_backend("rust"):
            pytest.fail("overlapping backend scope was accepted")

    async with conftest.isolated_backend("python"):
        callback: Final = object()
        litellm.callbacks.append(callback)
        with pytest.raises(RuntimeError, match="isolated_backend scopes cannot overlap across tasks"):
            await asyncio.create_task(competing_scope())
        assert configuration.rust_enabled() is False
        assert litellm.callbacks == [callback]
        async with conftest.isolated_backend("rust"):
            assert configuration.rust_enabled() is True
            assert litellm.callbacks == []
        assert configuration.rust_enabled() is False
        assert litellm.callbacks == [callback]
        with pytest.raises(RuntimeError, match="isolated_backend scopes cannot overlap across tasks"):
            await asyncio.create_task(competing_scope())

    async def subsequent_scope() -> None:
        async with conftest.isolated_backend("rust"):
            assert configuration.rust_enabled() is True

    await asyncio.create_task(subsequent_scope())


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
