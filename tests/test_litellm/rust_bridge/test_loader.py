from types import ModuleType

import pytest

from litellm.rust_bridge import loader


@pytest.fixture(autouse=True)
def _reset_loader() -> None:
    loader.reset_native_bridge_cache()
    yield
    loader.reset_native_bridge_cache()


def test_child_forked_after_the_extension_loaded_reports_it_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loader, "_cached_bridge", ModuleType("_native"))

    loader._disable_in_forked_child()

    assert loader.get_native_bridge() is None
    assert loader.native_bridge_available() is False


def test_child_forked_before_the_extension_loaded_keeps_it() -> None:
    loader._disable_in_forked_child()

    assert loader._forked_after_load is False
