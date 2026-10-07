from types import ModuleType

import pytest


@pytest.fixture
def native() -> ModuleType:
    module: ModuleType = pytest.importorskip("litellm.rust_bridge._native")  # pyright: ignore[reportAny]  # importorskip has no typed return
    return module
