from types import ModuleType

import pytest


@pytest.fixture
def native() -> ModuleType:
    return pytest.importorskip("litellm.rust_bridge._native")
