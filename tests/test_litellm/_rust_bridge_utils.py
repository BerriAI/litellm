from types import SimpleNamespace
from typing import Final

import pytest

from litellm.rust_bridge import bindings


def use_fake_native_bridge(monkeypatch: pytest.MonkeyPatch, **exports: object) -> None:
    native: Final = SimpleNamespace(**exports)
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: native)
