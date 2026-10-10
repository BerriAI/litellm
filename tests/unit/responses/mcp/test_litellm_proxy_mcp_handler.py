import importlib
import sys
from typing import Final

import pytest

from litellm.proxy._experimental.mcp_server import litellm_proxy_mcp_handler as relocated
from litellm.responses.mcp import litellm_proxy_mcp_handler as legacy


def test_legacy_import_leaves_relocated_proxy_module_unloaded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, legacy.__name__)
    monkeypatch.delitem(sys.modules, relocated.__name__)
    monkeypatch.setattr(sys.modules["litellm.responses.mcp"], "litellm_proxy_mcp_handler", legacy)

    reimported: Final = importlib.import_module(legacy.__name__)

    assert reimported is not legacy
    assert relocated.__name__ not in sys.modules


@pytest.mark.parametrize("name", legacy.__all__)
def test_legacy_path_returns_the_relocated_objects(name: str) -> None:
    assert getattr(legacy, name) is getattr(relocated, name)


def test_legacy_path_still_raises_for_unknown_attributes() -> None:
    with pytest.raises(AttributeError):
        getattr(legacy, "NotAHandlerAttribute")
