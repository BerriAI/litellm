from typing import Final

import pytest

from litellm.proxy._experimental.mcp_server import litellm_proxy_mcp_handler as relocated
from litellm.responses.mcp import litellm_proxy_mcp_handler as legacy
from tests.test_litellm_rust.support.child_interpreter import run_child_interpreter


def test_legacy_import_leaves_relocated_proxy_module_unloaded() -> None:
    result: Final = run_child_interpreter(
        "import sys, litellm.responses.mcp.litellm_proxy_mcp_handler\n"
        "print('litellm.proxy._experimental.mcp_server.litellm_proxy_mcp_handler' in sys.modules)",
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


@pytest.mark.parametrize("name", legacy.__all__)
def test_legacy_path_returns_the_relocated_objects(name: str) -> None:
    assert getattr(legacy, name) is getattr(relocated, name)


def test_legacy_path_still_raises_for_unknown_attributes() -> None:
    with pytest.raises(AttributeError):
        getattr(legacy, "NotAHandlerAttribute")
