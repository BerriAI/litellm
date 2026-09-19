from typing import Final

import pytest

from tests.test_litellm_rust.ocr.parity_todo import FUNCTION_TODO, TODO


@pytest.fixture(autouse=True)
def parity_todo(request: pytest.FixtureRequest) -> None:
    node: Final = request.node
    reason: Final = TODO.get(node.name) or FUNCTION_TODO.get(getattr(node, "originalname", node.name))
    if reason is not None:
        request.applymarker(pytest.mark.xfail(strict=True, reason=reason))
