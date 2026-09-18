from typing import Final

import pytest

from tests.test_litellm_rust.ocr.parity_todo import TODO


@pytest.fixture(autouse=True)
def parity_todo(request: pytest.FixtureRequest) -> None:
    reason: Final = TODO.get(request.node.name)
    if reason is not None:
        request.applymarker(pytest.mark.xfail(strict=True, reason=reason))
