from typing import Final

import litellm
import pytest

CALLBACK_LISTS: Final = (
    "callbacks",
    "success_callback",
    "failure_callback",
    "input_callback",
    "_async_success_callback",
    "_async_failure_callback",
    "_async_input_callback",
)


@pytest.fixture(autouse=True)
def isolate_litellm_callback_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in CALLBACK_LISTS:
        monkeypatch.setattr(litellm, name, list(getattr(litellm, name)))
