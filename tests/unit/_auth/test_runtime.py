from dataclasses import replace
from typing import Final

import pytest

from litellm._auth.decision import Deny, InspectKey, KeyBlocked, RequestFacts
from litellm._auth.runtime import get_auth_core
from tests.unit._auth.test_python_core import NOW, PLAIN_KEY, SETTINGS


@pytest.mark.parametrize("backend", (None, "rust", "python", "shadow"))
def test_default_and_native_modes_consume_their_decision_once(backend: str | None, monkeypatch: pytest.MonkeyPatch):
    if backend is None:
        monkeypatch.delenv("LITELLM_AUTH_CORE", raising=False)
    else:
        monkeypatch.setenv("LITELLM_AUTH_CORE", backend)
    step: Final = get_auth_core().start_resolved_key(RequestFacts("/chat/completions", "hash", True), SETTINGS, NOW)
    assert isinstance(step, InspectKey)
    blocked: Final = replace(PLAIN_KEY, blocked=True)
    assert step.resume(blocked) == Deny(KeyBlocked())
    with pytest.raises(RuntimeError, match="already consumed"):
        step.resume(blocked)
