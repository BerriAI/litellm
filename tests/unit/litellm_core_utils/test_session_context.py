import importlib
import sys
from typing import Final

import pytest

PI_AI_SESSION_ID: Final = "web:alice.smith:7f3e9c2a-5b1d-4e8f-9a6c-0d2b4f6e8a1c"


def test_resolver_imports_and_resolves_without_the_proxy_extras(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "fastapi", None)
    monkeypatch.delitem(sys.modules, "litellm.proxy.litellm_pre_call_utils", raising=False)
    monkeypatch.delitem(sys.modules, "litellm.litellm_core_utils.session_context", raising=False)

    session_context: Final = importlib.import_module("litellm.litellm_core_utils.session_context")

    context: Final = session_context.resolve_session_context({"x-session-affinity": PI_AI_SESSION_ID})
    assert (context.session_id, context.trace_id) == (PI_AI_SESSION_ID, PI_AI_SESSION_ID)
    assert (context.session_source, context.trace_source) == ("session_header", "session_header")
    assert "litellm.proxy.litellm_pre_call_utils" not in sys.modules
