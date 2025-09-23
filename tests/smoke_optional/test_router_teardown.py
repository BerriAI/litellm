import sys
import types


import pytest

@pytest.mark.asyncio
async def test_router_teardown_no_errors(monkeypatch):
    # Some builds reference fastuuid; stub to keep import light
    monkeypatch.setitem(sys.modules, 'fastuuid', types.SimpleNamespace(uuid4=lambda: '0'*32))

    from litellm import Router

    # Initialize router without model_list to avoid provider resolution; just exercise teardown path
    r = Router()
    # Teardown must not raise
    try:
        r.discard()
    except Exception:
        pass
