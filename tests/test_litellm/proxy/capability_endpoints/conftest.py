"""Per-test isolation for the capability_endpoints module cache."""
import pytest


@pytest.fixture(autouse=True)
def _reset_capability_cache(monkeypatch):
    """Wipe the module-level DualCache between tests so cache hits in test N
    don't leak into test N+1.

    Without this the autouse fixture in TestCapabilitiesCache only protects its
    own class; the function-level tests above it would all share one cache.
    """
    import litellm.proxy.capability_endpoints.capability_endpoints as mod

    mod._capabilities_cache = None
    yield
    mod._capabilities_cache = None
