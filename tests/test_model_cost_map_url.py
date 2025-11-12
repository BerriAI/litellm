import importlib
import sys


def test_model_cost_map_url_from_env(monkeypatch):
    """Ensure `LITELLM_MODEL_COST_MAP_URL` env var is picked up on import and used by get_model_cost_map."""
    test_url = "https://example.com/test_model_cost_map.json"

    # A minimal model cost map we expect to be loaded
    model_json = {
        "my-test-model": {
            "input_cost_per_token": 0.123,
            "output_cost_per_token": 0.456,
            "litellm_provider": "openai",
            "mode": "chat",
        }
    }

    class DummyResp:
        def raise_for_status(self):
            return None

        def json(self):
            return model_json

    # Point litellm at our test URL
    monkeypatch.setenv("LITELLM_MODEL_COST_MAP_URL", test_url)

    # Mock httpx.get to return our dummy response
    import httpx

    monkeypatch.setattr(httpx, "get", lambda url, timeout=5: DummyResp())

    # Reload the litellm package so top-level import picks up the env var
    if "litellm" in sys.modules:
        importlib.reload(sys.modules["litellm"])
    else:
        import litellm  # noqa: F401
        importlib.reload(litellm)

    import litellm as ll  # re-import for assertions

    # The package should have picked up the env var and loaded our model map
    assert getattr(ll, "model_cost_map_url") == test_url
    assert "my-test-model" in ll.model_cost
    assert ll.model_cost["my-test-model"]["input_cost_per_token"] == 0.123
