import json
from importlib.resources import files

import pytest


@pytest.fixture(scope="module")
def local_model_cost_map():
    """Pin litellm.model_cost to the bundled map so entries added in this repo are visible.

    litellm fetches the map remotely by default, so a test asserting on a freshly added
    entry would otherwise pass or fail depending on what main has published.
    """
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    import litellm
    from litellm.utils import _invalidate_model_cost_lowercase_map

    original_model_cost = litellm.model_cost
    litellm.model_cost = json.loads(
        files("litellm").joinpath("model_prices_and_context_window_backup.json").read_text(encoding="utf-8")
    )
    litellm.get_model_info.cache_clear()
    _invalidate_model_cost_lowercase_map()
    try:
        yield litellm
    finally:
        litellm.model_cost = original_model_cost
        litellm.get_model_info.cache_clear()
        _invalidate_model_cost_lowercase_map()
        monkeypatch.undo()
