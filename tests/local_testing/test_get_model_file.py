import os, sys, traceback
import importlib.resources
import json

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
import pytest


def test_get_model_cost_map():
    try:
        print(litellm.get_model_cost_map(url="fake-url"))
    except Exception as e:
        pytest.fail(f"An exception occurred: {e}")


def test_get_local_model_cost_map():
    """Test that load_local_model_cost_map returns a valid non-empty dict."""
    from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

    content = GetModelCostMap.load_local_model_cost_map()
    assert isinstance(content, dict)
    assert len(content) > 0
