import os, sys, traceback

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
import pytest

from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap


def test_get_model_cost_map():
    try:
        print(litellm.get_model_cost_map(url="fake-url"))
    except Exception as e:
        pytest.fail(f"An exception occurred: {e}")


def test_load_local_model_cost_map():
    content = GetModelCostMap.load_local_model_cost_map()
    assert (
        isinstance(content, dict) and content
    ), "load_local_model_cost_map must return a non-empty dict"
