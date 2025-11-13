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
    """Test that we can load the local model cost map"""
    from pathlib import Path

    # Test loading from project root (development scenario)
    project_root = Path(__file__).parent.parent.parent
    model_cost_map_path = project_root / "model_prices_and_context_window.json"

    with open(model_cost_map_path, "r") as f:
        print("inside local model cost map")
        content = json.load(f)
        print("content", content)
        assert content is not None
        assert len(content) > 0
