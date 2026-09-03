import os, sys, traceback
import importlib.resources
import json

import litellm
import pytest


def test_get_model_cost_map():
    try:
        print(litellm.get_model_cost_map(url="fake-url"))
    except Exception as e:
        pytest.fail(f"An exception occurred: {e}")


def test_get_backup_model_cost_map():
    with importlib.resources.open_text(
        "litellm", "model_prices_and_context_window_backup.json"
    ) as f:
        print("inside backup")
        content = json.load(f)
        print("content", content)
