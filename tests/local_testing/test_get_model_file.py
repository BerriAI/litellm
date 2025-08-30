import os
import sys
import json
import pytest

sys.path.insert(0, os.path.abspath("../.."))  # Adjust path to include your project root

import litellm
from litellm.litellm_core_utils.get_model_cost_map import load_local_backup


def test_get_model_cost_map():
    try:
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        os.environ["LITELLM_PRICE_DIR"] = "litellm"  # Ensure this is correct relative to the root

        result = litellm.get_model_cost_map(url="fake-url")
        print("Result from fallback:", result)
        assert isinstance(result, dict)
    except Exception as e:
        pytest.fail(f"An exception occurred: {e}")


def test_get_backup_model_cost_map():
    try:
        os.environ["LITELLM_PRICE_DIR"] = "litellm"  # Ensure this path is correct
        content = load_local_backup()
        print("Loaded backup content:", content)
        assert isinstance(content, dict)
    except Exception as e:
        pytest.fail(f"Failed to load backup model cost map: {e}")

