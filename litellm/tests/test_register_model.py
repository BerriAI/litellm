#### What this tests ####
#    This tests calling batch_completions by running 100 messages together

import sys, os
import traceback
import pytest
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm

def test_update_model_cost():
    try: 
        litellm.register_model({
             "gpt-4": {
                "max_tokens": 8192, 
                "input_cost_per_token": 0.00002, 
                "output_cost_per_token": 0.00006, 
                "litellm_provider": "openai", 
                "mode": "chat"
            },
        })
        assert litellm.model_cost["gpt-4"]["input_cost_per_token"] == 0.00002
    except Exception as e:
        pytest.fail(f"An error occurred: {e}")

# test_update_model_cost()

def test_update_model_cost_map_url(): 
    try: 
        litellm.register_model(model_cost="https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json")
        assert litellm.model_cost["gpt-4"]["input_cost_per_token"] == 0.00003
    except Exception as e:
        pytest.fail(f"An error occurred: {e}")

test_update_model_cost_map_url()