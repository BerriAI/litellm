import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.cost_calculator import completion_cost


class TestVertexAIProvider(unittest.TestCase):
    def setUp(self):
        # Ensure vertex_ai is in the provider list for testing
        if "vertex_ai" not in [p.value for p in litellm.provider_list]:
            # This is a mock addition just for testing
            litellm.provider_list.append(litellm.types.utils.LlmProviders("vertex_ai"))

    def test_get_llm_provider_vertex_ai(self):
        """Test get_llm_provider correctly identifies 'vertex_ai/' prefix."""
        # Case with vertex_ai prefix
        model, provider, _, _ = get_llm_provider(model="vertex_ai/claude-3-7-sonnet@20250219")
        self.assertEqual(model, "claude-3-7-sonnet@20250219")
        self.assertEqual(provider, "vertex_ai")
        
    # No need to test model name formatting anymore - we're now using the original model name
        
    def test_end_to_end_cost_calculation(self):
        """Test the end-to-end cost calculation pipeline for vertex_ai models with date formats."""
        # Create a mock response object with usage information
        mock_response = MagicMock()
        mock_response.usage = {"prompt_tokens": 100, "completion_tokens": 50}
        
        # This is a simple test that verifies the cost calculation process doesn't fail 
        # and that the model name with @ symbol works correctly with the vertex_ai provider
        try:
            # We're not testing the actual cost values here, just that the pipeline executes without error
            completion_cost(
                completion_response=mock_response,
                model="vertex_ai/claude-3-sonnet@20240229",  # Use a model known to exist in the database
                custom_llm_provider="vertex_ai"
            )
            success = True
        except Exception as e:
            success = False
            # Avoid using print for linting reasons
            error_msg = f"Cost calculation failed: {str(e)}"
            self.fail(error_msg)
        
        self.assertTrue(success, "Cost calculation should complete without errors")


if __name__ == "__main__":
    unittest.main()
