"""
Test Azure OpenAI Assistant Features Cost Tracking

Tests cost calculation for Azure's new assistant features:
- File Search (storage-based pricing)
- Code Interpreter (session-based pricing) 
- Computer Use (token-based pricing)
- Vector Store (storage-based pricing)
"""
import os
import pytest
from litellm.litellm_core_utils.llm_cost_calc.tool_call_cost_tracking import (
    StandardBuiltInToolCostTracking,
)
from litellm.constants import (
    AZURE_FILE_SEARCH_COST_PER_GB_PER_DAY,
    AZURE_COMPUTER_USE_INPUT_COST_PER_1K_TOKENS,
    AZURE_COMPUTER_USE_OUTPUT_COST_PER_1K_TOKENS,
    AZURE_VECTOR_STORE_COST_PER_GB_PER_DAY,
)
import litellm


class TestAzureAssistantCostTracking:
    """Test suite for Azure assistant features cost tracking."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test environment to use local model cost map."""
        # Force use of local model cost map for CI/CD consistency
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")
        
        yield
        
        # Cleanup not strictly necessary but good practice
        # Don't delete env var as other tests might need it

    def test_azure_file_search_cost_calculation(self):
        """Test Azure file search cost calculation with storage-based pricing."""
        # Test with 1.5 GB for 30 days
        cost = StandardBuiltInToolCostTracking.get_cost_for_file_search(
            file_search={},
            provider="azure",
            storage_gb=1.5,
            days=30,
        )
        expected_cost = 1.5 * 30 * AZURE_FILE_SEARCH_COST_PER_GB_PER_DAY  # $4.50
        assert cost == expected_cost, f"Expected {expected_cost}, got {cost}"

    def test_azure_file_search_no_storage_info(self):
        """Test Azure file search returns 0 when no storage info provided."""
        cost = StandardBuiltInToolCostTracking.get_cost_for_file_search(
            file_search={},
            provider="azure",
        )
        assert cost == 0.0, "Should return 0 when no storage info provided"

    def test_openai_file_search_unchanged(self):
        """Test OpenAI file search pricing remains unchanged."""
        from litellm.constants import OPENAI_FILE_SEARCH_COST_PER_1K_CALLS
        
        cost = StandardBuiltInToolCostTracking.get_cost_for_file_search(
            file_search={},
            provider="openai",
        )
        assert cost == OPENAI_FILE_SEARCH_COST_PER_1K_CALLS

    def test_azure_code_interpreter_cost_calculation(self):
        """Test Azure code interpreter cost calculation."""
        # Test with 5 sessions
        cost = StandardBuiltInToolCostTracking.get_cost_for_code_interpreter(
            sessions=5,
            provider="azure",
        )
        # Read expected cost from model cost map (azure/container)
        azure_container_info = litellm.model_cost.get("azure/container", {})
        cost_per_session = azure_container_info.get("code_interpreter_cost_per_session", 0.03)
        expected_cost = 5 * cost_per_session  # $0.15
        assert cost == expected_cost, f"Expected {expected_cost}, got {cost}"

    def test_azure_code_interpreter_zero_sessions(self):
        """Test Azure code interpreter with zero sessions."""
        cost = StandardBuiltInToolCostTracking.get_cost_for_code_interpreter(
            sessions=0,
            provider="azure",
        )
        assert cost == 0.0, "Should return 0 for zero sessions"

    def test_openai_code_interpreter_free(self):
        """Test OpenAI code interpreter cost from model cost map."""
        cost = StandardBuiltInToolCostTracking.get_cost_for_code_interpreter(
            sessions=5,
            provider="openai",
        )
        assert cost == 0.15, "OpenAI code interpreter should return 0.15 based on current implementation"

    @pytest.mark.parametrize("input_tokens,output_tokens,expected_cost", [
        (1000, 500, 1000/1000 * AZURE_COMPUTER_USE_INPUT_COST_PER_1K_TOKENS + 500/1000 * AZURE_COMPUTER_USE_OUTPUT_COST_PER_1K_TOKENS),  # $0.009
        (2000, 0, 2000/1000 * AZURE_COMPUTER_USE_INPUT_COST_PER_1K_TOKENS),  # $0.006
        (0, 1000, 1000/1000 * AZURE_COMPUTER_USE_OUTPUT_COST_PER_1K_TOKENS),  # $0.012
        (0, 0, 0.0),  # $0.000
    ])
    def test_azure_computer_use_cost_calculation(self, input_tokens, output_tokens, expected_cost):
        """Test Azure computer use cost calculation with various token combinations."""
        cost = StandardBuiltInToolCostTracking.get_cost_for_computer_use(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider="azure",
        )
        assert abs(cost - expected_cost) < 0.0001, f"Expected {expected_cost}, got {cost}"

    def test_openai_computer_use_free(self):
        """Test OpenAI computer use has no separate charges."""
        cost = StandardBuiltInToolCostTracking.get_cost_for_computer_use(
            input_tokens=1000,
            output_tokens=500,
            provider="openai",
        )
        assert cost == 0.0, "OpenAI should not charge separately for computer use"

    def test_azure_vector_store_cost_calculation(self):
        """Test Azure vector store cost calculation."""
        vector_store_usage = {
            "storage_gb": 2.0,
            "days": 15,
        }
        cost = StandardBuiltInToolCostTracking.get_cost_for_vector_store(
            vector_store_usage=vector_store_usage,
            provider="azure",
        )
        expected_cost = 2.0 * 15 * AZURE_VECTOR_STORE_COST_PER_GB_PER_DAY  # $3.00
        assert cost == expected_cost, f"Expected {expected_cost}, got {cost}"

    def test_openai_vector_store_free(self):
        """Test OpenAI vector store has no separate charges."""
        vector_store_usage = {
            "storage_gb": 2.0,
            "days": 15,
        }
        cost = StandardBuiltInToolCostTracking.get_cost_for_vector_store(
            vector_store_usage=vector_store_usage,
            provider="openai",
        )
        assert cost == 0.0, "OpenAI should not charge separately for vector store"

    def test_model_specific_pricing_overrides(self):
        """Test model-specific pricing overrides from JSON config."""
        # Test file search with model-specific pricing
        model_info = {
            "file_search_cost_per_gb_per_day": 0.2,  # Custom pricing
        }
        cost = StandardBuiltInToolCostTracking.get_cost_for_file_search(
            file_search={},
            provider="azure",
            model_info=model_info,
            storage_gb=1.0,
            days=10,
        )
        expected_cost = 1.0 * 10 * 0.2  # $2.00
        assert cost == expected_cost, f"Expected {expected_cost}, got {cost}"

        # Test computer use with model-specific pricing
        model_info = {
            "computer_use_input_cost_per_1k_tokens": 5.0,  # Custom pricing
            "computer_use_output_cost_per_1k_tokens": 15.0,  # Custom pricing
        }
        cost = StandardBuiltInToolCostTracking.get_cost_for_computer_use(
            input_tokens=1000,
            output_tokens=500,
            provider="azure",
            model_info=model_info,
        )
        expected_cost = 1000/1000 * 5.0 + 500/1000 * 15.0  # $12.50
        assert cost == expected_cost, f"Expected {expected_cost}, got {cost}"

        # Test code interpreter with model-specific pricing
        model_info = {
            "code_interpreter_cost_per_session": 0.05,  # Custom pricing
        }
        cost = StandardBuiltInToolCostTracking.get_cost_for_code_interpreter(
            sessions=3,
            provider="azure",
            model_info=model_info,
        )
        expected_cost = 3 * 0.05  # $0.15
        assert cost == expected_cost, f"Expected {expected_cost}, got {cost}"

    def test_none_inputs_return_zero(self):
        """Test that None inputs return zero cost."""
        assert StandardBuiltInToolCostTracking.get_cost_for_file_search(None) == 0.0
        assert StandardBuiltInToolCostTracking.get_cost_for_code_interpreter(None) == 0.0
        assert StandardBuiltInToolCostTracking.get_cost_for_computer_use(None, None) == 0.0
        assert StandardBuiltInToolCostTracking.get_cost_for_vector_store(None) == 0.0

    def test_constants_loaded_correctly(self):
        """Test that Azure pricing constants are loaded with expected values."""
        assert AZURE_FILE_SEARCH_COST_PER_GB_PER_DAY == 0.1
        
        # Code interpreter cost is now in model cost map
        azure_container_info = litellm.model_cost.get("azure/container", {})
        assert azure_container_info.get("code_interpreter_cost_per_session") == 0.03
        
        assert AZURE_COMPUTER_USE_INPUT_COST_PER_1K_TOKENS == 3.0
        assert AZURE_COMPUTER_USE_OUTPUT_COST_PER_1K_TOKENS == 12.0
        assert AZURE_VECTOR_STORE_COST_PER_GB_PER_DAY == 0.1