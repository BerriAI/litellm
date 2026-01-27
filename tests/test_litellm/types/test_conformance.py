"""
Provider-agnostic conformance tests for Usage validation.

These tests ensure that regardless of which provider returns data,
LiteLLM enforces a consistent schema at the ingestion boundary.
"""
import pytest
from pydantic import ValidationError
from litellm.types.utils import Usage, ModelResponse, UsageAdapter


class TestUsageConformance:
    """Test that Usage validation catches all schema violations."""

    # Valid Input Conformance

    def test_valid_usage_dict_complete(self):
        """Complete dict with all required fields should pass."""
        data = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        usage = UsageAdapter.validate_python(data)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30

    def test_valid_usage_dict_with_optional_fields(self):
        """Dict with optional fields should pass and map correctly."""
        data = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "reasoning_tokens": 5,
            "cost": 0.0001,
        }
        usage = UsageAdapter.validate_python(data)

        # Check Cost (Usage class keeps it if not None)
        assert usage.cost == 0.0001

        # Check Reasoning Tokens (Mapped to completion_tokens_details by Usage class)
        assert usage.completion_tokens_details.reasoning_tokens == 5

    def test_valid_usage_object(self):
        """Pre-constructed Usage object should pass."""
        usage_obj = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        usage = UsageAdapter.validate_python(usage_obj)
        assert usage.prompt_tokens == 10

    #  Type Violation Detection

    def test_modelresponse_rejects_invalid_usage_types(self):
        """
        Ensures ModelResponse throws an error if usage is a string/int.
        This closes the 'else: usage = usage' problem.
        """
        invalid_inputs = [
            "100 tokens",  # String
            123,  # Integer
            ["tokens"],  # List
            object(),  # Random object
        ]

        for bad_input in invalid_inputs:
            # This MUST raise ValueError now (due to our try/except block in utils.py)
            with pytest.raises(ValueError, match="Invalid 'usage' object"):
                ModelResponse(id="test", usage=bad_input)

    #  Adapter Unit Tests

    def test_adapter_coerces_valid_strings(self):
        """Pydantic standard behavior: string ints ('15') are coerced to ints (15)."""
        data = {"prompt_tokens": "15", "completion_tokens": 25, "total_tokens": 40}
        usage = UsageAdapter.validate_python(data)
        assert usage.prompt_tokens == 15  # Coercion successful
        assert isinstance(usage.prompt_tokens, int)

    def test_adapter_rejects_invalid_strings(self):
        """Adapter should fail if string cannot be coerced to int."""
        data = {
            "prompt_tokens": "ten",  # Invalid int
            "completion_tokens": 25,
            "total_tokens": 40,
        }
        with pytest.raises(ValidationError) as exc_info:
            UsageAdapter.validate_python(data)

        assert "prompt_tokens" in str(exc_info.value)

    def test_usage_defaults_for_missing_fields(self):
        """
        Usage class allows missing fields (defaults to 0).
        This test confirms we didn't break that behavior.
        """
        data = {"prompt_tokens": 10}  # Missing others
        usage = UsageAdapter.validate_python(data)

        # Usage.__init__ sets defaults to 0
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_cost_field_none_behavior(self):
        """
        Usage.__init__ deletes 'cost' attribute if it is None.
        We verify this behavior is preserved.
        """
        data = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "cost": None,
        }
        usage = UsageAdapter.validate_python(data)

        # Usage logic: if cost is None, del self.cost
        assert not hasattr(usage, "cost")
