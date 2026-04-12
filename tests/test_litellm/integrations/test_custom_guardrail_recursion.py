import pytest
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.guardrails import GuardrailEventHooks
import json


class TestCustomGuardrailRecursion:
    """
    Specific tests for the circular reference / RecursionError fix in logging.
    """

    def test_log_guardrail_information_handles_circular_references(self):
        """
        Test that add_standard_logging method sanitizes input data containing circular references
        instead of crashing.

        This reproduces the Langfuse crash scenario:
        Request -> Metadata -> GuardrailResponse -> DebugContext -> Request
        """
        guardrail = CustomGuardrail(
            guardrail_name="recursion_test_guardrail",
            event_hook=GuardrailEventHooks.pre_call,
        )

        # 1. Setup Circular Data
        request_data = {"user_id": "test_recursive_user"}
        metadata = {"session_id": "123"}
        request_data["metadata"] = metadata

        # Create the danger: Guardrail Response holding a reference back to request_data
        dirty_response = {
            "flagged": False,
            "debug_context": request_data,  # <--- ACCESS TO ROOT (Circular Ref)
        }

        # 2. Invoke the logging method
        # If the fix is working, this will NOT raise RecursionError
        try:
            guardrail.add_standard_logging_guardrail_information_to_request_data(
                guardrail_json_response=dirty_response,
                request_data=request_data,
                guardrail_status="success",
                start_time=1.0,
                end_time=2.0,
                duration=1.0,
                masked_entity_count={},
                event_type=GuardrailEventHooks.pre_call,
            )
        except RecursionError:
            pytest.fail(
                "RecursionError raised! The cyclic reference sanitization failed."
            )

        # 3. Verify the data stored is safe
        stored_info = request_data["metadata"][
            "standard_logging_guardrail_information"
        ][0]
        stored_response = stored_info["guardrail_response"]

        # Check that we can dump it to JSON without crashing (Ultimate proof)
        try:
            json.dumps(stored_response)
        except Exception as e:
            pytest.fail(f"Stored data is not JSON serializable: {e}")

        # Check content - keys should be preserved but recursion broken
        assert "debug_context" in stored_response
        debug_context = stored_response["debug_context"]

        # In a sanitized copy, the nested metadata should be a copy, not the original live dict
        assert debug_context["user_id"] == "test_recursive_user"
        # The 'metadata' inside 'debug_context' would be where recursion stops or is filtered
        assert "metadata" in debug_context
