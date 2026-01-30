import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import asyncio
import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.types.utils import CallTypes, StandardLoggingGuardrailInformation
import time

# Mock Guardrail that fails
class FailGuardrail(CustomGuardrail):
    def __init__(self):
        super().__init__()

    async def async_guardrail_pre_call(self, context):
        # Simulate detection
        return {
            "allowed": False, 
            "message": "Violated guardrail", 
            "metadata": {"guardrail": "fail_guardrail"}
        }

async def test_guardrail_logging_on_failure():
    """
    Test that applied_guardrails is present in logging metadata when a guardrail fails.
    """
    # 1. Setup Mock Logging Callback
    log_queue = []
    def mock_failure_callback(kwargs, exception_obj, start_time, end_time):
        print(f"Mock Failure Callback Triggered. kwargs keys: {kwargs.keys()}")
        log_queue.append(kwargs)

    litellm.failure_callback = [mock_failure_callback]
    
    # 2. Inject Mock Guardrail
    litellm.guardrail_list = ["fail_guardrail"]
    # We need to register the class so litellm can find it, 
    # but litellm uses string mapping. 
    # For unit testing, it's easier to mock the `run_pre_call_guardrails` or relevant function,
    # OR simpler: use `litellm.completion` with a mocked guardrail flow if possible.
    # But `CustomGuardrail` integration usually runs in Proxy.
    
    # Let's try to simulate what `proxy_server.py` does.
    # It calls `litellm.completion`.
    
    # Actually, simpler reproduction:
    # `litellm.completion` handles guardrails if configured.
    # But the issue specifies "LiteLLM Guardrail detect something as pre_call".
    
    # Let's verify if `litellm.completion` captures it.
    
    # To properly simulate "pre_call" failure from a guardrail:
    # We can mock `litellm.utils.get_llm_provider` or similar to fail? 
    # No, guardrails run before that.
    
    # 3. Define the test
    try:
        # Mocking the guardrail execution to simulate a failure
        # We'll use a side_effect on a known function if possible, but 
        # litellm's guardrail system initializes classes. 
        
        # Alternative: We can manually invoke the failure flow that happens when a guardrail fails.
        # When a guardrail fails, it raises an exception (often ContextWindowExceededError or generic).
        # We need to ensure that when this exception is raised, the logging object has 'applied_guardrails'.
        
        from litellm.utils import CustomStreamWrapper
        from litellm import ModelResponse
        
        # Create a mock logging object
        from litellm.litellm_core_utils.litellm_logging import Logging
        import time
        from litellm.types.utils import CallTypes
        
        logging_obj = Logging(
            model="gpt-3.5-turbo", 
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
            call_type=CallTypes.completion,
            start_time=time.time(),
            litellm_call_id="test-call-id",
            function_id="test-func-id",
        )
        
        # Simulate guardrail application via metadata (this is what CustomGuardrail does)
        # logging_obj.applied_guardrails = ["test_guardrail"] # Removed to test extraction from metadata
        
        # Ensure metadata exists
        if not hasattr(logging_obj, "model_call_details") or "metadata" not in logging_obj.model_call_details:
             logging_obj.model_call_details["metadata"] = {}
             
        # Simulate CustomGuardrail adding info to metadata
        from litellm.types.utils import StandardLoggingGuardrailInformation
        # We need to construct the object or dict as CustomGuardrail does
        # CustomGuardrail adds StandardLoggingGuardrailInformation objects
        slg = StandardLoggingGuardrailInformation(
             guardrail_name="test_guardrail",
             guardrail_status="failure",
             duration=0.1
        )
        if "standard_logging_guardrail_information" not in logging_obj.model_call_details["metadata"]:
             logging_obj.model_call_details["metadata"]["standard_logging_guardrail_information"] = []
        logging_obj.model_call_details["metadata"]["standard_logging_guardrail_information"].append(slg)
        
        logging_obj.guardrail_information = [{"guardrail": "test_guardrail", "duration": 0.1}]
        
        # Simulate failure
        exception = Exception("Guardrail failed")
        
        # Trigger failure callback manually to check what it receives
        # In the real code, litellm catches the exception and calls failure_callback.
        # The key is whether 'applied_guardrails' from logging_obj is passed to the callback kwargs.
        
        # Let's inspect `litellm_logging.py`'s `failure_callback` handling.
        # But for reproduction, let's assume we call `completion` and it fails.
        
        # For now, let's assume we need to verify if `applied_guardrails` is preserved in the kwargs passed to failure_callback.
        
        # Let's mock a failure in `completion` where guardrails were applied.
        kwargs = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "hi"}],
            "litellm_logging_obj": logging_obj
        }
        
        # Manually invoke the callback logic to see what it extracts
        logging_obj.failure_handler(
            exception=exception,
            traceback_exception="Traceback...",
            start_time=asyncio.get_event_loop().time(),
            end_time=asyncio.get_event_loop().time()
        )
        
        # Check model_call_details directly
        print(f"Model Call Details keys: {logging_obj.model_call_details.keys()}")
        if "applied_guardrails" in logging_obj.model_call_details:
             print(f"Applied Guardrails: {logging_obj.model_call_details['applied_guardrails']}")
        else:
             print("applied_guardrails NOT in model_call_details")

        # ASSERTION
        assert "applied_guardrails" in logging_obj.model_call_details, "applied_guardrails missing from model_call_details"
        assert logging_obj.model_call_details["applied_guardrails"] == ["test_guardrail"]
        
    except Exception as e:
        raise e

if __name__ == "__main__":
    import sys
    try:
        asyncio.run(test_guardrail_logging_on_failure())
        print("Test Passed!")
    except Exception as e:
        print(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
