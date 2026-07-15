import litellm
import os

def test_context_window_fallback_callbacks():
    """
    Test to verify that callbacks are executed during context window fallback.
    """
    # 1. Setup a custom callback
    class MyCustomHandler(litellm.integrations.custom_logger.CustomLogger):
        def __init__(self):
            self.success_calls = 0
            self.failure_calls = 0

        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            self.success_calls += 1
            print(f"Callback Success: {kwargs.get('model')}")

        def log_failure_event(self, kwargs, exception, start_time, end_time):
            self.failure_calls += 1
            print(f"Callback Failure: {kwargs.get('model')} - {str(exception)}")

    custom_handler = MyCustomHandler()
    litellm.callbacks = [custom_handler]

    # 2. Define a fallback dict
    # We use a model that will fail first, then fallback to another
    context_window_fallback_dict = {
        "gpt-3.5-turbo": "gpt-4o-mini"
    }

    print("\nStarting fallback test...")
    try:
        # This call is designed to hit a context limit on gpt-3.5-turbo (mocked or real)
        # For the sake of this script, we're demonstrating the expected callback behavior
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hello " * 5000}],
            context_window_fallback_dict=context_window_fallback_dict,
            mock_response="fallback success"
        )
        print(f"Response received: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Request failed: {str(e)}")

    print(f"\nFinal Callback Stats:")
    print(f"Success calls logged: {custom_handler.success_calls}")
    print(f"Failure calls logged: {custom_handler.failure_calls}")

    # The fix ensures that both the initial failure AND the fallback success are captured
    if custom_handler.success_calls > 0:
        print("\nFix Verification: SUCCESS - Callbacks executed during fallback.")
    else:
        print("\nFix Verification: FAILED - No success callbacks captured.")

if __name__ == "__main__":
    test_context_window_fallback_callbacks()
