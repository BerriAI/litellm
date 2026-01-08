"""
Test to reproduce issue #18216: Custom litellm_params in config.yaml not passed to custom_handler.py

This test verifies that custom parameters defined in litellm_params are passed through to custom handlers.
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import Any, Optional, Union, Callable

import litellm
from litellm import CustomLLM, ModelResponse
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.litellm_core_utils.get_litellm_params import get_litellm_params
import httpx


class TestCustomLLM(CustomLLM):
    """A test custom LLM handler that captures litellm_params for inspection."""

    captured_litellm_params: Optional[dict] = None

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[HTTPHandler] = None,
    ):
        # Capture the litellm_params for inspection
        TestCustomLLM.captured_litellm_params = litellm_params
        print(f"\n🔍 Captured litellm_params: {litellm_params}")

        # Return a mock response with proper structure
        from litellm.types.utils import Choices, Message

        model_response.choices = [
            Choices(
                message=Message(role="assistant", content="Test response"),
                index=0,
                finish_reason="stop",
            )
        ]
        model_response.model = model
        return model_response


class TestCustomParamsBug18216:
    """Test suite to reproduce and verify the custom params bug."""

    def test_get_litellm_params_drops_custom_params(self):
        """
        Test that demonstrates get_litellm_params() drops unknown/custom parameters.

        This is the core of issue #18216 - custom params like 'context_id' are passed
        in via kwargs but are NOT included in the returned dictionary.
        """
        # Simulate params that would come from config.yaml litellm_params
        result = get_litellm_params(
            api_key="test-key",
            custom_llm_provider="my-custom-provider",
            metadata={"user": "test"},
            # These are custom params that a user might define in config.yaml
            context_id="gemini-2.5-pro",  # Custom param from config
            use_case="my-use-case",  # Another custom param
            rpm=1,  # This is actually a known param but let's check
        )

        print(f"\n📋 Result from get_litellm_params():")
        print(f"   api_key: {result.get('api_key')}")
        print(f"   custom_llm_provider: {result.get('custom_llm_provider')}")
        print(f"   metadata: {result.get('metadata')}")
        print(f"   context_id: {result.get('context_id')}")
        print(f"   use_case: {result.get('use_case')}")

        # These should be present (known params)
        assert result.get("api_key") == "test-key", "api_key should be present"
        assert (
            result.get("custom_llm_provider") == "my-custom-provider"
        ), "custom_llm_provider should be present"
        assert result.get("metadata") == {"user": "test"}, "metadata should be present"

        # AFTER FIX: These custom params should now be present!
        context_id = result.get("context_id")
        use_case = result.get("use_case")

        print(f"\n✅ FIX VERIFICATION:")
        print(f"   context_id in result: {context_id is not None}")
        print(f"   use_case in result: {use_case is not None}")

        # These assertions verify the fix works
        assert context_id == "gemini-2.5-pro", "context_id should be preserved after fix"
        assert use_case == "my-use-case", "use_case should be preserved after fix"
        
        print("\n🎉 SUCCESS: Custom params are now correctly passed through!")

    def test_custom_handler_receives_incomplete_params(self):
        """
        End-to-end test showing that custom handlers don't receive custom params.
        """
        # Register our test custom handler
        my_custom_llm = TestCustomLLM()
        litellm.custom_provider_map = [
            {"provider": "test-custom-provider", "custom_handler": my_custom_llm}
        ]
        litellm._custom_providers.append("test-custom-provider")

        try:
            # Make a completion call with custom params
            # In real usage, these would come from config.yaml
            response = litellm.completion(
                model="test-custom-provider/test-model",
                messages=[{"role": "user", "content": "Hello"}],
                # Custom params that should be passed through
                context_id="my-context-id",
                use_case="my-use-case",
            )

            # Check what the handler received
            captured = TestCustomLLM.captured_litellm_params
            print(f"\n📦 Handler received litellm_params: {captured}")

            assert captured is not None, "Handler should have received litellm_params"
            
            context_id = captured.get("context_id")
            use_case = captured.get("use_case")

            print(f"\n🔍 Checking for custom params in handler:")
            print(f"   context_id: {context_id}")
            print(f"   use_case: {use_case}")

            # AFTER FIX: These should now be present
            assert context_id == "my-context-id", "context_id should be passed to custom handler"
            assert use_case == "my-use-case", "use_case should be passed to custom handler"
            
            print("\n🎉 SUCCESS: Custom handler received all custom params!")

        except Exception as e:
            print(f"\n⚠️ Error during test: {e}")
            # The test might fail for other reasons, but we still want to see the params issue
            if TestCustomLLM.captured_litellm_params:
                print(f"   Captured params before error: {TestCustomLLM.captured_litellm_params}")
            raise

        finally:
            # Cleanup
            litellm.custom_provider_map = []
            if "test-custom-provider" in litellm._custom_providers:
                litellm._custom_providers.remove("test-custom-provider")


if __name__ == "__main__":
    # Run the tests directly
    test = TestCustomParamsBug18216()

    print("=" * 60)
    print("TEST 1: get_litellm_params drops custom params")
    print("=" * 60)
    test.test_get_litellm_params_drops_custom_params()

    print("\n" + "=" * 60)
    print("TEST 2: Custom handler receives incomplete params")
    print("=" * 60)
    test.test_custom_handler_receives_incomplete_params()
