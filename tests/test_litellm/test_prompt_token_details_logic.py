
import pytest
from litellm.types.utils import Usage, PromptTokensDetailsWrapper

def test_usage_optimizer_prompt_tokens_details_wrapper():
    """
    Test to verify that if PromptTokensDetailsWrapper is passed to Usage,
    it is used directly and not re-instantiated (cloned).
    
    Issue #19927: The order of isinstance checks caused this to fail (it re-created the object).
    """
    
    # 1. Create a wrapper instance
    original_wrapper = PromptTokensDetailsWrapper(
        text_tokens=10, 
        image_tokens=5, 
        cached_tokens=0,
        audio_tokens=0
    )
    
    # 2. Pass it to Usage
    usage = Usage(prompt_tokens_details=original_wrapper)
    
    # 3. Assert Identity (IS the same object)
    # The current buggy implementation will clone it so 'is' check will fail.
    # We assert strict identity to prove we aren't creating garbage.
    
    # NOTE: Initially this should FAIL if the bug exists.
    assert usage.prompt_tokens_details is original_wrapper
