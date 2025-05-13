import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system paths

from base_llm_unit_tests import BaseLLMChatTest
from litellm.llms.vertex_ai.context_caching.transformation import (
    separate_cached_messages,
)
import litellm
from litellm import completion

class TestGoogleAIStudioGemini(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        return {"model": "gemini/gemini-2.0-flash"}
    
    def get_base_completion_call_args_with_reasoning_model(self) -> dict:
        return {"model": "gemini/gemini-2.5-flash-preview-04-17"}

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
        from litellm.litellm_core_utils.prompt_templates.factory import (
            convert_to_gemini_tool_call_invoke,
        )

        result = convert_to_gemini_tool_call_invoke(tool_call_no_arguments)
        print(result)


def test_gemini_context_caching_separate_messages():
    messages = [
        # System Message
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Here is the full text of a complex legal agreement" * 400,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        # marked for caching with the cache_control parameter, so that this checkpoint can read from the previous cache.
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What are the key terms and conditions in this agreement?",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "assistant",
            "content": "Certainly! the key terms and conditions are the following: the contract is 1 year long for $10/mo",
        },
        # The final turn is marked with cache-control, for continuing in followups.
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What are the key terms and conditions in this agreement?",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
    ]
    cached_messages, non_cached_messages = separate_cached_messages(messages)
    print(cached_messages)
    print(non_cached_messages)
    assert len(cached_messages) > 0, "Cached messages should be present"
    assert len(non_cached_messages) > 0, "Non-cached messages should be present"


def test_gemini_image_generation():
    # litellm._turn_on_debug()
    response = completion(
        model="gemini/gemini-2.0-flash-exp-image-generation",
        messages=[{"role": "user", "content": "Generate an image of a cat"}],
        modalities=["image", "text"],
    )
    assert response.choices[0].message.content is not None



def test_gemini_thinking():
    litellm._turn_on_debug()
    from litellm.types.utils import Message, CallTypes
    from litellm.utils import return_raw_request
    import json

    messages = [
        {"role": "user", "content": "Explain the concept of Occam's Razor and provide a simple, everyday example"}
    ]
    reasoning_content = "I'm thinking about Occam's Razor."
    assistant_message = Message(content='Okay, let\'s break down Occam\'s Razor.', reasoning_content=reasoning_content, role='assistant', tool_calls=None, function_call=None, provider_specific_fields=None)

    messages.append(assistant_message)

    raw_request = return_raw_request(
        endpoint=CallTypes.completion,
        kwargs={
            "model": "gemini/gemini-2.5-flash-preview-04-17",
            "messages": messages,
        }
    )
    assert reasoning_content in json.dumps(raw_request)
    response = completion(
        model="gemini/gemini-2.5-flash-preview-04-17",
        messages=messages, # make sure call works
    )
    print(response.choices[0].message)
    assert response.choices[0].message.content is not None


def test_gemini_thinking_budget_0():
    litellm._turn_on_debug()
    from litellm.types.utils import Message, CallTypes
    from litellm.utils import return_raw_request
    import json

    raw_request = return_raw_request(
        endpoint=CallTypes.completion,
        kwargs={
            "model": "gemini/gemini-2.5-flash-preview-04-17",
            "messages": [{"role": "user", "content": "Explain the concept of Occam's Razor and provide a simple, everyday example"}],
            "thinking": {"type": "enabled", "budget_tokens": 0}
        }
    )
    print(raw_request)
    assert "0" in json.dumps(raw_request["raw_request_body"])


def test_gemini_finish_reason():
    import os
    from litellm import completion
    litellm._turn_on_debug()
    response = completion(model="gemini/gemini-1.5-pro", messages=[{"role": "user", "content": "give me 3 random words"}], max_tokens=2)
    print(response)
    assert response.choices[0].finish_reason is not None
    assert response.choices[0].finish_reason == "length"