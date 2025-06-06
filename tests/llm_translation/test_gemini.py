import os
import sys

import pytest

from litellm.utils import supports_url_context

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

    def test_url_context(self):
        from litellm.utils import supports_url_context
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        litellm._turn_on_debug()

        base_completion_call_args = self.get_base_completion_call_args()

        if not supports_url_context(base_completion_call_args["model"], None):
            pytest.skip("Model does not support url context")

        response = self.completion_function(
            **base_completion_call_args,
            messages=[{"role": "user", "content": "Summarize the content of this URL: https://en.wikipedia.org/wiki/Artificial_intelligence"}],
            tools=[{"urlContext": {}}],
        )

        assert response is not None
        assert response.model_extra['vertex_ai_url_context_metadata'] is not None, "URL context metadata should be present"
        print(f"response={response}")

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


def test_gemini_url_context():
    from litellm import completion
    litellm._turn_on_debug()

    url = "https://ai.google.dev/gemini-api/docs/models"
    prompt = f"""
    Summarize this document:
    {url}
    """
    response = completion(
            model="gemini/gemini-2.0-flash",
            messages=[{"role": "user", "content": prompt}],
            tools=[{"urlContext": {}}],
        )
    print(response)
    message = response.choices[0].message.content
    assert message is not None
    url_context_metadata = response.model_extra['vertex_ai_url_context_metadata']
    assert url_context_metadata is not None
    urlMetadata = url_context_metadata[0]['urlMetadata'][0]
    assert urlMetadata['retrievedUrl'] == url
    assert urlMetadata['urlRetrievalStatus'] == 'URL_RETRIEVAL_STATUS_SUCCESS'



def test_gemini_with_grounding():
    from litellm import completion, Usage, stream_chunk_builder
    litellm._turn_on_debug()
    litellm.set_verbose = True
    tools = [{"googleSearch": {}}]

    # response = completion(model="gemini/gemini-2.0-flash", messages=[{"role": "user", "content": "What is the capital of France?"}], tools=tools)
    # print(response)
    # usage: Usage = response.usage
    # assert usage.prompt_tokens_details.web_search_requests is not None
    # assert usage.prompt_tokens_details.web_search_requests > 0


    ## Check streaming

    response = completion(model="gemini/gemini-2.0-flash", messages=[{"role": "user", "content": "What is the capital of France?"}], tools=tools, stream=True, stream_options={"include_usage": True})
    chunks = []
    for chunk in response:
        chunks.append(chunk)
    print(f"chunks before stream_chunk_builder: {chunks}")
    assert len(chunks) > 0
    complete_response = stream_chunk_builder(chunks)
    print(complete_response)
    assert complete_response is not None
    usage: Usage = complete_response.usage
    assert usage.prompt_tokens_details.web_search_requests is not None
    assert usage.prompt_tokens_details.web_search_requests > 0