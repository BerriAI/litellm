from base_llm_unit_tests import BaseLLMChatTest
import pytest
import litellm


# Test implementations
@pytest.mark.skip(reason="Deepseek API is hanging")
class TestDeepSeekChatCompletion(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        return {
            "model": "deepseek/deepseek-reasoner",
        }

def test_completion_cost_deepseek():
    litellm.set_verbose = True
    model_name = "deepseek/deepseek-chat"
    messages_1 = [
        {
            "role": "system",
            "content": "You are a history expert. The user will provide a series of questions, and your answers should be concise and start with `Answer:`",
        },
        {
            "role": "user",
            "content": "In what year did Qin Shi Huang unify the six states?",
        },
        {"role": "assistant", "content": "Answer: 221 BC"},
        {"role": "user", "content": "Who was the founder of the Han Dynasty?"},
        {"role": "assistant", "content": "Answer: Liu Bang"},
        {"role": "user", "content": "Who was the last emperor of the Tang Dynasty?"},
        {"role": "assistant", "content": "Answer: Li Zhu"},
        {
            "role": "user",
            "content": "Who was the founding emperor of the Ming Dynasty?",
        },
        {"role": "assistant", "content": "Answer: Zhu Yuanzhang"},
        {
            "role": "user",
            "content": "Who was the founding emperor of the Qing Dynasty?",
        },
    ]

    message_2 = [
        {
            "role": "system",
            "content": "You are a history expert. The user will provide a series of questions, and your answers should be concise and start with `Answer:`",
        },
        {
            "role": "user",
            "content": "In what year did Qin Shi Huang unify the six states?",
        },
        {"role": "assistant", "content": "Answer: 221 BC"},
        {"role": "user", "content": "Who was the founder of the Han Dynasty?"},
        {"role": "assistant", "content": "Answer: Liu Bang"},
        {"role": "user", "content": "Who was the last emperor of the Tang Dynasty?"},
        {"role": "assistant", "content": "Answer: Li Zhu"},
        {
            "role": "user",
            "content": "Who was the founding emperor of the Ming Dynasty?",
        },
        {"role": "assistant", "content": "Answer: Zhu Yuanzhang"},
        {"role": "user", "content": "When did the Shang Dynasty fall?"},
    ]
    try:
        response_1 = litellm.completion(model=model_name, messages=messages_1)
        response_2 = litellm.completion(model=model_name, messages=message_2)
        # Add any assertions here to check the response
        print(response_2)
        assert response_2.usage.prompt_cache_hit_tokens is not None
        assert response_2.usage.prompt_cache_miss_tokens is not None
        assert (
            response_2.usage.prompt_tokens
            == response_2.usage.prompt_cache_miss_tokens
            + response_2.usage.prompt_cache_hit_tokens
        )
        assert (
            response_2.usage._cache_read_input_tokens
            == response_2.usage.prompt_cache_hit_tokens
        )
    except litellm.APIError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_deepseek_fill_reasoning_content_multiturn():
    """
    Unit test for _fill_reasoning_content.
    Reproduces issue #28045: DeepSeek thinking mode fails in multi-turn conversations
    because reasoning_content is not passed back to the API.
    """
    from litellm.llms.deepseek.chat.transformation import DeepSeekChatConfig

    config = DeepSeekChatConfig()

    # Case 1: assistant message already has reasoning_content — should be left as-is
    messages_with_rc = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi", "reasoning_content": "I thought about it"},
        {"role": "user", "content": "Follow up"},
    ]
    result = config._fill_reasoning_content(messages_with_rc)
    assert result[1]["reasoning_content"] == "I thought about it"

    # Case 2: assistant message has reasoning_content in provider_specific_fields — should be promoted
    messages_with_psf = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": "Hi",
            "provider_specific_fields": {"reasoning_content": "stored thinking"},
        },
        {"role": "user", "content": "Follow up"},
    ]
    result = config._fill_reasoning_content(messages_with_psf)
    assert result[1]["reasoning_content"] == "stored thinking"
    # Should be removed from provider_specific_fields to avoid duplication
    assert "reasoning_content" not in result[1].get("provider_specific_fields", {})

    # Case 3: assistant message has no reasoning_content anywhere — should inject placeholder
    messages_no_rc = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Follow up"},
    ]
    result = config._fill_reasoning_content(messages_no_rc)
    assert result[1]["reasoning_content"] == " "

    # Case 4: non-assistant messages should never be touched
    messages_user_only = [
        {"role": "user", "content": "Hello"},
        {"role": "system", "content": "You are helpful"},
    ]
    result = config._fill_reasoning_content(messages_user_only)
    assert "reasoning_content" not in result[0]
    assert "reasoning_content" not in result[1]


def test_deepseek_fill_reasoning_content_guard_in_transform_request():
    """
    _fill_reasoning_content must only run when BOTH conditions are true:
      1. supports_reasoning() is True for the model
      2. thinking mode is explicitly enabled in optional_params ({"type": "enabled"})

    This prevents spurious injection on models like deepseek-v3.2 that support
    thinking as opt-in but not always-on. Addresses oss-pr-review-agent feedback
    on PR #28057.
    """
    from litellm.llms.deepseek.chat.transformation import DeepSeekChatConfig

    config = DeepSeekChatConfig()

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Follow up"},
    ]

    # Case 1: reasoning model + thinking enabled -> injection should happen
    result = config.transform_request(
        model="deepseek-reasoner",
        messages=messages,
        optional_params={"thinking": {"type": "enabled"}},
        litellm_params={},
        headers={},
    )
    assert result["messages"][1].get("reasoning_content") == " ", (
        "reasoning_content should be injected when thinking is enabled"
    )

    # Case 2: reasoning model + thinking NOT in optional_params -> no injection
    result = config.transform_request(
        model="deepseek-reasoner",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert "reasoning_content" not in result["messages"][1], (
        "reasoning_content should not be injected when thinking is not enabled"
    )

    # Case 3: non-reasoning model + thinking enabled -> no injection
    result = config.transform_request(
        model="deepseek-chat",
        messages=messages,
        optional_params={"thinking": {"type": "enabled"}},
        litellm_params={},
        headers={},
    )
    assert "reasoning_content" not in result["messages"][1], (
        "reasoning_content should not be injected for non-reasoning models"
    )
