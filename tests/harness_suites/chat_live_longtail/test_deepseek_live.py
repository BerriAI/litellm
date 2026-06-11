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
