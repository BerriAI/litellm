from base_llm_unit_tests import BaseLLMChatTest
import pytest


# Test implementations
@pytest.mark.skip(reason="Deepseek API is hanging")
class TestDeepSeekChatCompletion(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        return {
            "model": "deepseek/deepseek-reasoner",
        }

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
        pass

    def test_multilingual_requests(self):
        """
        DeepSeek API raises a 400 BadRequest error when the request contains invalid utf-8 sequences.

        Todo: if litellm.modify_params is True ensure it's a valid utf-8 sequence
        """
        pass


@pytest.mark.parametrize("stream", [True, False])
def test_deepseek_mock_completion(stream):
    """
    Deepseek API is hanging. Mock the call, to a fake endpoint, so we can confirm our integration is working.
    """
    import litellm
    from litellm import completion

    litellm._turn_on_debug()

    response = completion(
        model="deepseek/deepseek-reasoner",
        messages=[{"role": "user", "content": "Hello, world!"}],
        api_base="https://exampleopenaiendpoint-production.up.railway.app/v1/chat/completions",
        stream=stream,
    )
    print(f"response: {response}")
    if stream:
        for chunk in response:
            print(chunk)
    else:
        assert response is not None


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.asyncio
async def test_deepseek_mock_async_completion(stream):
    """
    Deepseek API is hanging. Mock the call, to a fake endpoint, so we can confirm our integration is working.
    """
    import litellm
    from litellm import completion, acompletion

    litellm._turn_on_debug()

    response = await acompletion(
        model="deepseek/deepseek-reasoner",
        messages=[{"role": "user", "content": "Hello, world!"}],
        api_base="https://exampleopenaiendpoint-production.up.railway.app/v1/chat/completions",
        stream=stream,
    )
    print(f"response: {response}")
    if stream:
        async for chunk in response:
            print(chunk)
    else:
        assert response is not None
