import pytest
import litellm
from litellm.utils import should_run_mock_completion


def test_should_run_mock_completion_with_empty_string():
    assert (
        should_run_mock_completion(
            mock_response="", mock_tool_calls=None, mock_timeout=None
        )
        is True
    )
    assert (
        should_run_mock_completion(
            mock_response=None, mock_tool_calls=None, mock_timeout=None
        )
        is False
    )
    assert (
        should_run_mock_completion(
            mock_response="test", mock_tool_calls=None, mock_timeout=None
        )
        is True
    )


def test_completion_mock_response_empty_string():
    response = litellm.completion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="",
        api_base="http://127.0.0.1:9",
        api_key="sk-test",
        num_retries=0,
    )
    assert response.choices[0].message.content == ""


@pytest.mark.asyncio
async def test_acompletion_mock_response_empty_string():
    response = await litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="",
        api_base="http://127.0.0.1:9",
        api_key="sk-test",
        num_retries=0,
    )
    assert response.choices[0].message.content == ""
