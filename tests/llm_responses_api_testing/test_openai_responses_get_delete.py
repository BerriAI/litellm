import litellm
import pytest
import asyncio 


@pytest.mark.parametrize("model", ["gpt-4o"])
def test_sync_retrieve_delete(model):
    '''
    Test creation, then getting, then deleting.
    '''
    response = litellm.responses(
            model=model,
            input="What color is the sky?",
        )
    assert response is not None

    id = response['id']

    # Test retrieve functionality
    retrieved_response = litellm.responses_retrieve(id, llm_provider="openai")
    assert retrieved_response["text"] == response["text"]

    # Test delete functionality
    deleted_response = litellm.responses_delete(id, llm_provider="openai")
    assert deleted_response is not None


@pytest.mark.parametrize("model", ["gpt-4o"])
@pytest.mark.asyncio  
async def test_async_retrieve_delete(model):
    '''
    Test async creation, then getting, then deleting.
    '''
    response = await litellm.aresponses(
            model=model,
            input="What color is the sky?",
        )
    assert response is not None

    id = response['id']

    # Test async retrieve functionality
    retrieved_response = await litellm.aresponses_retrieve(id, llm_provider="openai")
    assert retrieved_response["text"] == response["text"]

    # Test async delete functionality
    deleted_response = await litellm.aresponses_delete(id, llm_provider="openai")
    assert deleted_response is not None