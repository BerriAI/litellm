import litellm
import json
import asyncio
from dotenv import load_dotenv
load_dotenv()
from litellm.llms.custom_httpx.httpx_handler import HTTPHandler


def test_sync_retrieve():
    print(litellm.responses_retrieve("resp_67d10c95f70c8191a4ecc9c6452062690a0b9965f83e0b98"))
    # response = litellm.responses(
    #         model="gpt-4o",
    #         input="Hi!"
    #     )
    # print(response)

async def test_async_retrieve():
    response = await litellm.aresponses_retrieve("resp_67d10c95f70c8191a4ecc9c6452062690a0b9965f83e0b98", llm_provider="openai")
    print(response)


def test_sync_delete():
    print(litellm.responses_delete("resp_67d10dc296108191905f2b7cd6d5ed380a1dc205c7fa6118", llm_provider="openai"))

async def test_async_delete(id):
    response = await litellm.aresponses_delete(id, llm_provider="openai")
    print(response)

# test_sync_delete()
# id = "resp_67d10dfd0e58819190b14fcc8450f0d003e1306c8a9d24dd"
# asyncio.run(test_async_delete(id))
# test_sync_retrieve()

# asyncio.run(test_async_retrieve())


def test_sync_retrieve_delete():
    # Create a response
    response = litellm.responses(
            model="gpt-4o",
            input="What color is the sky?",
        )
    assert response is not None

    id = response['id']

    # Test retrieve functionality
    retrieved_response = litellm.responses_retrieve(id, custom_llm_provider="openai")
    assert retrieved_response["text"] == response["text"]

    # Test delete functionality
    deleted_response = litellm.responses_delete(id, custom_llm_provider="openai")
    assert deleted_response is not None

async def test_async_retrieve_delete():
    # Create a response
    response = await litellm.aresponses(
            model="gpt-4o",
            input="What color is the sky?",
        )
    assert response is not None

    id = response['id']

    # Test retrieve functionality
    retrieved_response = await litellm.aresponses_retrieve(id, custom_llm_provider="openai")
    assert retrieved_response["text"] == response["text"]

    # Test delete functionality
    deleted_response = await litellm.aresponses_delete(id, custom_llm_provider="openai")
    assert deleted_response is not None
    
# test_sync_retrieve_delete()
asyncio.run(test_async_retrieve_delete())

"""
response = await litellm.responses.get()
"""