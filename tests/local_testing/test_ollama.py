import json

from dotenv import load_dotenv

load_dotenv()

import pytest

import litellm

## for ollama we can't test making the completion call


def test_ollama_vision_model():
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()
    from unittest.mock import patch

    with patch.object(client, "post") as mock_post:
        try:
            litellm.completion(
                model="ollama/llama3.2-vision:11b",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Whats in this image?"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://dummyimage.com/100/100/fff&text=Test+image"
                                },
                            },
                        ],
                    }
                ],
                client=client,
            )
        except Exception as e:
            print(e)
        mock_post.assert_called()

        print(mock_post.call_args.kwargs)

        json_data = json.loads(mock_post.call_args.kwargs["data"])
        assert json_data["model"] == "llama3.2-vision:11b"
        assert "images" in json_data
        assert "prompt" in json_data
        assert json_data["prompt"].startswith("### User:\n")


def test_ollama_ssl_verify():
    import ssl

    import httpx

    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    try:
        response = litellm.completion(
            model="ollama/llama3.1",
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather like in San Francisco?",
                }
            ],
            ssl_verify=False,
        )
    except Exception as e:
        print(e)

    client: HTTPHandler = litellm.in_memory_llm_clients_cache.get_cache(
        "httpx_clientssl_verify_False"
    )

    test_client = httpx.Client(verify=False)
    print(client)
    assert (
        client.client._transport._pool._ssl_context.verify_mode
        == test_client._transport._pool._ssl_context.verify_mode
    )


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.asyncio
async def test_async_ollama_ssl_verify(stream):
    import httpx

    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    try:
        response = await litellm.acompletion(
            model="ollama/llama3.1",
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather like in San Francisco?",
                }
            ],
            ssl_verify=False,
            stream=stream,
        )
    except Exception as e:
        print(e)

    client: AsyncHTTPHandler = litellm.in_memory_llm_clients_cache.get_cache("async_httpx_clientssl_verify_Falseollama")

    # check client
    print("type of transport in client=", type(client.client._transport))
    print("vars in transport in client=", vars(client.client._transport))
    litellm_created_session = client.client._transport.get_valid_client_session()
    print("litellm_created_session=", litellm_created_session)
    # check session ssl
    print("litellm_created_session ssl=", litellm_created_session.connector._ssl)

    # create aiohttp transport with ssl_verify=False
    import aiohttp

    aiohttp_session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False))
    print("aiohttp_session ssl=", aiohttp_session.connector._ssl)

    assert litellm_created_session.connector._ssl is False
    assert litellm_created_session.connector._ssl == aiohttp_session.connector._ssl
