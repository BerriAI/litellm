import json
import os
import sys
import traceback

import openai
import pytest
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
from litellm import completion, completion_cost, embedding

litellm.set_verbose = False


def test_openai_embedding():
    try:
        litellm.set_verbose = True
        response = embedding(
            model="text-embedding-ada-002",
            input=["good morning from litellm", "this is another item"],
            metadata={"anything": "good day"},
        )
        litellm_response = dict(response)
        litellm_response_keys = set(litellm_response.keys())
        litellm_response_keys.discard("_response_ms")

        print(litellm_response_keys)
        print("LiteLLM Response\n")
        # print(litellm_response)

        # same request with OpenAI 1.0+
        import openai

        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=["good morning from litellm", "this is another item"],
        )

        response = dict(response)
        openai_response_keys = set(response.keys())
        print(openai_response_keys)
        assert (
            litellm_response_keys == openai_response_keys
        )  # ENSURE the Keys in litellm response is exactly what the openai package returns
        assert (
            len(litellm_response["data"]) == 2
        )  # expect two embedding responses from litellm_response since input had two
        print(openai_response_keys)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_openai_embedding()


def test_openai_embedding_3():
    try:
        litellm.set_verbose = True
        response = embedding(
            model="text-embedding-3-small",
            input=["good morning from litellm", "this is another item"],
            metadata={"anything": "good day"},
            dimensions=5,
        )
        print(f"response:", response)
        litellm_response = dict(response)
        litellm_response_keys = set(litellm_response.keys())
        litellm_response_keys.discard("_response_ms")

        print(litellm_response_keys)
        print("LiteLLM Response\n")
        # print(litellm_response)

        # same request with OpenAI 1.0+
        import openai

        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=["good morning from litellm", "this is another item"],
            dimensions=5,
        )

        response = dict(response)
        openai_response_keys = set(response.keys())
        print(openai_response_keys)
        assert (
            litellm_response_keys == openai_response_keys
        )  # ENSURE the Keys in litellm response is exactly what the openai package returns
        assert (
            len(litellm_response["data"]) == 2
        )  # expect two embedding responses from litellm_response since input had two
        print(openai_response_keys)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize(
    "model, api_base, api_key",
    [
        # ("azure/text-embedding-ada-002", None, None),
        ("together_ai/BAAI/bge-base-en-v1.5", None, None),  # Updated to current Together AI embedding model
    ],
)
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_together_ai_embedding(model, api_base, api_key, sync_mode):
    try:
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")
        # litellm.set_verbose = True
        if sync_mode:
            response = embedding(
                model=model,
                input=["good morning from litellm"],
                api_base=api_base,
                api_key=api_key,
            )
        else:
            response = await litellm.aembedding(
                model=model,
                input=["good morning from litellm"],
                api_base=api_base,
                api_key=api_key,
            )
            # print(await response)
        print(response)
        print(response._hidden_params)
        response_keys = set(dict(response).keys())
        response_keys.discard("_response_ms")
        assert set(["usage", "model", "object", "data"]) == set(
            response_keys
        )  # assert litellm response has expected keys from OpenAI embedding response

        request_cost = litellm.completion_cost(
            completion_response=response, call_type="embedding"
        )

        print("Calculated request cost=", request_cost)

        assert isinstance(response.usage, litellm.Usage)
    except litellm.BadRequestError:
        print(
            "Bad request error occurred - Together AI raises 404s for their embedding models"
        )
        pass

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_openai_azure_embedding_simple()
import base64

import requests

litellm.set_verbose = True
url = "https://dummyimage.com/100/100/fff&text=Test+image"
response = requests.get(url)
file_data = response.content

encoded_file = base64.b64encode(file_data).decode("utf-8")
base64_image = f"data:image/png;base64,{encoded_file}"


from openai.types.embedding import Embedding


def _azure_ai_image_mock_response(*args, **kwargs):
    new_response = MagicMock()
    new_response.headers = {"azureml-model-group": "offer-cohere-embed-multili-paygo"}

    new_response.json.return_value = {
        "data": [Embedding(embedding=[1234], index=0, object="embedding")],
        "model": "",
        "object": "list",
        "usage": {"prompt_tokens": 1, "total_tokens": 2},
    }

    return new_response


@pytest.mark.parametrize(
    "model, api_base, api_key",
    [
        (
            "azure_ai/Cohere-embed-v3-multilingual-jzu",
            "https://Cohere-embed-v3-multilingual-jzu.eastus2.models.ai.azure.com",
            os.getenv("AZURE_AI_COHERE_API_KEY_2"),
        )
    ],
)
@pytest.mark.parametrize("sync_mode", [True])  # , False
@pytest.mark.asyncio
async def test_azure_ai_embedding_image(model, api_base, api_key, sync_mode):
    try:
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")
        input = base64_image
        if sync_mode:
            client = HTTPHandler()
        else:
            client = AsyncHTTPHandler()
        with patch.object(
            client, "post", side_effect=_azure_ai_image_mock_response
        ) as mock_client:
            if sync_mode:
                response = embedding(
                    model=model,
                    input=[input],
                    api_base=api_base,
                    api_key=api_key,
                    client=client,
                )
            else:
                response = await litellm.aembedding(
                    model=model,
                    input=[input],
                    api_base=api_base,
                    api_key=api_key,
                    client=client,
                )
        print(response)

        assert len(response.data) == 1

        print(response._hidden_params)
        response_keys = set(dict(response).keys())
        response_keys.discard("_response_ms")
        assert set(["usage", "model", "object", "data"]) == set(
            response_keys
        )  # assert litellm response has expected keys from OpenAI embedding response

        request_cost = litellm.completion_cost(completion_response=response)

        print("Calculated request cost=", request_cost)

        assert isinstance(response.usage, litellm.Usage)

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_openai_azure_embedding_timeouts():
    try:
        response = embedding(
            model="azure/text-embedding-ada-002",
            input=["good morning from litellm"],
            timeout=0.00001,
        )
        print(response)
    except openai.APITimeoutError:
        print("Good job got timeout error!")
        pass
    except Exception as e:
        pytest.fail(
            f"Expected timeout error, did not get the correct error. Instead got {e}"
        )


# test_openai_azure_embedding_timeouts()


def test_openai_embedding_timeouts():
    try:
        response = embedding(
            model="text-embedding-ada-002",
            input=["good morning from litellm"],
            timeout=0.00001,
        )
        print(response)
    except openai.APITimeoutError:
        print("Good job got OpenAI timeout error!")
        pass
    except Exception as e:
        pytest.fail(
            f"Expected timeout error, did not get the correct error. Instead got {e}"
        )


# test_openai_embedding_timeouts()


def test_openai_azure_embedding():
    try:
        api_key = os.environ["AZURE_API_KEY"]
        api_base = os.environ["AZURE_API_BASE"]
        api_version = os.environ["AZURE_API_VERSION"]

        os.environ["AZURE_API_VERSION"] = ""
        os.environ["AZURE_API_BASE"] = ""
        os.environ["AZURE_API_KEY"] = ""

        response = embedding(
            model="azure/text-embedding-ada-002",
            input=["good morning from litellm", "this is another item"],
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
        )
        print(response)

        os.environ["AZURE_API_VERSION"] = api_version
        os.environ["AZURE_API_BASE"] = api_base
        os.environ["AZURE_API_KEY"] = api_key

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.skipif(
    os.environ.get("CIRCLE_OIDC_TOKEN") is None,
    reason="Cannot run without being in CircleCI Runner",
)
def test_aaaaaa_openai_azure_embedding_with_oidc_and_cf():
    # TODO: Switch to our own Azure account, currently using ai.moda's account
    os.environ["AZURE_TENANT_ID"] = "17c0a27a-1246-4aa1-a3b6-d294e80e783c"
    os.environ["AZURE_CLIENT_ID"] = "4faf5422-b2bd-45e8-a6d7-46543a38acd0"

    old_key = os.environ["AZURE_API_KEY"]
    os.environ.pop("AZURE_API_KEY", None)

    try:
        response = embedding(
            model="azure/text-embedding-ada-002",
            input=["Hello"],
            azure_ad_token="oidc/circleci/",
            api_base="https://eastus2-litellm.openai.azure.com/",
            api_version="2024-06-01",
        )
        print(response)

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
    finally:
        os.environ["AZURE_API_KEY"] = old_key


from openai.types.embedding import Embedding


def _openai_mock_response(*args, **kwargs):
    new_response = MagicMock()
    new_response.headers = {"hello": "world"}

    new_response.parse.return_value = (
        openai.types.create_embedding_response.CreateEmbeddingResponse(
            data=[Embedding(embedding=[1234, 45667], index=0, object="embedding")],
            model="azure/test",
            object="list",
            usage=openai.types.create_embedding_response.Usage(
                prompt_tokens=1, total_tokens=2
            ),
        )
    )
    return new_response


def test_openai_azure_embedding_optional_arg():

    with patch.object(
        openai.resources.embeddings.Embeddings,
        "create",
        side_effect=_openai_mock_response,
    ) as mock_client:
        _ = litellm.embedding(
            model="azure/test",
            input=["test"],
            api_version="test",
            api_base="test",
            azure_ad_token="test",
        )

        mock_client.assert_called_once_with(
            model="test", 
            input=["test"], 
            extra_body={"azure_ad_token": "test"}, 
            timeout=600, 
            extra_headers={"X-Stainless-Raw-Response": "true"}
        )
        # Verify azure_ad_token is passed in extra_body, not as a direct parameter
        assert "azure_ad_token" not in mock_client.call_args.kwargs
        assert mock_client.call_args.kwargs["extra_body"]["azure_ad_token"] == "test"


# test_openai_azure_embedding()

# test_openai_embedding()


@pytest.mark.parametrize(
    "model, api_base",
    [
        ("embed-english-v2.0", None),
    ],
)
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_cohere_embedding(sync_mode, model, api_base):
    try:
        # litellm.set_verbose=True
        data = {
            "model": model,
            "input": ["good morning from litellm", "this is another item"],
            "input_type": "search_query",
            "api_base": api_base,
        }
        if sync_mode:
            response = embedding(**data)
        else:
            response = await litellm.aembedding(**data)

        print(f"response:", response)

        assert isinstance(response.usage, litellm.Usage)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_cohere_embedding()


@pytest.mark.parametrize("custom_llm_provider", ["cohere", "cohere_chat"])
@pytest.mark.asyncio()
async def test_cohere_embedding3(custom_llm_provider):
    try:
        litellm.set_verbose = True
        response = await litellm.aembedding(
            model=f"{custom_llm_provider}/embed-english-v3.0",
            input=["good morning from litellm", "this is another item"],
            timeout=None,
            max_retries=0,
        )
        print(f"response:", response)

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_cohere_embedding3()


@pytest.mark.parametrize(
    "model",
    [
        "bedrock/amazon.titan-embed-text-v1",
        "bedrock/amazon.titan-embed-image-v1",
        "bedrock/amazon.titan-embed-text-v2:0",
    ],
)
@pytest.mark.parametrize("sync_mode", [True, False])  # ,
@pytest.mark.asyncio
async def test_bedrock_embedding_titan(model, sync_mode):
    try:
        # this tests if we support str input for bedrock embedding
        litellm.set_verbose = True
        litellm.enable_cache()
        import time

        current_time = str(time.time())
        # DO NOT MAKE THE INPUT A LIST in this test
        if sync_mode:
            response = embedding(
                model=model,
                input=f"good morning from litellm, attempting to embed data {current_time}",  # input should always be a string in this test
                aws_region_name="us-west-2",
            )
        else:
            response = await litellm.aembedding(
                model=model,
                input=f"good morning from litellm, attempting to embed data {current_time}",  # input should always be a string in this test
                aws_region_name="us-west-2",
            )
        print("response:", response)
        assert isinstance(
            response["data"][0]["embedding"], list
        ), "Expected response to be a list"
        print("type of first embedding:", type(response["data"][0]["embedding"][0]))
        assert all(
            isinstance(x, float) for x in response["data"][0]["embedding"]
        ), "Expected response to be a list of floats"
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize(
    "model",
    [
        "bedrock/amazon.titan-embed-text-v1",
        "bedrock/amazon.titan-embed-image-v1",
        "bedrock/amazon.titan-embed-text-v2:0",
    ],
)
@pytest.mark.parametrize("sync_mode", [True])  # True,
@pytest.mark.asyncio
async def test_bedrock_embedding_titan_caching(model, sync_mode):
    try:
        # this tests if we support str input for bedrock embedding
        litellm.set_verbose = True
        litellm.enable_cache()
        import time

        current_time = str(time.time())
        # DO NOT MAKE THE INPUT A LIST in this test
        if sync_mode:
            response = embedding(
                model=model,
                input=f"good morning from litellm, attempting to embed data {current_time}",  # input should always be a string in this test
                aws_region_name="us-west-2",
            )
        else:
            response = await litellm.aembedding(
                model=model,
                input=f"good morning from litellm, attempting to embed data {current_time}",  # input should always be a string in this test
                aws_region_name="us-west-2",
            )
        print("response:", response)
        assert isinstance(
            response["data"][0]["embedding"], list
        ), "Expected response to be a list"
        print("type of first embedding:", type(response["data"][0]["embedding"][0]))
        assert all(
            isinstance(x, float) for x in response["data"][0]["embedding"]
        ), "Expected response to be a list of floats"

        # this also tests if we can return a cache response for this scenario
        import time

        start_time = time.time()

        if sync_mode:
            response = embedding(
                model=model,
                input=f"good morning from litellm, attempting to embed data {current_time}",  # input should always be a string in this test
            )
        else:
            response = await litellm.aembedding(
                model=model,
                input=f"good morning from litellm, attempting to embed data {current_time}",  # input should always be a string in this test
            )
        print(response)

        end_time = time.time()
        print(response._hidden_params)
        print(f"Embedding 2 response time: {end_time - start_time} seconds")

        assert end_time - start_time < 0.1
        litellm.disable_cache()

        assert isinstance(response.usage, litellm.Usage)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_bedrock_embedding_titan()


def test_bedrock_embedding_cohere():
    try:
        litellm.set_verbose = False
        response = embedding(
            model="cohere.embed-multilingual-v3",
            input=[
                "good morning from litellm, attempting to embed data",
                "lets test a second string for good measure",
            ],
            aws_region_name="os.environ/AWS_REGION_NAME_2",
        )
        assert isinstance(
            response["data"][0]["embedding"], list
        ), "Expected response to be a list"
        print(f"type of first embedding:", type(response["data"][0]["embedding"][0]))
        assert all(
            isinstance(x, float) for x in response["data"][0]["embedding"]
        ), "Expected response to be a list of floats"
        # print(f"response:", response)

        assert isinstance(response.usage, litellm.Usage)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_bedrock_embedding_cohere()


def test_demo_tokens_as_input_to_embeddings_fails_for_titan():
    litellm.set_verbose = True

    with pytest.raises(
        litellm.BadRequestError,
        match='litellm.BadRequestError: BedrockException - {"message":"Malformed input request: expected type: String, found: JSONArray, please reformat your input and try again."}',
    ):
        litellm.embedding(model="amazon.titan-embed-text-v1", input=[[1]])

    with pytest.raises(
        litellm.BadRequestError,
        match='litellm.BadRequestError: BedrockException - {"message":"Malformed input request: expected type: String, found: Integer, please reformat your input and try again."}',
    ):
        litellm.embedding(
            model="amazon.titan-embed-text-v1",
            input=[1],
        )


# comment out hf tests - since hf endpoints are unstable
def test_hf_embedding():
    try:
        # huggingface/microsoft/codebert-base
        # huggingface/facebook/bart-large
        response = embedding(
            model="huggingface/sentence-transformers/all-MiniLM-L6-v2",
            input=["good morning from litellm", "this is another item"],
        )
        print(f"response:", response)

        assert isinstance(response.usage, litellm.Usage)
    except Exception as e:
        # Note: Huggingface inference API is unstable and fails with "model loading errors all the time"
        pass


# test_hf_embedding()

from unittest.mock import MagicMock, patch


def tgi_mock_post(*args, **kwargs):
    import json

    expected_data = {
        "inputs": {
            "source_sentence": "good morning from litellm",
            "sentences": ["this is another item"],
        }
    }
    assert (
        json.loads(kwargs["data"]) == expected_data
    ), "Data does not match the expected data"
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/json"}
    mock_response.json.return_value = [0.7708950042724609]
    return mock_response


from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler


@pytest.mark.asyncio
@patch(
    "litellm.llms.huggingface.embedding.handler.async_get_hf_task_embedding_for_model"
)
@patch("litellm.llms.huggingface.embedding.handler.get_hf_task_embedding_for_model")
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_hf_embedding_sentence_sim(
    mock_async_get_hf_task_embedding_for_model,
    mock_get_hf_task_embedding_for_model,
    sync_mode,
):
    try:
        # huggingface/microsoft/codebert-base
        # huggingface/facebook/bart-large
        mock_get_hf_task_embedding_for_model.return_value = "sentence-similarity"
        mock_async_get_hf_task_embedding_for_model.return_value = "sentence-similarity"
        if sync_mode is True:
            client = HTTPHandler(concurrent_limit=1)
        else:
            client = AsyncHTTPHandler(concurrent_limit=1)
        with patch.object(client, "post", side_effect=tgi_mock_post) as mock_client:
            data = {
                "model": "huggingface/sentence-transformers/TaylorAI/bge-micro-v2",
                "input": ["good morning from litellm", "this is another item"],
                "client": client,
            }
            if sync_mode is True:
                response = embedding(**data)
            else:
                response = await litellm.aembedding(**data)

            print(f"response:", response)

            mock_client.assert_called_once()

        assert isinstance(response.usage, litellm.Usage)

    except Exception as e:
        # Note: Huggingface inference API is unstable and fails with "model loading errors all the time"
        raise e


# test async embeddings
def test_aembedding():
    try:
        import asyncio

        async def embedding_call():
            try:
                response = await litellm.aembedding(
                    model="text-embedding-ada-002",
                    input=["good morning from litellm", "this is another item"],
                )
                print(response)
                return response
            except Exception as e:
                pytest.fail(f"Error occurred: {e}")

        response = asyncio.run(embedding_call())
        print("Before caclulating cost, response", response)

        cost = litellm.completion_cost(completion_response=response)

        print("COST=", cost)
        assert cost == float("1e-06")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_aembedding()


def test_aembedding_azure():
    try:
        import asyncio

        async def embedding_call():
            try:
                response = await litellm.aembedding(
                    model="azure/text-embedding-ada-002",
                    input=["good morning from litellm", "this is another item"],
                )
                print(response)

                print(
                    "hidden params - custom_llm_provider",
                    response._hidden_params["custom_llm_provider"],
                )
                assert response._hidden_params["custom_llm_provider"] == "azure"

                assert isinstance(response.usage, litellm.Usage)
            except Exception as e:
                pytest.fail(f"Error occurred: {e}")

        asyncio.run(embedding_call())
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_aembedding_azure()


@pytest.mark.skip(reason="AWS Suspended Account")
def test_sagemaker_embeddings():
    try:
        response = litellm.embedding(
            model="sagemaker/berri-benchmarking-gpt-j-6b-fp16",
            input=["good morning from litellm", "this is another item"],
            input_cost_per_second=0.000420,
        )
        print(f"response: {response}")
        cost = completion_cost(completion_response=response)
        assert (
            cost > 0.0 and cost < 1.0
        )  # should never be > $1 for a single embedding call
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.skip(reason="AWS Suspended Account")
@pytest.mark.asyncio
async def test_sagemaker_aembeddings():
    try:
        response = await litellm.aembedding(
            model="sagemaker/berri-benchmarking-gpt-j-6b-fp16",
            input=["good morning from litellm", "this is another item"],
            input_cost_per_second=0.000420,
        )
        print(f"response: {response}")
        cost = completion_cost(completion_response=response)
        assert (
            cost > 0.0 and cost < 1.0
        )  # should never be > $1 for a single embedding call
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_mistral_embeddings():
    try:
        litellm.set_verbose = True
        response = litellm.embedding(
            model="mistral/mistral-embed",
            input=["good morning from litellm"],
        )
        print(f"response: {response}")
        assert isinstance(response.usage, litellm.Usage)
    except litellm.RateLimitError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_fireworks_embeddings():
    try:
        litellm.set_verbose = True
        response = litellm.embedding(
            model="fireworks_ai/nomic-ai/nomic-embed-text-v1.5",
            input=["good morning from litellm"],
        )
        print(f"response: {response}")
        assert isinstance(response.usage, litellm.Usage)
        cost = completion_cost(completion_response=response)
        print("cost", cost)
        assert cost > 0.0
        print(response._hidden_params)
        assert response._hidden_params["response_cost"] > 0.0
    except litellm.RateLimitError as e:
        pass
    except litellm.InternalServerError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_watsonx_embeddings():
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()

    def mock_wx_embed_request(url: str, **kwargs):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json.return_value = {
            "model_id": "ibm/slate-30m-english-rtrvr",
            "created_at": "2024-01-01T00:00:00.00Z",
            "results": [{"embedding": [0.0] * 254}],
            "input_token_count": 8,
        }
        return mock_response

    try:
        litellm.set_verbose = True
        with patch.object(client, "post", side_effect=mock_wx_embed_request):
            response = litellm.embedding(
                model="watsonx/ibm/slate-30m-english-rtrvr",
                input=["good morning from litellm"],
                token="secret-token",
                client=client,
            )

        print(f"response: {response}")
        assert isinstance(response.usage, litellm.Usage)
    except litellm.RateLimitError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.asyncio
async def test_watsonx_aembeddings():
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    client = AsyncHTTPHandler()

    def mock_async_client(*args, **kwargs):

        mocked_client = MagicMock()

        async def mock_send(request, *args, stream: bool = False, **kwags):
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.json.return_value = {
                "model_id": "ibm/slate-30m-english-rtrvr",
                "created_at": "2024-01-01T00:00:00.00Z",
                "results": [{"embedding": [0.0] * 254}],
                "input_token_count": 8,
            }
            mock_response.is_error = False
            return mock_response

        mocked_client.send = mock_send

        return mocked_client

    try:
        litellm.set_verbose = True
        with patch.object(client, "post", side_effect=mock_async_client) as mock_client:
            response = await litellm.aembedding(
                model="watsonx/ibm/slate-30m-english-rtrvr",
                input=["good morning from litellm"],
                token="secret-token",
                client=client,
            )
            mock_client.assert_called_once()
        print(f"response: {response}")
        assert isinstance(response.usage, litellm.Usage)
    except litellm.RateLimitError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_mistral_embeddings()


@pytest.mark.skip(
    reason="Community maintained embedding provider - they are quite unstable"
)
def test_voyage_embeddings():
    try:
        litellm.set_verbose = True
        response = litellm.embedding(
            model="voyage/voyage-01",
            input=["good morning from litellm"],
        )
        print(f"response: {response}")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize(
    "input", ["good morning from litellm", ["good morning from litellm"]]  #
)
@pytest.mark.asyncio
async def test_gemini_embeddings(sync_mode, input):
    try:
        litellm.set_verbose = True
        if sync_mode:
            response = litellm.embedding(
                model="gemini/text-embedding-004",
                input=input,
            )
        else:
            response = await litellm.aembedding(
                model="gemini/text-embedding-004",
                input=input,
            )
        print(f"response: {response}")

        # stubbed endpoint is setup to return this
        assert isinstance(response.data[0]["embedding"], list)
        assert response.usage.prompt_tokens > 0
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_voyage_embeddings()
# def test_xinference_embeddings():
#     try:
#         litellm.set_verbose = True
#         response = litellm.embedding(
#             model="xinference/bge-base-en",
#             input=["good morning from litellm"],
#         )
#         print(f"response: {response}")
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")
# test_xinference_embeddings()

# test_sagemaker_embeddings()
# def local_proxy_embeddings():
#     litellm.set_verbose=True
#     response = embedding(
#             model="openai/custom_embedding",
#             input=["good morning from litellm"],
#             api_base="http://0.0.0.0:8000/"
#         )
#     print(response)

# local_proxy_embeddings()


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
@pytest.mark.flaky(retries=6, delay=1)
@pytest.mark.skip(reason="Skipping test due to flakyness")
async def test_hf_embedddings_with_optional_params(sync_mode):
    litellm.set_verbose = True

    if sync_mode:
        client = HTTPHandler(concurrent_limit=1)
        mock_obj = MagicMock()
    else:
        client = AsyncHTTPHandler(concurrent_limit=1)
        mock_obj = AsyncMock()

    with patch.object(client, "post", new=mock_obj) as mock_client:
        try:
            if sync_mode:
                response = embedding(
                    model="huggingface/jinaai/jina-embeddings-v2-small-en",
                    input=["good morning from litellm"],
                    top_p=10,
                    top_k=10,
                    wait_for_model=True,
                    client=client,
                )
            else:
                response = await litellm.aembedding(
                    model="huggingface/jinaai/jina-embeddings-v2-small-en",
                    input=["good morning from litellm"],
                    top_p=10,
                    top_k=10,
                    wait_for_model=True,
                    client=client,
                )
        except Exception as e:
            print(e)

        mock_client.assert_called_once()

        print(f"mock_client.call_args.kwargs: {mock_client.call_args.kwargs}")
        assert "options" in mock_client.call_args.kwargs["data"]
        json_data = json.loads(mock_client.call_args.kwargs["data"])
        assert "wait_for_model" in json_data["options"]
        assert json_data["options"]["wait_for_model"] is True
        assert json_data["parameters"]["top_p"] == 10
        assert json_data["parameters"]["top_k"] == 10


def test_hosted_vllm_embedding(monkeypatch):
    monkeypatch.setenv("HOSTED_VLLM_API_BASE", "http://localhost:8000")
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()
    with patch.object(client, "post") as mock_post:
        try:
            embedding(
                model="hosted_vllm/jina-embeddings-v3",
                input=["Hello world"],
                client=client,
            )
        except Exception as e:
            print(e)

        mock_post.assert_called_once()

        json_data = json.loads(mock_post.call_args.kwargs["data"])
        assert json_data["input"] == ["Hello world"]
        assert json_data["model"] == "jina-embeddings-v3"


def test_llamafile_embedding(monkeypatch):
    monkeypatch.setenv("LLAMAFILE_API_BASE", "http://localhost:8080/v1")
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()
    with patch.object(client, "post") as mock_post:
        try:
            embedding(
                model="llamafile/jina-embeddings-v3",
                input=["Hello world"],
                client=client,
            )
        except Exception as e:
            print(e)

        mock_post.assert_called_once()

        json_data = json.loads(mock_post.call_args.kwargs["data"])
        assert json_data["input"] == ["Hello world"]
        assert json_data["model"] == "jina-embeddings-v3"


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_lm_studio_embedding(monkeypatch, sync_mode):
    monkeypatch.setenv("LM_STUDIO_API_BASE", "http://localhost:8000")
    from litellm.llms.custom_httpx.http_handler import HTTPHandler, AsyncHTTPHandler

    client = HTTPHandler() if sync_mode else AsyncHTTPHandler()
    with patch.object(client, "post") as mock_post:
        try:
            if sync_mode:
                embedding(
                    model="lm_studio/jina-embeddings-v3",
                    input=["Hello world"],
                    client=client,
                )
            else:
                await litellm.aembedding(
                    model="lm_studio/jina-embeddings-v3",
                    input=["Hello world"],
                    client=client,
                )
        except Exception as e:
            print(e)

        mock_post.assert_called_once()

        json_data = json.loads(mock_post.call_args.kwargs["data"])
        assert json_data["input"] == ["Hello world"]
        assert json_data["model"] == "jina-embeddings-v3"


@pytest.mark.parametrize(
    "model",
    [
        "text-embedding-ada-002",
        "azure/text-embedding-ada-002",
    ],
)
def test_embedding_response_ratelimit_headers(model):
    response = embedding(
        model=model,
        input=["Hello world"],
    )
    hidden_params = response._hidden_params
    additional_headers = hidden_params.get("additional_headers", {})

    print("additional_headers", additional_headers)

    # Azure is flaky with returning x-ratelimit-remaining-requests, we need to verify the upstream api returns this header
    # if upstream api returns this header, we need to verify the header is transformed by litellm
    if (
        "llm_provider-x-ratelimit-limit-requests" in additional_headers
        or "x-ratelimit-limit-requests" in additional_headers
    ):
        assert "x-ratelimit-remaining-requests" in additional_headers
        assert int(additional_headers["x-ratelimit-remaining-requests"]) > 0

    assert "x-ratelimit-remaining-tokens" in additional_headers
    assert int(additional_headers["x-ratelimit-remaining-tokens"]) > 0


@pytest.mark.parametrize(
    "input, input_type",
    [
        (
            [
                "data:image/jpeg;base64,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"
            ],
            "image",
        ),
        (["hello world"], "text"),
    ],
)
def test_cohere_img_embeddings(input, input_type):
    litellm.set_verbose = True
    try:
        response = embedding(
            model="cohere/embed-english-v3.0",
            input=input,
        )

        if input_type == "image":
            assert response.usage.prompt_tokens_details.image_tokens > 0
        else:
            assert response.usage.prompt_tokens_details.text_tokens > 0
    except litellm.InternalServerError as e:
        # Cohere API is experiencing internal server errors - this is expected
        # and our exception mapping is working correctly
        if "internal server error" in str(e).lower():
            pytest.skip("Cohere API is currently experiencing internal server errors")
        else:
            raise e


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_embedding_with_extra_headers(sync_mode):

    input = ["hello world"]
    from litellm.llms.custom_httpx.http_handler import HTTPHandler, AsyncHTTPHandler

    if sync_mode:
        client = HTTPHandler()
    else:
        client = AsyncHTTPHandler()

    data = {
        "model": "cohere/embed-english-v3.0",
        "input": input,
        "extra_headers": {"my-test-param": "hello-world"},
        "client": client,
    }
    with patch.object(client, "post") as mock_post:
        try:
            if sync_mode:
                embedding(**data)
            else:
                await litellm.aembedding(**data)
        except Exception as e:
            print(e)

        mock_post.assert_called_once()
        assert "my-test-param" in mock_post.call_args.kwargs["headers"]


@pytest.mark.parametrize(
    "input_data, expected_payload_input",
    [
        # Case 1: Input with only text strings
        (
            ["hello world", "foo bar"],
            ["hello world", "foo bar"],
        ),
        # Case 2: Input with a mix of text and a base64 encoded image
        (
            [
                "A picture of a cat",
                "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
            ],
            [
                {"text": "A picture of a cat"},
                {
                    "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
                },
            ],
        ),
        # Case 3: Input with only a base64 encoded image
        (
            [
                "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
            ],
            [
                {
                    "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
                }
            ],
        ),
    ],
)
def test_jina_ai_img_embeddings(input_data, expected_payload_input):
    """
    Tests the input transformation logic for Jina AI embeddings using mocks.

    This test verifies that when litellm.embedding is called with a jina_ai model,
    the 'input' field in the request payload is formatted correctly based on whether
    the input contains text or base64 encoded images.
    """
    # We patch the `post` method of the HTTPHandler. This intercepts the network
    # request before it's actually sent.
    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post") as mock_post:
        # Configure the mock to return a successful, minimal valid response.
        # This prevents litellm from raising an error when processing the response.
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1] * 768,  # Dummy embedding vector
                }
            ],
            "model": "jina-embeddings-v4",
        }
        mock_post.return_value = mock_response

        # Call the function we want to test
        try:
            litellm.embedding(
                model="jina_ai/jina-embeddings-v4", input=input_data
            )
        except Exception as e:
            pytest.fail(
                f"litellm.embedding call failed with an unexpected exception: {e}"
            )

        # --- Assertions ---
        # 1. Check that our mock `post` method was called exactly once.
        mock_post.assert_called_once()

        # 2. Extract the keyword arguments passed to the mock call.
        #    The request payload is in the 'data' keyword argument.
        kwargs = mock_post.call_args.kwargs
        assert "data" in kwargs

        # 3. Parse the JSON payload string into a Python dictionary.
        sent_data = json.loads(kwargs["data"])

        # 4. This is the core of our test:
        #    Assert that the 'input' field in the payload matches our expectation.
        assert "input" in sent_data
        assert sent_data["input"] == expected_payload_input
