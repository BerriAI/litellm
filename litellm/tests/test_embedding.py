import sys, os
import traceback
import pytest
from dotenv import load_dotenv
import openai

load_dotenv()

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import embedding, completion, completion_cost, EmbeddingResponse

litellm.set_verbose = False


def format_tests(response: EmbeddingResponse):
    """
    Tests to check if the response is in the openai format
    """

    # 1. if data is a pydantic object
    assert hasattr(response.data[0], "embedding")  # type: ignore


def test_openai_embedding():
    try:
        litellm.set_verbose = True
        response = embedding(
            model="text-embedding-ada-002",
            input=["good morning from litellm", "this is another item"],
            metadata={"anything": "good day"},
        )

        format_tests(response=response)

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
        format_tests(response=response)
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


def test_openai_azure_embedding_simple():
    try:
        litellm.set_verbose = True
        response = embedding(
            model="azure/azure-embedding-model",
            input=["good morning from litellm"],
        )
        print(response)
        format_tests(response=response)
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


# test_openai_azure_embedding_simple()


def test_openai_azure_embedding_timeouts():
    try:
        response = embedding(
            model="azure/azure-embedding-model",
            input=["good morning from litellm"],
            timeout=0.00001,
        )
        format_tests(response=response)
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
        format_tests(response=response)
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
            model="azure/azure-embedding-model",
            input=["good morning from litellm", "this is another item"],
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
        )
        print(response)

        format_tests(response=response)

        os.environ["AZURE_API_VERSION"] = api_version
        os.environ["AZURE_API_BASE"] = api_base
        os.environ["AZURE_API_KEY"] = api_key

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_openai_azure_embedding_optional_arg(mocker):
    mocked_create_embeddings = mocker.patch.object(
        openai.resources.embeddings.Embeddings,
        "create",
        return_value=openai.types.create_embedding_response.CreateEmbeddingResponse(
            data=[],
            model="azure/test",
            object="list",
            usage=openai.types.create_embedding_response.Usage(
                prompt_tokens=1, total_tokens=2
            ),
        ),
    )
    _ = litellm.embedding(
        model="azure/test",
        input=["test"],
        api_version="test",
        api_base="test",
        azure_ad_token="test",
    )

    assert mocked_create_embeddings.called_once_with(
        model="test", input=["test"], timeout=600
    )
    assert "azure_ad_token" not in mocked_create_embeddings.call_args.kwargs


# test_openai_azure_embedding()

# test_openai_embedding()


def test_cohere_embedding():
    try:
        # litellm.set_verbose=True
        response = embedding(
            model="embed-english-v2.0",
            input=["good morning from litellm", "this is another item"],
        )
        print(f"response:", response)

        format_tests(response=response)

        assert isinstance(response.usage, litellm.Usage)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_cohere_embedding()


def test_cohere_embedding3():
    try:
        litellm.set_verbose = True
        response = embedding(
            model="embed-english-v3.0",
            input=["good morning from litellm", "this is another item"],
        )
        print(f"response:", response)

        format_tests(response=response)
        custom_llm_provider = response._hidden_params["custom_llm_provider"]

        assert custom_llm_provider == "cohere"

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_cohere_embedding3()


def test_bedrock_embedding_titan():
    try:
        # this tests if we support str input for bedrock embedding
        litellm.set_verbose = True
        litellm.enable_cache()
        import time

        current_time = str(time.time())
        # DO NOT MAKE THE INPUT A LIST in this test
        response = embedding(
            model="bedrock/amazon.titan-embed-text-v1",
            input=f"good morning from litellm, attempting to embed data {current_time}",  # input should always be a string in this test
        )

        format_tests(response=response)

        # print(f"response:", response)
        assert isinstance(
            response["data"][0]["embedding"], list
        ), "Expected response to be a list"
        print(f"type of first embedding:", type(response["data"][0]["embedding"][0]))
        assert all(
            isinstance(x, float) for x in response["data"][0]["embedding"]
        ), "Expected response to be a list of floats"

        # this also tests if we can return a cache response for this scenario
        import time

        start_time = time.time()

        response = embedding(
            model="bedrock/amazon.titan-embed-text-v1",
            input=f"good morning from litellm, attempting to embed data {current_time}",  # input should always be a string in this test
        )
        print(response)

        format_tests(response=response)

        end_time = time.time()
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

        format_tests(response=response)

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
        match="BedrockException - Bedrock Embedding API input must be type str | List[str]",
    ):
        litellm.embedding(model="amazon.titan-embed-text-v1", input=[[1]])

    with pytest.raises(
        litellm.BadRequestError,
        match="BedrockException - Bedrock Embedding API input must be type str | List[str]",
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

        format_tests(response=response)

        print(f"response:", response)

        assert isinstance(response.usage, litellm.Usage)
    except Exception as e:
        # Note: Huggingface inference API is unstable and fails with "model loading errors all the time"
        if isinstance(e, litellm.APIError) or isinstance(e, litellm.APIConnectionError):
            pass
        else:
            pytest.fail(f"Failed - {str(e)}")


# test_hf_embedding()


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
                format_tests(response=response)
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
                    model="azure/azure-embedding-model",
                    input=["good morning from litellm", "this is another item"],
                )
                print(response)

                format_tests(response=response)

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
        format_tests(response=response)
        print(f"response: {response}")
        assert isinstance(response.usage, litellm.Usage)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_mistral_embeddings()


def test_voyage_embeddings():
    try:
        litellm.set_verbose = True
        response = litellm.embedding(
            model="voyage/voyage-01",
            input=["good morning from litellm"],
        )
        format_tests(response=response)
        print(f"response: {response}")
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
