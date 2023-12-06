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
from litellm import embedding, completion

litellm.set_verbose = False

def test_openai_embedding():
    try:
        litellm.set_verbose=True
        response = embedding(
            model="text-embedding-ada-002", 
            input=["good morning from litellm", "this is another item"], 
            metadata =  {"anything": "good day"}
        )
        litellm_response = dict(response)
        litellm_response_keys = set(litellm_response.keys())
        litellm_response_keys.discard('_response_ms')

        print(litellm_response_keys)
        print("LiteLLM Response\n")
        # print(litellm_response)
        
        # same request with OpenAI 1.0+ 
        import openai
        client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        response = client.embeddings.create(
            model="text-embedding-ada-002", input=["good morning from litellm", "this is another item"]
        )

        response = dict(response)
        openai_response_keys = set(response.keys())
        print(openai_response_keys)
        assert litellm_response_keys == openai_response_keys # ENSURE the Keys in litellm response is exactly what the openai package returns
        assert len(litellm_response["data"]) == 2 # expect two embedding responses from litellm_response since input had two
        print(openai_response_keys)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
# test_openai_embedding()

def test_openai_azure_embedding_simple():
    try:
        response = embedding(
            model="azure/azure-embedding-model",
            input=["good morning from litellm"],
        )
        print(response)
        response_keys = set(dict(response).keys())
        response_keys.discard('_response_ms')
        assert set(["usage", "model", "object", "data"]) == set(response_keys) #assert litellm response has expected keys from OpenAI embedding response

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

# test_openai_azure_embedding_simple()


def test_openai_azure_embedding_timeouts():
    try:
        response = embedding(
            model="azure/azure-embedding-model",
            input=["good morning from litellm"],
            timeout=0.00001
        )
        print(response)
    except openai.APITimeoutError:
        print("Good job got timeout error!")
        pass
    except Exception as e:
        pytest.fail(f"Expected timeout error, did not get the correct error. Instead got {e}")

# test_openai_azure_embedding_timeouts()

def test_openai_embedding_timeouts():
    try:
        response = embedding(
            model="text-embedding-ada-002",
            input=["good morning from litellm"],
            timeout=0.00001
        )
        print(response)
    except openai.APITimeoutError:
        print("Good job got OpenAI timeout error!")
        pass
    except Exception as e:
        pytest.fail(f"Expected timeout error, did not get the correct error. Instead got {e}")
# test_openai_embedding_timeouts()

def test_openai_azure_embedding():
    try:
        api_key = os.environ['AZURE_API_KEY']
        api_base = os.environ['AZURE_API_BASE']
        api_version = os.environ['AZURE_API_VERSION']

        os.environ['AZURE_API_VERSION'] = ""
        os.environ['AZURE_API_BASE'] = ""
        os.environ['AZURE_API_KEY'] = ""

        response = embedding(
            model="azure/azure-embedding-model",
            input=["good morning from litellm", "this is another item"],
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
        )
        print(response)


        os.environ['AZURE_API_VERSION'] = api_version
        os.environ['AZURE_API_BASE'] = api_base
        os.environ['AZURE_API_KEY'] = api_key

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

# test_openai_azure_embedding()

# test_openai_embedding()

def test_cohere_embedding():
    try:
        # litellm.set_verbose=True
        response = embedding(
            model="embed-english-v2.0", input=["good morning from litellm", "this is another item"]
        )
        print(f"response:", response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

# test_cohere_embedding()

def test_cohere_embedding3():
    try:
        litellm.set_verbose=True
        response = embedding(
            model="embed-english-v3.0", 
            input=["good morning from litellm", "this is another item"], 
        )
        print(f"response:", response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

# test_cohere_embedding3()

def test_bedrock_embedding():
    try:
        response = embedding(
            model="amazon.titan-embed-text-v1", input=["good morning from litellm, attempting to embed data",
                                                       "lets test a second string for good measure"]
        )
        print(f"response:", response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
# test_bedrock_embedding()

# comment out hf tests - since hf endpoints are unstable
def test_hf_embedding():
    try:
        # huggingface/microsoft/codebert-base
        # huggingface/facebook/bart-large
        response = embedding(
            model="huggingface/sentence-transformers/all-MiniLM-L6-v2", input=["good morning from litellm", "this is another item"]
        )
        print(f"response:", response)
    except Exception as e:
        # Note: Huggingface inference API is unstable and fails with "model loading errors all the time"
        pass
# test_hf_embedding()

# test async embeddings
def test_aembedding():
    try:
        import asyncio
        async def embedding_call():
            try:
                response = await litellm.aembedding(
                    model="text-embedding-ada-002", 
                    input=["good morning from litellm", "this is another item"]
                )
                print(response)
            except Exception as e:
                pytest.fail(f"Error occurred: {e}")
        asyncio.run(embedding_call())
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
                    input=["good morning from litellm", "this is another item"]
                )
                print(response)
            except Exception as e:
                pytest.fail(f"Error occurred: {e}")
        asyncio.run(embedding_call())
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

# test_aembedding_azure()

def test_sagemaker_embeddings(): 
    try: 
        response = litellm.embedding(model="sagemaker/berri-benchmarking-gpt-j-6b-fp16", input=["good morning from litellm", "this is another item"])
        print(f"response: {response}")
    except Exception as e: 
        pytest.fail(f"Error occurred: {e}")
test_sagemaker_embeddings()
# def local_proxy_embeddings():
#     litellm.set_verbose=True
#     response = embedding(
#             model="openai/custom_embedding", 
#             input=["good morning from litellm"],
#             api_base="http://0.0.0.0:8000/"
#         )
#     print(response)

# local_proxy_embeddings()
