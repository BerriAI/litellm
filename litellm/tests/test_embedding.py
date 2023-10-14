import sys, os
import traceback
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import embedding, completion

litellm.set_verbose = False

def test_openai_embedding():
    try:
        response = embedding(
            model="text-embedding-ada-002", input=["good morning from litellm", "this is another item"]
        )
        print(response)
        # Add any assertions here to check the response
        # print(f"response: {str(response)}")
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

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

# test_openai_azure_embedding_simple()

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
        response = embedding(
            model="embed-english-v2.0", input=["good morning from litellm", "this is another item"]
        )
        print(f"response:", response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

# test_cohere_embedding()

# comment out hf tests - since hf endpoints are unstable
# def test_hf_embedding():
#     try:
#         # huggingface/microsoft/codebert-base
#         # huggingface/facebook/bart-large
#         response = embedding(
#             model="huggingface/BAAI/bge-large-zh", input=["good morning from litellm", "this is another item"]
#         )
#         print(f"response:", response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")
# test_hf_embedding()


