#### What this tests ####
#    This tests setting provider specific configs across providers
# There are 2 types of tests - changing config dynamically or by setting class variables

import os
import sys
import traceback

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
from litellm import RateLimitError, completion

#  Huggingface - Expensive to deploy models and keep them running. Maybe we can try doing this via baseten??
# def hf_test_completion_tgi():
#     litellm.HuggingfaceConfig(max_new_tokens=200)
#     litellm.set_verbose=True
#     try:
#         # OVERRIDE WITH DYNAMIC MAX TOKENS
#         response_1 = litellm.completion(
#             model="huggingface/mistralai/Mistral-7B-Instruct-v0.1",
#             messages=[{ "content": "Hello, how are you?","role": "user"}],
#             api_base="https://n9ox93a8sv5ihsow.us-east-1.aws.endpoints.huggingface.cloud",
#             max_tokens=10
#         )
#         # Add any assertions here to check the response
#         print(response_1)
#         response_1_text = response_1.choices[0].message.content

#         # USE CONFIG TOKENS
#         response_2 = litellm.completion(
#             model="huggingface/mistralai/Mistral-7B-Instruct-v0.1",
#             messages=[{ "content": "Hello, how are you?","role": "user"}],
#             api_base="https://n9ox93a8sv5ihsow.us-east-1.aws.endpoints.huggingface.cloud",
#         )
#         # Add any assertions here to check the response
#         print(response_2)
#         response_2_text = response_2.choices[0].message.content

#         assert len(response_2_text) > len(response_1_text)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")
# hf_test_completion_tgi()

# Anthropic


def claude_test_completion():
    litellm.AnthropicConfig(max_tokens_to_sample=200)
    # litellm.set_verbose=True
    try:
        # OVERRIDE WITH DYNAMIC MAX TOKENS
        response_1 = litellm.completion(
            model="claude-instant-1.2",
            messages=[{"content": "Hello, how are you?", "role": "user"}],
            max_tokens=10,
        )
        # Add any assertions here to check the response
        print(response_1)
        response_1_text = response_1.choices[0].message.content

        # USE CONFIG TOKENS
        response_2 = litellm.completion(
            model="claude-instant-1.2",
            messages=[{"content": "Hello, how are you?", "role": "user"}],
        )
        # Add any assertions here to check the response
        print(response_2)
        response_2_text = response_2.choices[0].message.content

        assert len(response_2_text) > len(response_1_text)

        try:
            response_3 = litellm.completion(
                model="claude-instant-1.2",
                messages=[{"content": "Hello, how are you?", "role": "user"}],
                n=2,
            )

        except Exception as e:
            print(e)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# claude_test_completion()

#  Replicate


def replicate_test_completion():
    litellm.ReplicateConfig(max_new_tokens=200)
    # litellm.set_verbose=True
    try:
        # OVERRIDE WITH DYNAMIC MAX TOKENS
        response_1 = litellm.completion(
            model="meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
            messages=[{"content": "Hello, how are you?", "role": "user"}],
            max_tokens=10,
        )
        # Add any assertions here to check the response
        print(response_1)
        response_1_text = response_1.choices[0].message.content

        # USE CONFIG TOKENS
        response_2 = litellm.completion(
            model="meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
            messages=[{"content": "Hello, how are you?", "role": "user"}],
        )
        # Add any assertions here to check the response
        print(response_2)
        response_2_text = response_2.choices[0].message.content

        assert len(response_2_text) > len(response_1_text)
        try:
            response_3 = litellm.completion(
                model="meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
                messages=[{"content": "Hello, how are you?", "role": "user"}],
                n=2,
            )
        except:
            pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# replicate_test_completion()

#  Cohere


def cohere_test_completion():
    # litellm.CohereConfig(max_tokens=200)
    litellm.set_verbose = True
    try:
        # OVERRIDE WITH DYNAMIC MAX TOKENS
        response_1 = litellm.completion(
            model="command-nightly",
            messages=[{"content": "Hello, how are you?", "role": "user"}],
            max_tokens=10,
        )
        response_1_text = response_1.choices[0].message.content

        # USE CONFIG TOKENS
        response_2 = litellm.completion(
            model="command-nightly",
            messages=[{"content": "Hello, how are you?", "role": "user"}],
        )
        response_2_text = response_2.choices[0].message.content

        assert len(response_2_text) > len(response_1_text)

        response_3 = litellm.completion(
            model="command-nightly",
            messages=[{"content": "Hello, how are you?", "role": "user"}],
            n=2,
        )
        assert len(response_3.choices) > 1
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# cohere_test_completion()

#  AI21


def ai21_test_completion():
    litellm.AI21Config(maxTokens=10)
    litellm.set_verbose = True
    try:
        # OVERRIDE WITH DYNAMIC MAX TOKENS
        response_1 = litellm.completion(
            model="j2-mid",
            messages=[
                {
                    "content": "Hello, how are you? Be as verbose as possible",
                    "role": "user",
                }
            ],
            max_tokens=100,
        )
        response_1_text = response_1.choices[0].message.content
        print(f"response_1_text: {response_1_text}")

        # USE CONFIG TOKENS
        response_2 = litellm.completion(
            model="j2-mid",
            messages=[
                {
                    "content": "Hello, how are you? Be as verbose as possible",
                    "role": "user",
                }
            ],
        )
        response_2_text = response_2.choices[0].message.content
        print(f"response_2_text: {response_2_text}")

        assert len(response_2_text) < len(response_1_text)

        response_3 = litellm.completion(
            model="j2-light",
            messages=[{"content": "Hello, how are you?", "role": "user"}],
            n=2,
        )
        assert len(response_3.choices) > 1
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# ai21_test_completion()

#  TogetherAI


def togetherai_test_completion():
    litellm.TogetherAIConfig(max_tokens=10)
    litellm.set_verbose = True
    try:
        # OVERRIDE WITH DYNAMIC MAX TOKENS
        response_1 = litellm.completion(
            model="together_ai/togethercomputer/llama-2-70b-chat",
            messages=[
                {
                    "content": "Hello, how are you? Be as verbose as possible",
                    "role": "user",
                }
            ],
            max_tokens=100,
        )
        response_1_text = response_1.choices[0].message.content
        print(f"response_1_text: {response_1_text}")

        # USE CONFIG TOKENS
        response_2 = litellm.completion(
            model="together_ai/togethercomputer/llama-2-70b-chat",
            messages=[
                {
                    "content": "Hello, how are you? Be as verbose as possible",
                    "role": "user",
                }
            ],
        )
        response_2_text = response_2.choices[0].message.content
        print(f"response_2_text: {response_2_text}")

        assert len(response_2_text) < len(response_1_text)

        try:
            response_3 = litellm.completion(
                model="together_ai/togethercomputer/llama-2-70b-chat",
                messages=[{"content": "Hello, how are you?", "role": "user"}],
                n=2,
            )
            pytest.fail(f"Error not raised when n=2 passed to provider")
        except:
            pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# togetherai_test_completion()

#  Palm


# palm_test_completion()

#  NLP Cloud


def nlp_cloud_test_completion():
    litellm.NLPCloudConfig(max_length=10)
    # litellm.set_verbose=True
    try:
        # OVERRIDE WITH DYNAMIC MAX TOKENS
        response_1 = litellm.completion(
            model="dolphin",
            messages=[
                {
                    "content": "Hello, how are you? Be as verbose as possible",
                    "role": "user",
                }
            ],
            max_tokens=100,
        )
        response_1_text = response_1.choices[0].message.content
        print(f"response_1_text: {response_1_text}")

        # USE CONFIG TOKENS
        response_2 = litellm.completion(
            model="dolphin",
            messages=[
                {
                    "content": "Hello, how are you? Be as verbose as possible",
                    "role": "user",
                }
            ],
        )
        response_2_text = response_2.choices[0].message.content
        print(f"response_2_text: {response_2_text}")

        assert len(response_2_text) < len(response_1_text)

        try:
            response_3 = litellm.completion(
                model="dolphin",
                messages=[{"content": "Hello, how are you?", "role": "user"}],
                n=2,
            )
            pytest.fail(f"Error not raised when n=2 passed to provider")
        except:
            pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# nlp_cloud_test_completion()

#  AlephAlpha


def aleph_alpha_test_completion():
    litellm.AlephAlphaConfig(maximum_tokens=10)
    # litellm.set_verbose=True
    try:
        # OVERRIDE WITH DYNAMIC MAX TOKENS
        response_1 = litellm.completion(
            model="luminous-base",
            messages=[
                {
                    "content": "Hello, how are you? Be as verbose as possible",
                    "role": "user",
                }
            ],
            max_tokens=100,
        )
        response_1_text = response_1.choices[0].message.content
        print(f"response_1_text: {response_1_text}")

        # USE CONFIG TOKENS
        response_2 = litellm.completion(
            model="luminous-base",
            messages=[
                {
                    "content": "Hello, how are you? Be as verbose as possible",
                    "role": "user",
                }
            ],
        )
        response_2_text = response_2.choices[0].message.content
        print(f"response_2_text: {response_2_text}")

        assert len(response_2_text) < len(response_1_text)

        response_3 = litellm.completion(
            model="luminous-base",
            messages=[{"content": "Hello, how are you?", "role": "user"}],
            n=2,
        )

        assert len(response_3.choices) > 1
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# aleph_alpha_test_completion()

#  Petals - calls are too slow, will cause circle ci to fail due to delay. Test locally.
# def petals_completion():
#     litellm.PetalsConfig(max_new_tokens=10)
#     # litellm.set_verbose=True
#     try:
#         # OVERRIDE WITH DYNAMIC MAX TOKENS
#         response_1 = litellm.completion(
#             model="petals/petals-team/StableBeluga2",
#             messages=[{ "content": "Hello, how are you? Be as verbose as possible","role": "user"}],
#             api_base="https://chat.petals.dev/api/v1/generate",
#             max_tokens=100
#         )
#         response_1_text = response_1.choices[0].message.content
#         print(f"response_1_text: {response_1_text}")

#         # USE CONFIG TOKENS
#         response_2 = litellm.completion(
#             model="petals/petals-team/StableBeluga2",
#             api_base="https://chat.petals.dev/api/v1/generate",
#             messages=[{ "content": "Hello, how are you? Be as verbose as possible","role": "user"}],
#         )
#         response_2_text = response_2.choices[0].message.content
#         print(f"response_2_text: {response_2_text}")

#         assert len(response_2_text) < len(response_1_text)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# petals_completion()

#  VertexAI
# We don't have vertex ai configured for circle ci yet -- need to figure this out.
# def vertex_ai_test_completion():
#     litellm.VertexAIConfig(max_output_tokens=10)
#     # litellm.set_verbose=True
#     try:
#         # OVERRIDE WITH DYNAMIC MAX TOKENS
#         response_1 = litellm.completion(
#             model="chat-bison",
#             messages=[{ "content": "Hello, how are you? Be as verbose as possible","role": "user"}],
#             max_tokens=100
#         )
#         response_1_text = response_1.choices[0].message.content
#         print(f"response_1_text: {response_1_text}")

#         # USE CONFIG TOKENS
#         response_2 = litellm.completion(
#             model="chat-bison",
#             messages=[{ "content": "Hello, how are you? Be as verbose as possible","role": "user"}],
#         )
#         response_2_text = response_2.choices[0].message.content
#         print(f"response_2_text: {response_2_text}")

#         assert len(response_2_text) < len(response_1_text)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# vertex_ai_test_completion()

#  Sagemaker


@pytest.mark.skip(reason="AWS Suspended Account")
def sagemaker_test_completion():
    litellm.SagemakerConfig(max_new_tokens=10)
    # litellm.set_verbose=True
    try:
        # OVERRIDE WITH DYNAMIC MAX TOKENS
        response_1 = litellm.completion(
            model="sagemaker/berri-benchmarking-Llama-2-70b-chat-hf-4",
            messages=[
                {
                    "content": "Hello, how are you? Be as verbose as possible",
                    "role": "user",
                }
            ],
            max_tokens=100,
        )
        response_1_text = response_1.choices[0].message.content
        print(f"response_1_text: {response_1_text}")

        # USE CONFIG TOKENS
        response_2 = litellm.completion(
            model="sagemaker/berri-benchmarking-Llama-2-70b-chat-hf-4",
            messages=[
                {
                    "content": "Hello, how are you? Be as verbose as possible",
                    "role": "user",
                }
            ],
        )
        response_2_text = response_2.choices[0].message.content
        print(f"response_2_text: {response_2_text}")

        assert len(response_2_text) < len(response_1_text)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# sagemaker_test_completion()


def test_sagemaker_default_region():
    """
    If no regions are specified in config or in environment, the default region is us-west-2
    """
    mock_response = MagicMock()

    def return_val():
        return {
            "generated_text": "This is a mock response from SageMaker.",
            "id": "cmpl-mockid",
            "object": "text_completion",
            "created": 1629800000,
            "model": "sagemaker/jumpstart-dft-hf-textgeneration1-mp-20240815-185614",
            "choices": [
                {
                    "text": "This is a mock response from SageMaker.",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 8, "total_tokens": 9},
        }

    mock_response.json = return_val
    mock_response.status_code = 200

    with patch(
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        response = litellm.completion(
            model="sagemaker/mock-endpoint",
            messages=[{"content": "Hello, world!", "role": "user"}],
        )
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        args_to_sagemaker = kwargs["json"]
        print("Arguments passed to sagemaker=", args_to_sagemaker)
        print("url=", kwargs["url"])

        assert (
            kwargs["url"]
            == "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/mock-endpoint/invocations"
        )


# test_sagemaker_default_region()


def test_sagemaker_environment_region():
    """
    If a region is specified in the environment, use that region instead of us-west-2
    """
    expected_region = "us-east-1"
    os.environ["AWS_REGION_NAME"] = expected_region
    mock_response = MagicMock()

    def return_val():
        return {
            "generated_text": "This is a mock response from SageMaker.",
            "id": "cmpl-mockid",
            "object": "text_completion",
            "created": 1629800000,
            "model": "sagemaker/jumpstart-dft-hf-textgeneration1-mp-20240815-185614",
            "choices": [
                {
                    "text": "This is a mock response from SageMaker.",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 8, "total_tokens": 9},
        }

    mock_response.json = return_val
    mock_response.status_code = 200

    with patch(
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        response = litellm.completion(
            model="sagemaker/mock-endpoint",
            messages=[{"content": "Hello, world!", "role": "user"}],
        )
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        args_to_sagemaker = kwargs["json"]
        print("Arguments passed to sagemaker=", args_to_sagemaker)
        print("url=", kwargs["url"])

        assert (
            kwargs["url"]
            == f"https://runtime.sagemaker.{expected_region}.amazonaws.com/endpoints/mock-endpoint/invocations"
        )

        del os.environ["AWS_REGION_NAME"]  # cleanup


# test_sagemaker_environment_region()


def test_sagemaker_config_region():
    """
    If a region is specified as part of the optional parameters of the completion, including as
    part of the config file, then use that region instead of us-west-2
    """
    expected_region = "us-east-1"
    mock_response = MagicMock()

    def return_val():
        return {
            "generated_text": "This is a mock response from SageMaker.",
            "id": "cmpl-mockid",
            "object": "text_completion",
            "created": 1629800000,
            "model": "sagemaker/jumpstart-dft-hf-textgeneration1-mp-20240815-185614",
            "choices": [
                {
                    "text": "This is a mock response from SageMaker.",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 8, "total_tokens": 9},
        }

    mock_response.json = return_val
    mock_response.status_code = 200

    with patch(
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:

        response = litellm.completion(
            model="sagemaker/mock-endpoint",
            messages=[{"content": "Hello, world!", "role": "user"}],
            aws_region_name=expected_region,
        )

        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        args_to_sagemaker = kwargs["json"]
        print("Arguments passed to sagemaker=", args_to_sagemaker)
        print("url=", kwargs["url"])

        assert (
            kwargs["url"]
            == f"https://runtime.sagemaker.{expected_region}.amazonaws.com/endpoints/mock-endpoint/invocations"
        )


# test_sagemaker_config_region()


# test_sagemaker_config_and_environment_region()


#  Bedrock


def bedrock_test_completion():
    litellm.AmazonCohereConfig(max_tokens=10)
    # litellm.set_verbose=True
    try:
        # OVERRIDE WITH DYNAMIC MAX TOKENS
        response_1 = litellm.completion(
            model="bedrock/cohere.command-text-v14",
            messages=[
                {
                    "content": "Hello, how are you? Be as verbose as possible",
                    "role": "user",
                }
            ],
            max_tokens=100,
        )
        response_1_text = response_1.choices[0].message.content
        print(f"response_1_text: {response_1_text}")

        # USE CONFIG TOKENS
        response_2 = litellm.completion(
            model="bedrock/cohere.command-text-v14",
            messages=[
                {
                    "content": "Hello, how are you? Be as verbose as possible",
                    "role": "user",
                }
            ],
        )
        response_2_text = response_2.choices[0].message.content
        print(f"response_2_text: {response_2_text}")

        assert len(response_2_text) < len(response_1_text)
    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# bedrock_test_completion()


# OpenAI Chat Completion
def openai_test_completion():
    litellm.OpenAIConfig(max_tokens=10)
    # litellm.set_verbose=True
    try:
        # OVERRIDE WITH DYNAMIC MAX TOKENS
        response_1 = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "content": "Hello, how are you? Be as verbose as possible",
                    "role": "user",
                }
            ],
            max_tokens=100,
        )
        response_1_text = response_1.choices[0].message.content
        print(f"response_1_text: {response_1_text}")

        # USE CONFIG TOKENS
        response_2 = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "content": "Hello, how are you? Be as verbose as possible",
                    "role": "user",
                }
            ],
        )
        response_2_text = response_2.choices[0].message.content
        print(f"response_2_text: {response_2_text}")

        assert len(response_2_text) < len(response_1_text)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# openai_test_completion()


# OpenAI Text Completion
def openai_text_completion_test():
    litellm.OpenAITextCompletionConfig(max_tokens=10)
    # litellm.set_verbose=True
    try:
        # OVERRIDE WITH DYNAMIC MAX TOKENS
        response_1 = litellm.completion(
            model="gpt-3.5-turbo-instruct",
            messages=[
                {
                    "content": "Hello, how are you? Be as verbose as possible",
                    "role": "user",
                }
            ],
            max_tokens=100,
        )
        response_1_text = response_1.choices[0].message.content
        print(f"response_1_text: {response_1_text}")

        # USE CONFIG TOKENS
        response_2 = litellm.completion(
            model="gpt-3.5-turbo-instruct",
            messages=[
                {
                    "content": "Hello, how are you? Be as verbose as possible",
                    "role": "user",
                }
            ],
        )
        response_2_text = response_2.choices[0].message.content
        print(f"response_2_text: {response_2_text}")

        assert len(response_2_text) < len(response_1_text)

        response_3 = litellm.completion(
            model="gpt-3.5-turbo-instruct",
            messages=[{"content": "Hello, how are you?", "role": "user"}],
            n=2,
        )
        assert len(response_3.choices) > 1
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# openai_text_completion_test()


# Azure OpenAI
def azure_openai_test_completion():
    litellm.AzureOpenAIConfig(max_tokens=10)
    # litellm.set_verbose=True
    try:
        # OVERRIDE WITH DYNAMIC MAX TOKENS
        response_1 = litellm.completion(
            model="azure/chatgpt-v-2",
            messages=[
                {
                    "content": "Hello, how are you? Be as verbose as possible",
                    "role": "user",
                }
            ],
            max_tokens=100,
        )
        response_1_text = response_1.choices[0].message.content
        print(f"response_1_text: {response_1_text}")

        # USE CONFIG TOKENS
        response_2 = litellm.completion(
            model="azure/chatgpt-v-2",
            messages=[
                {
                    "content": "Hello, how are you? Be as verbose as possible",
                    "role": "user",
                }
            ],
        )
        response_2_text = response_2.choices[0].message.content
        print(f"response_2_text: {response_2_text}")

        assert len(response_2_text) < len(response_1_text)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# azure_openai_test_completion()
