import sys, os
import traceback
from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
import litellm
from litellm import embedding, completion, text_completion, completion_cost

user_message = "Write a short poem about the sky"
messages = [{"content": user_message, "role": "user"}]


def logger_fn(user_model_dict):
    print(f"user_model_dict: {user_model_dict}")


def test_completion_custom_provider_model_name():
    try:
        response = completion(
            model="together_ai/togethercomputer/llama-2-70b-chat",
            messages=messages,
            logger_fn=logger_fn,
        )
        # Add any assertions here to check the response
        print(response)
        print(response['choices'][0]['finish_reason'])
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_custom_provider_model_name()


def test_completion_claude():
    try:
        response = completion(
            model="claude-instant-1", messages=messages, logger_fn=logger_fn
        )
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
# test_completion_claude()
# aleph alpha
# def test_completion_aleph_alpha():
#     try:
#         response = completion(
#             model="luminous-base", messages=messages, logger_fn=logger_fn
#         )
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")
# test_completion_aleph_alpha()


# def test_completion_aleph_alpha_control_models():
#     try:
#         response = completion(
#             model="luminous-base-control", messages=messages, logger_fn=logger_fn
#         )
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")
# test_completion_aleph_alpha_control_models()

def test_completion_with_litellm_call_id():
    try:
        litellm.use_client = False
        response = completion(
            model="gpt-3.5-turbo", messages=messages)
        print(response)
        if 'litellm_call_id' in response:
            pytest.fail(f"Error occurred: litellm_call_id in response objects")
        
        litellm.use_client = True
        response2 = completion(
            model="gpt-3.5-turbo", messages=messages)
        
        if 'litellm_call_id' not in response2:
            pytest.fail(f"Error occurred: litellm_call_id not in response object when use_client = True")
        # Add any assertions here to check the response
        print(response2)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

# commenting out as this is a flaky test on circle ci
# def test_completion_nlp_cloud():
#     try:
#         messages = [
#             {"role": "system", "content": "You are a helpful assistant."},
#             {
#                 "role": "user",
#                 "content": "how does a court case get to the Supreme Court?",
#             },
#         ]
#         response = completion(model="dolphin", messages=messages, logger_fn=logger_fn)
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# test_completion_nlp_cloud()
# def test_completion_hf_api():
#     try:
#         user_message = "write some code to find the sum of two numbers"
#         messages = [{ "content": user_message,"role": "user"}]
#         api_base = "https://ecd4sb5n09bo4ei2.us-east-1.aws.endpoints.huggingface.cloud"
#         response = completion(model="togethercomputer/LLaMA-2-7B-32K", messages=messages, custom_llm_provider="huggingface", api_base=api_base, logger_fn=logger_fn)
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         if "loading" in str(e):
#             pass
#         pytest.fail(f"Error occurred: {e}")

# test_completion_hf_api()

# def test_completion_hf_deployed_api():
#     try:
#         user_message = "There's a llama in my garden 😱 What should I do?"
#         messages = [{ "content": user_message,"role": "user"}]
#         response = completion(model="huggingface/https://ji16r2iys9a8rjk2.us-east-1.aws.endpoints.huggingface.cloud", messages=messages, logger_fn=logger_fn)
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")


def test_completion_cohere(): # commenting for now as the cohere endpoint is being flaky
    try:
        response = completion(
            model="command-nightly",
            messages=messages,
            max_tokens=100,
            logit_bias={40: 10},
            logger_fn=logger_fn
        )
        # Add any assertions here to check the response
        print(response)
        response_str = response["choices"][0]["message"]["content"]
        response_str_2 = response.choices[0].message.content
        if type(response_str) != str:
            pytest.fail(f"Error occurred: {e}")
        if type(response_str_2) != str:
            pytest.fail(f"Error occurred: {e}")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

# test_completion_cohere()


def test_completion_openai():
    try:
        litellm.api_key = os.environ['OPENAI_API_KEY']
        response = completion(model="gpt-3.5-turbo", messages=messages)
        
        response_str = response["choices"][0]["message"]["content"]
        response_str_2 = response.choices[0].message.content
        print("response\n", response)
        cost = completion_cost(completion_response=response)
        print("Cost for completion call with gpt-3.5-turbo: ", f"${float(cost):.10f}")
        assert response_str == response_str_2
        assert type(response_str) == str
        assert len(response_str) > 1

        litellm.api_key = None
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
# test_completion_openai()


def test_completion_openai_prompt():
    try:
        response = text_completion(
            model="gpt-3.5-turbo", prompt="What's the weather in SF?"
        )
        response_str = response["choices"][0]["message"]["content"]
        response_str_2 = response.choices[0].message.content
        print(response)
        assert response_str == response_str_2
        assert type(response_str) == str
        assert len(response_str) > 1
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_completion_text_openai():
    try:
        # litellm.set_verbose=True
        response = completion(model="text-davinci-003", messages=messages)
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_completion_openai_with_optional_params():
    try:
        response = completion(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.5,
            top_p=0.1,
            user="ishaan_dev@berri.ai",
        )
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_completion_openai_litellm_key():
    try:
        litellm.api_key = os.environ['OPENAI_API_KEY']

        # ensure key is set to None in .env and in openai.api_key
        os.environ['OPENAI_API_KEY'] = ""
        import openai
        openai.api_key = ""
        ##########################################################

        response = completion(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.5,
            top_p=0.1,
            max_tokens=10,
            user="ishaan_dev@berri.ai",
        )
        # Add any assertions here to check the response
        print(response)

        ###### reset environ key
        os.environ['OPENAI_API_KEY'] = litellm.api_key

        ##### unset litellm var
        litellm.api_key = None
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

# test_completion_openai_litellm_key()

# commented out for now, as openrouter is quite flaky - causing our deployments to fail. Please run this before pushing changes.
# def test_completion_openrouter():
#     try:
#         response = completion(
#             model="google/palm-2-chat-bison",
#             messages=messages,
#             temperature=0.5,
#             top_p=0.1,
#         )
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

def test_completion_openai_with_more_optional_params():
    try:
        response = completion(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.5,
            top_p=0.1,
            n=2,
            max_tokens=150,
            presence_penalty=0.5,
            frequency_penalty=-0.5,
            logit_bias={123: 5},
            user="ishaan_dev@berri.ai",
        )
        # Add any assertions here to check the response
        print(response)
        response_str = response["choices"][0]["message"]["content"]
        response_str_2 = response.choices[0].message.content
        print(response["choices"][0]["message"]["content"])
        print(response.choices[0].message.content)
        if type(response_str) != str:
            pytest.fail(f"Error occurred: {e}")
        if type(response_str_2) != str:
            pytest.fail(f"Error occurred: {e}")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# def test_completion_openai_azure_with_functions():
#     function1 = [
#         {
#             "name": "get_current_weather",
#             "description": "Get the current weather in a given location",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "location": {
#                         "type": "string",
#                         "description": "The city and state, e.g. San Francisco, CA",
#                     },
#                     "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
#                 },
#                 "required": ["location"],
#             },
#         }
#     ]
#     try:
#         response = completion(
#             model="azure/chatgpt-functioncalling", messages=messages, stream=True
#         )
#         # Add any assertions here to check the response
#         print(response)
#         for chunk in response:
#             print(chunk)
#             print(chunk["choices"][0]["finish_reason"])
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")
# test_completion_openai_azure_with_functions()


def test_completion_azure():
    try:
        print("azure gpt-3.5 test\n\n")
        response = completion(
            model="azure/chatgpt-v-2",
            messages=messages,
        )
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

# test_completion_azure()

# new azure test for using litellm. vars, 
# use the following vars in this test and make an azure_api_call
#  litellm.api_type = self.azure_api_type 
#  litellm.api_base = self.azure_api_base 
#  litellm.api_version = self.azure_api_version 
#  litellm.api_key = self.api_key 
def test_completion_azure_with_litellm_key():
    try:
        print("azure gpt-3.5 test\n\n")
        import openai


        #### set litellm vars
        litellm.api_type = "azure"
        litellm.api_base = os.environ['AZURE_API_BASE']
        litellm.api_version = os.environ['AZURE_API_VERSION']
        litellm.api_key = os.environ['AZURE_API_KEY']

        ######### UNSET ENV VARs for this ################
        os.environ['AZURE_API_BASE'] = ""
        os.environ['AZURE_API_VERSION'] = ""
        os.environ['AZURE_API_KEY'] = ""

        ######### UNSET OpenAI vars for this ##############
        openai.api_type = ""
        openai.api_base = "gm"
        openai.api_version = "333"
        openai.api_key = "ymca"

        response = completion(
            model="azure/chatgpt-v-2",
            messages=messages,
        )
        # Add any assertions here to check the response
        print(response)


        ######### RESET ENV VARs for this ################
        os.environ['AZURE_API_BASE'] = litellm.api_base
        os.environ['AZURE_API_VERSION'] = litellm.api_version
        os.environ['AZURE_API_KEY'] = litellm.api_key

        ######### UNSET litellm vars
        litellm.api_type = None
        litellm.api_base = None
        litellm.api_version = None
        litellm.api_key = None

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
# test_completion_azure()


def test_completion_azure_deployment_id():
    try:
        response = completion(
            deployment_id="chatgpt-v-2",
            model="gpt-3.5-turbo",
            messages=messages,
        )
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
# test_completion_azure_deployment_id()

# Replicate API endpoints are unstable -> throw random CUDA errors -> this means our tests can fail even if our tests weren't incorrect.

# def test_completion_replicate_llama_2():
#     model_name = "replicate/llama-2-70b-chat:2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf"
#     try:
#         response = completion(
#             model=model_name, 
#             messages=messages, 
#             max_tokens=20,
#             custom_llm_provider="replicate"
#         )
#         print(response)
#         cost = completion_cost(completion_response=response)
#         print("Cost for completion call with llama-2: ", f"${float(cost):.10f}")
#         # Add any assertions here to check the response
#         response_str = response["choices"][0]["message"]["content"]
#         print(response_str)
#         if type(response_str) != str:
#             pytest.fail(f"Error occurred: {e}")
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")
# test_completion_replicate_llama_2()

def test_completion_replicate_vicuna():
    model_name = "replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b"
    try:
        response = completion(
            model=model_name, 
            messages=messages, 
            custom_llm_provider="replicate",
            temperature=0.1,
            max_tokens=20,
        )
        print(response)
        # Add any assertions here to check the response
        response_str = response["choices"][0]["message"]["content"]
        print(response_str)
        if type(response_str) != str:
            pytest.fail(f"Error occurred: {e}")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

# def test_completion_replicate_stability_stream():
#     model_name = "stability-ai/stablelm-tuned-alpha-7b:c49dae362cbaecd2ceabb5bd34fdb68413c4ff775111fea065d259d577757beb"
#     try:
#         response = completion(
#             model=model_name,
#             messages=messages,
#             # stream=True,
#             custom_llm_provider="replicate",
#         )
#         # print(response)
#         # Add any assertions here to check the response
#         # for chunk in response:
#         #     print(chunk["choices"][0]["delta"])
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")
# test_completion_replicate_stability_stream()





######## Test TogetherAI ########
def test_completion_together_ai():
    model_name = "togethercomputer/llama-2-70b-chat"
    try:
        response = completion(model=model_name, messages=messages, max_tokens=256, logger_fn=logger_fn)
        # Add any assertions here to check the response
        print(response)
        cost = completion_cost(completion_response=response)
        print("Cost for completion call together-computer/llama-2-70b: ", f"${float(cost):.10f}")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

# test_completion_together_ai()
# def test_customprompt_together_ai():
#     try:
#         litellm.register_prompt_template(
#             model="OpenAssistant/llama2-70b-oasst-sft-v10",
#             roles={"system":"<|im_start|>system", "assistant":"<|im_start|>assistant", "user":"<|im_start|>user"}, # tell LiteLLM how you want to map the openai messages to this model
#             pre_message_sep= "\n",
#             post_message_sep= "\n"
#         )
#         response = completion(model="together_ai/OpenAssistant/llama2-70b-oasst-sft-v10", messages=messages)
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

def test_completion_sagemaker():
    try:
        response = completion(
            model="sagemaker/jumpstart-dft-meta-textgeneration-llama-2-7b", 
            messages=messages,
            temperature=0.2,
            max_tokens=80,
            logger_fn=logger_fn
        )
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_completion_bedrock_titan():
    try:
        response = completion(
            model="bedrock/amazon.titan-tg1-large", 
            messages=messages,
            temperature=0.2,
            max_tokens=200,
            top_p=0.8,
            logger_fn=logger_fn
        )
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
# test_completion_bedrock_titan()


def test_completion_bedrock_ai21():
    try:
        litellm.set_verbose = False
        response = completion(
            model="bedrock/ai21.j2-mid", 
            messages=messages,
            temperature=0.2,
            top_p=0.2,
            max_tokens=20
        )
        # Add any assertions here to check the response 
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


######## Test VLLM ########
# def test_completion_vllm():
#     try:
#         response = completion(
#             model="vllm/facebook/opt-125m", 
#             messages=messages,
#             temperature=0.2,
#             max_tokens=80,
#         )
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# test_completion_vllm()

# def test_completion_hosted_chatCompletion():
#     # this tests calling a server where vllm is hosted
#     # this should make an openai.Completion() call to the specified api_base
#     # send a request to this proxy server: https://replit.com/@BerriAI/openai-proxy#main.py
#     # it checks if model == facebook/opt-125m and returns test passed
#     try:
#         litellm.set_verbose = True
#         response = completion(
#             model="facebook/opt-125m", 
#             messages=messages,
#             temperature=0.2,
#             max_tokens=80,
#             api_base="https://openai-proxy.berriai.repl.co",
#             custom_llm_provider="openai"
#         )
#         print(response)

#         if response['choices'][0]['message']['content'] != "passed":
#             # see https://replit.com/@BerriAI/openai-proxy#main.py
#             pytest.fail(f"Error occurred: proxy server did not respond")
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# test_completion_hosted_chatCompletion()

# def test_completion_custom_api_base():
#     try:
#         response = completion(
#             model="custom/meta-llama/Llama-2-13b-hf", 
#             messages=messages,
#             temperature=0.2,
#             max_tokens=10,
#             api_base="https://api.autoai.dev/inference",
#             request_timeout=300,
#         )
#         # Add any assertions here to check the response
#         print("got response\n", response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# test_completion_custom_api_base()

# def test_vertex_ai():
#     litellm.vertex_project = "hardy-device-386718"
#     litellm.vertex_location = "us-central1"
#     test_models = litellm.vertex_chat_models + litellm.vertex_code_chat_models + litellm.vertex_text_models + litellm.vertex_code_text_models
#     for model in test_models:
#         try:
#             print("making request", model)
#             response = completion(model=model, messages=[{"role": "user", "content": "write code for saying hi"}])
#             print(response)
#         except Exception as e:
#             pytest.fail(f"Error occurred: {e}")
# test_vertex_ai()

# def test_vertex_ai_stream():
#     litellm.vertex_project = "hardy-device-386718"
#     litellm.vertex_location = "us-central1"
#     test_models = litellm.vertex_chat_models + litellm.vertex_code_chat_models + litellm.vertex_text_models + litellm.vertex_code_text_models
#     for model in test_models:
#         try:
#             print("making request", model)
#             response = completion(model=model, messages=[{"role": "user", "content": "write code for saying hi"}], stream=True)
#             print(response)
#             for chunk in response:
#                 print(chunk)
#                 # pass
#         except Exception as e:
#             pytest.fail(f"Error occurred: {e}")
# test_vertex_ai_stream() 


def test_completion_with_fallbacks():
    fallbacks = ["gpt-3.5-turb", "gpt-3.5-turbo", "command-nightly"]
    try:
        response = completion(
            model="bad-model", messages=messages, force_timeout=120, fallbacks=fallbacks
        )
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

# def test_baseten():
#     try:

#         response = completion(model="baseten/7qQNLDB", messages=messages, logger_fn=logger_fn)
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# test_baseten()
# def test_baseten_falcon_7bcompletion():
#     model_name = "qvv0xeq"
#     try:
#         response = completion(model=model_name, messages=messages, custom_llm_provider="baseten")
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")
# test_baseten_falcon_7bcompletion()

# def test_baseten_falcon_7bcompletion_withbase():
#     model_name = "qvv0xeq"
#     litellm.api_base = "https://app.baseten.co"
#     try:
#         response = completion(model=model_name, messages=messages)
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")
#     litellm.api_base = None

# test_baseten_falcon_7bcompletion_withbase()


# def test_baseten_wizardLMcompletion_withbase():
#     model_name = "q841o8w"
#     litellm.api_base = "https://app.baseten.co"
#     try:
#         response = completion(model=model_name, messages=messages)
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# test_baseten_wizardLMcompletion_withbase()

# def test_baseten_mosaic_ML_completion_withbase():
#     model_name = "31dxrj3"
#     litellm.api_base = "https://app.baseten.co"
#     try:
#         response = completion(model=model_name, messages=messages)
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")


#### Test A121 ###################
def test_completion_ai21():
    model_name = "j2-light"
    try:
        response = completion(model=model_name, messages=messages)
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

# test_completion_ai21()
# test config file with completion #
# def test_completion_openai_config():
#     try:
#         litellm.config_path = "../config.json"
#         litellm.set_verbose = True
#         response = litellm.config_completion(messages=messages)
#         # Add any assertions here to check the response
#         print(response)
#         litellm.config_path = None
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")


# import asyncio
# def test_completion_together_ai_stream():
#     user_message = "Write 1pg about YC & litellm"
#     messages = [{ "content": user_message,"role": "user"}]
#     try:
#         response = completion(model="togethercomputer/llama-2-70b-chat", messages=messages, stream=True, max_tokens=800)
#         print(response)
#         asyncio.run(get_response(response))
#         # print(string_response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")


# async def get_response(generator):
#     async for elem in generator:
#         print(elem)
#     return

# test_completion_together_ai_stream()
