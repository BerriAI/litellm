import sys, os
import traceback
from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
from litellm.llms.prompt_templates.factory import prompt_factory

def test_prompt_formatting(): 
    try:
        prompt = prompt_factory(model="mistralai/Mistral-7B-Instruct-v0.1", messages=[{"role": "system", "content": "Be a good bot"}, {"role": "user", "content": "Hello world"}])
        assert prompt == "<s>[INST] Be a good bot [/INST]</s> [INST] Hello world [/INST]"
    except Exception as e: 
        pytest.fail(f"An exception occurred: {str(e)}")
# def logger_fn(user_model_dict):
#     return
#     print(f"user_model_dict: {user_model_dict}")

# messages=[{"role": "user", "content": "Write me a function to print hello world"}]

# # test if the first-party prompt templates work 
# def test_huggingface_supported_models():
#     model = "huggingface/WizardLM/WizardCoder-Python-34B-V1.0"
#     response = completion(model=model, messages=messages, max_tokens=256, api_base="https://ji16r2iys9a8rjk2.us-east-1.aws.endpoints.huggingface.cloud", logger_fn=logger_fn)
#     print(response['choices'][0]['message']['content'])
#     return response

# test_huggingface_supported_models()

# # test if a custom prompt template works 
# litellm.register_prompt_template(
# 	model="togethercomputer/LLaMA-2-7B-32K",
# 	roles={"system":"", "assistant":"Assistant:", "user":"User:"},
# 	pre_message_sep= "\n",
# 	post_message_sep= "\n"
# )
# def test_huggingface_custom_model():
#     model = "huggingface/togethercomputer/LLaMA-2-7B-32K"
#     response = completion(model=model, messages=messages, api_base="https://ecd4sb5n09bo4ei2.us-east-1.aws.endpoints.huggingface.cloud", logger_fn=logger_fn)
#     print(response['choices'][0]['message']['content'])
#     return response

# test_huggingface_custom_model()