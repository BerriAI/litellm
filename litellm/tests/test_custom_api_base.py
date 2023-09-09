# import sys, os
# import traceback
# from dotenv import load_dotenv

# load_dotenv()
# import os

# sys.path.insert(
#     0, os.path.abspath("../..")
# )  # Adds the parent directory to the system path
# import litellm
# from litellm import completion


# def logging_fn(model_call_dict):
#     print(f"model call details: {model_call_dict}")


# models = ["gorilla-7b-hf-v1", "gpt-4"]
# custom_llm_provider = None
# messages = [{"role": "user", "content": "Hey,  how's it going?"}]
# for model in models:  # iterate through list
#     api_base = None
#     if model == "gorilla-7b-hf-v1":
#         custom_llm_provider = "custom_openai"
#         api_base = "http://zanino.millennium.berkeley.edu:8000/v1"
#     completion(
#         model=model,
#         messages=messages,
#         custom_llm_provider=custom_llm_provider,
#         api_base=api_base,
#         logger_fn=logging_fn,
#     )
