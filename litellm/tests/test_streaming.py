#### What this tests ####
#    This tests streaming for the completion endpoint

import sys, os
import traceback
import time 
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import completion
litellm.logging = False
litellm.set_verbose = False

score = 0


def logger_fn(model_call_object: dict):
    return
    print(f"model call details: {model_call_object}")


user_message = "Hello, how are you?"
messages = [{"content": user_message, "role": "user"}]

# test on openai completion call
# try:
#     response = completion(
#         model="gpt-3.5-turbo", messages=messages, stream=True, logger_fn=logger_fn
#     )
#     complete_response = ""
#     start_time = time.time()
#     for chunk in response:
#         chunk_time = time.time() 
#         print(f"time since initial request: {chunk_time - start_time:.5f}")
#         print(chunk["choices"][0]["delta"])
#         complete_response += chunk["choices"][0]["delta"]["content"]
#     if complete_response == "": 
#         raise Exception("Empty response received")
# except:
#     print(f"error occurred: {traceback.format_exc()}")
#     pass


# # test on azure completion call
# try:
#     response = completion(
#         model="azure/chatgpt-test", messages=messages, stream=True, logger_fn=logger_fn
#     )
#     response = ""
#     start_time = time.time()
#     for chunk in response:
#         chunk_time = time.time() 
#         print(f"time since initial request: {chunk_time - start_time:.2f}")
#         print(chunk["choices"][0]["delta"])
#         response += chunk["choices"][0]["delta"]
#     if response == "": 
#         raise Exception("Empty response received")
# except:
#     print(f"error occurred: {traceback.format_exc()}")
#     pass


# test on anthropic completion call
try:
    response = completion(
        model="claude-instant-1", messages=messages, stream=True, logger_fn=logger_fn
    )
    complete_response = ""
    start_time = time.time()
    for chunk in response:
        chunk_time = time.time()
        print(f"time since initial request: {chunk_time - start_time:.5f}")
        print(chunk["choices"][0]["delta"])
        complete_response += chunk["choices"][0]["delta"]["content"]
    if complete_response == "": 
        raise Exception("Empty response received")
except:
    print(f"error occurred: {traceback.format_exc()}")
    pass


# # test on huggingface completion call
# try:
#     start_time = time.time()
#     response = completion(
#         model="gpt-3.5-turbo", messages=messages, stream=True, logger_fn=logger_fn
#     )
#     complete_response = ""
#     for chunk in response:
#         chunk_time = time.time() 
#         print(f"time since initial request: {chunk_time - start_time:.2f}")
#         print(chunk["choices"][0]["delta"])
#         complete_response += chunk["choices"][0]["delta"]["content"] if len(chunk["choices"][0]["delta"].keys()) > 0 else "" 
#     if complete_response == "": 
#         raise Exception("Empty response received")
# except:
#     print(f"error occurred: {traceback.format_exc()}")
#     pass

# test on together ai completion call - replit-code-3b
try:
    start_time = time.time()
    response = completion(
        model="Replit-Code-3B", messages=messages, logger_fn=logger_fn, stream= True
    )
    complete_response = ""
    print(f"returned response object: {response}")
    for chunk in response:
        chunk_time = time.time() 
        print(f"time since initial request: {chunk_time - start_time:.2f}")
        print(chunk["choices"][0]["delta"])
        complete_response += chunk["choices"][0]["delta"]["content"] if len(chunk["choices"][0]["delta"].keys()) > 0 else "" 
    if complete_response == "": 
        raise Exception("Empty response received")
except:
    print(f"error occurred: {traceback.format_exc()}")
    pass

# test on together ai completion call - starcoder
try:
    start_time = time.time()
    response = completion(
        model="together_ai/bigcode/starcoder", messages=messages, logger_fn=logger_fn, stream= True
    )
    complete_response = ""
    print(f"returned response object: {response}")
    for chunk in response:
        chunk_time = time.time() 
        complete_response += chunk["choices"][0]["delta"]["content"] if len(chunk["choices"][0]["delta"].keys()) > 0 else "" 
        if len(complete_response) > 0:
            print(complete_response)
    if complete_response == "": 
        raise Exception("Empty response received")
except:
    print(f"error occurred: {traceback.format_exc()}")
    pass


# # test on azure completion call
# try:
#     response = completion(
#         model="azure/chatgpt-test", messages=messages, stream=True, logger_fn=logger_fn
#     )
#     response = ""
#     for chunk in response:
#         chunk_time = time.time() 
#         print(f"time since initial request: {chunk_time - start_time:.2f}")
#         print(chunk["choices"][0]["delta"])
#         response += chunk["choices"][0]["delta"]
#     if response == "": 
#         raise Exception("Empty response received")
# except:
#     print(f"error occurred: {traceback.format_exc()}")
#     pass


# # test on anthropic completion call
# try:
#     response = completion(
#         model="claude-instant-1", messages=messages, stream=True, logger_fn=logger_fn
#     )
#     response = ""
#     for chunk in response:
#         chunk_time = time.time() 
#         print(f"time since initial request: {chunk_time - start_time:.2f}")
#         print(chunk["choices"][0]["delta"])
#         response += chunk["choices"][0]["delta"]
#     if response == "": 
#         raise Exception("Empty response received")
# except:
#     print(f"error occurred: {traceback.format_exc()}")
#     pass


# # # test on huggingface completion call
# # try:
# #     response = completion(
# #         model="meta-llama/Llama-2-7b-chat-hf",
# #         messages=messages,
# #         custom_llm_provider="huggingface",
# #         custom_api_base="https://s7c7gytn18vnu4tw.us-east-1.aws.endpoints.huggingface.cloud",
# #         stream=True,
# #         logger_fn=logger_fn,
# #     )
# #     for chunk in response:
# #         print(chunk["choices"][0]["delta"])
# #     score += 1
# # except:
# #     print(f"error occurred: {traceback.format_exc()}")
# #     pass
