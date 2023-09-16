#### What this tests ####
#    This tests streaming for the completion endpoint

import sys, os, asyncio
import traceback
import time, pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import completion, acompletion

litellm.logging = False
litellm.set_verbose = False

score = 0


def logger_fn(model_call_object: dict):
    print(f"model call details: {model_call_object}")


user_message = "Hello, how are you?"
messages = [{"content": user_message, "role": "user"}]

def test_completion_cohere_stream():
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "how does a court case get to the Supreme Court?",
            },
        ]
        response = completion(
            model="command-nightly", messages=messages, stream=True, max_tokens=50
        )
        complete_response = ""
        # Add any assertions here to check the response
        for chunk in response:
            print(f"chunk: {chunk}")
            complete_response += chunk["choices"][0]["delta"]["content"]
        if complete_response == "": 
            raise Exception("Empty response received")
        print(f"completion_response: {complete_response}")
    except KeyError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
        
# test on baseten completion call
# try:
#     response = completion(
#         model="baseten/RqgAEn0", messages=messages, logger_fn=logger_fn
#     )
#     print(f"response: {response}")
#     complete_response = ""
#     start_time = time.time()
#     for chunk in response:
#         chunk_time = time.time()
#         print(f"time since initial request: {chunk_time - start_time:.5f}")
#         print(chunk["choices"][0]["delta"])
#         complete_response += chunk["choices"][0]["delta"]["content"]
#     if complete_response == "": 
#         raise Exception("Empty response received")
#     print(f"complete response: {complete_response}")
# except:
#     print(f"error occurred: {traceback.format_exc()}")
#     pass

# test on openai completion call
def test_openai_text_completion_call():
    try:
        response = completion(
            model="text-davinci-003", messages=messages, stream=True, logger_fn=logger_fn
        )
        complete_response = ""
        start_time = time.time()
        for chunk in response:
            chunk_time = time.time()
            print(f"chunk: {chunk}")
            if "content" in chunk["choices"][0]["delta"]:
                complete_response += chunk["choices"][0]["delta"]["content"]
        if complete_response == "": 
            raise Exception("Empty response received")
    except:
        print(f"error occurred: {traceback.format_exc()}")
        pass

# # test on ai21 completion call
def ai21_completion_call():
    try:
        response = completion(
            model="j2-ultra", messages=messages, stream=True, logger_fn=logger_fn
        )
        print(f"response: {response}")
        complete_response = ""
        start_time = time.time()
        for chunk in response:
            chunk_time = time.time()
            print(f"time since initial request: {chunk_time - start_time:.5f}")
            print(chunk)
            if "content" in chunk["choices"][0]["delta"]:
                complete_response += chunk["choices"][0]["delta"]["content"]
        if complete_response == "": 
            raise Exception("Empty response received")
    except:
        print(f"error occurred: {traceback.format_exc()}")
        pass

# test on openai completion call
def test_openai_chat_completion_call():
    try:
        response = completion(
            model="gpt-3.5-turbo", messages=messages, stream=True, logger_fn=logger_fn
        )
        complete_response = ""
        start_time = time.time()
        for chunk in response:
            print(chunk)
            if chunk["choices"][0]["finish_reason"]:
                break
            # if chunk["choices"][0]["delta"]["role"] != "assistant":
            #     raise Exception("invalid role")
            if "content" in chunk["choices"][0]["delta"]:
                complete_response += chunk["choices"][0]["delta"]["content"]
            # print(f'complete_chunk: {complete_response}')
        if complete_response.strip() == "": 
            raise Exception("Empty response received")
    except:
        print(f"error occurred: {traceback.format_exc()}")
        pass

test_openai_chat_completion_call()
async def completion_call():
    try:
        response = completion(
            model="gpt-3.5-turbo", messages=messages, stream=True, logger_fn=logger_fn
        )
        print(f"response: {response}")
        complete_response = ""
        start_time = time.time()
        # Change for loop to async for loop
        async for chunk in response:
            chunk_time = time.time()
            print(f"time since initial request: {chunk_time - start_time:.5f}")
            print(chunk["choices"][0]["delta"])
            if "content" in chunk["choices"][0]["delta"]:
                complete_response += chunk["choices"][0]["delta"]["content"]
        if complete_response == "": 
            raise Exception("Empty response received")
    except:
        print(f"error occurred: {traceback.format_exc()}")
        pass

# asyncio.run(completion_call())

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
def test_together_ai_completion_call_replit():
    try:
        start_time = time.time()
        response = completion(
            model="Replit-Code-3B", messages=messages, logger_fn=logger_fn, stream=True
        )
        complete_response = ""
        print(f"returned response object: {response}")
        for chunk in response:
            chunk_time = time.time()
            print(f"time since initial request: {chunk_time - start_time:.2f}")
            print(chunk["choices"][0]["delta"])
            complete_response += (
                chunk["choices"][0]["delta"]["content"]
                if len(chunk["choices"][0]["delta"].keys()) > 0
                else ""
            )
        if complete_response == "":
            raise Exception("Empty response received")
    except KeyError as e:
        pass
    except:
        print(f"error occurred: {traceback.format_exc()}")
        pass

# # test on together ai completion call - starcoder
def test_together_ai_completion_call_starcoder():
    try:
        start_time = time.time()
        response = completion(
            model="together_ai/bigcode/starcoder",
            messages=messages,
            logger_fn=logger_fn,
            stream=True,
        )
        complete_response = ""
        print(f"returned response object: {response}")
        for chunk in response:
            chunk_time = time.time()
            complete_response += (
                chunk["choices"][0]["delta"]["content"]
                if len(chunk["choices"][0]["delta"].keys()) > 0
                else ""
            )
            if len(complete_response) > 0:
                print(complete_response)
        if complete_response == "":
            raise Exception("Empty response received")
    except KeyError as e:
        pass
    except:
        print(f"error occurred: {traceback.format_exc()}")
        pass

# test on aleph alpha completion call - commented out as it's expensive to run this on circle ci for every build
# def test_aleph_alpha_call():
#     try:
#         start_time = time.time()
#         response = completion(
#             model="luminous-base",
#             messages=messages,
#             logger_fn=logger_fn,
#             stream=True,
#         )
#         complete_response = ""
#         print(f"returned response object: {response}")
#         for chunk in response:
#             chunk_time = time.time()
#             complete_response += (
#                 chunk["choices"][0]["delta"]["content"]
#                 if len(chunk["choices"][0]["delta"].keys()) > 0
#                 else ""
#             )
#             if len(complete_response) > 0:
#                 print(complete_response)
#         if complete_response == "":
#             raise Exception("Empty response received")
#     except:
#         print(f"error occurred: {traceback.format_exc()}")
#         pass
#### Test Async streaming 

# # test on ai21 completion call
async def ai21_async_completion_call():
    try:
        response = completion(
            model="j2-ultra", messages=messages, stream=True, logger_fn=logger_fn
        )
        print(f"response: {response}")
        complete_response = ""
        start_time = time.time()
        # Change for loop to async for loop
        async for chunk in response:
            chunk_time = time.time()
            print(f"time since initial request: {chunk_time - start_time:.5f}")
            print(chunk["choices"][0]["delta"])
            complete_response += chunk["choices"][0]["delta"]["content"]
        if complete_response == "": 
            raise Exception("Empty response received")
    except:
        print(f"error occurred: {traceback.format_exc()}")
        pass