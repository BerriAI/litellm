# ##### THESE TESTS CAN ONLY RUN LOCALLY WITH THE OLLAMA SERVER RUNNING ######
# # https://ollama.ai/

# import sys, os
# import traceback
# from dotenv import load_dotenv
# load_dotenv()
# import os
# sys.path.insert(0, os.path.abspath('../..'))  # Adds the parent directory to the system path
# import pytest
# import litellm
# from litellm import embedding, completion
# import asyncio


# user_message = "respond in 20 words. who are you?"
# messages = [{ "content": user_message,"role": "user"}]

# async def test_ollama_aembeddings():
#     litellm.set_verbose = True
#     input = "The food was delicious and the waiter..."
#     response = await litellm.aembedding(model="ollama/mistral", input=input)
#     print(response)

# asyncio.run(test_ollama_aembeddings())

# def test_ollama_embeddings():
#     litellm.set_verbose = True
#     input = "The food was delicious and the waiter..."
#     response = litellm.embedding(model="ollama/mistral", input=input)
#     print(response)

# test_ollama_embeddings()

# def test_ollama_streaming():
#     try:
#         litellm.set_verbose = False
#         messages = [
#             {"role": "user", "content": "What is the weather like in Boston?"}
#         ]
#         functions = [
#             {
#             "name": "get_current_weather",
#             "description": "Get the current weather in a given location",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                 "location": {
#                     "type": "string",
#                     "description": "The city and state, e.g. San Francisco, CA"
#                 },
#                 "unit": {
#                     "type": "string",
#                     "enum": ["celsius", "fahrenheit"]
#                 }
#                 },
#                 "required": ["location"]
#             }
#             }
#         ]
#         response = litellm.completion(model="ollama/mistral",
#                                              messages=messages,
#                                              functions=functions,
#                                              stream=True)
#         for chunk in response:
#             print(f"CHUNK: {chunk}")
#     except Exception as e:
#         print(e)

# # test_ollama_streaming()

# async def test_async_ollama_streaming():
#     try:
#         litellm.set_verbose = False
#         response = await litellm.acompletion(model="ollama/mistral-openorca",
#                                              messages=[{"role": "user", "content": "Hey, how's it going?"}],
#                                              stream=True)
#         async for chunk in response:
#             print(f"CHUNK: {chunk}")
#     except Exception as e:
#         print(e)

# # asyncio.run(test_async_ollama_streaming())

# def test_completion_ollama():
#     try:
#         litellm.set_verbose = True
#         response = completion(
#             model="ollama/mistral",
#             messages=[{"role": "user", "content": "Hey, how's it going?"}],
#             max_tokens=200,
#             request_timeout = 10,
#             stream=True
#         )
#         for chunk in response:
#             print(chunk)
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# # test_completion_ollama()

# def test_completion_ollama_function_calling():
#     try:
#         litellm.set_verbose = True
#         messages = [
#             {"role": "user", "content": "What is the weather like in Boston?"}
#         ]
#         functions = [
#             {
#             "name": "get_current_weather",
#             "description": "Get the current weather in a given location",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                 "location": {
#                     "type": "string",
#                     "description": "The city and state, e.g. San Francisco, CA"
#                 },
#                 "unit": {
#                     "type": "string",
#                     "enum": ["celsius", "fahrenheit"]
#                 }
#                 },
#                 "required": ["location"]
#             }
#             }
#         ]
#         response = completion(
#             model="ollama/mistral",
#             messages=messages,
#             functions=functions,
#             max_tokens=200,
#             request_timeout = 10,
#         )
#         for chunk in response:
#             print(chunk)
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")
# # test_completion_ollama_function_calling()

# async def async_test_completion_ollama_function_calling():
#     try:
#         litellm.set_verbose = True
#         messages = [
#             {"role": "user", "content": "What is the weather like in Boston?"}
#         ]
#         functions = [
#             {
#             "name": "get_current_weather",
#             "description": "Get the current weather in a given location",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                 "location": {
#                     "type": "string",
#                     "description": "The city and state, e.g. San Francisco, CA"
#                 },
#                 "unit": {
#                     "type": "string",
#                     "enum": ["celsius", "fahrenheit"]
#                 }
#                 },
#                 "required": ["location"]
#             }
#             }
#         ]
#         response = await litellm.acompletion(
#             model="ollama/mistral",
#             messages=messages,
#             functions=functions,
#             max_tokens=200,
#             request_timeout = 10,
#         )
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# # asyncio.run(async_test_completion_ollama_function_calling())


# def test_completion_ollama_with_api_base():
#     try:
#         response = completion(
#             model="ollama/llama2",
#             messages=messages,
#             api_base="http://localhost:11434"
#         )
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# # test_completion_ollama_with_api_base()


# def test_completion_ollama_custom_prompt_template():
#     user_message = "what is litellm?"
#     litellm.register_prompt_template(
#         model="ollama/llama2",
#         roles={
#             "system": {"pre_message": "System: "},
#             "user": {"pre_message": "User: "},
#             "assistant": {"pre_message": "Assistant: "}
#         }
#     )
#     messages = [{ "content": user_message,"role": "user"}]
#     litellm.set_verbose = True
#     try:
#         response = completion(
#             model="ollama/llama2",
#             messages=messages,
#             stream=True
#         )
#         print(response)
#         for chunk in response:
#             print(chunk)
#             # print(chunk['choices'][0]['delta'])

#     except Exception as e:
#         traceback.print_exc()
#         pytest.fail(f"Error occurred: {e}")

# # test_completion_ollama_custom_prompt_template()

# async def test_completion_ollama_async_stream():
#     user_message = "what is the weather"
#     messages = [{ "content": user_message,"role": "user"}]
#     try:
#         response = await litellm.acompletion(
#             model="ollama/llama2",
#             messages=messages,
#             api_base="http://localhost:11434",
#             stream=True
#         )
#         async for chunk in response:
#             print(chunk['choices'][0]['delta'])


#         print("TEST ASYNC NON Stream")
#         response = await litellm.acompletion(
#             model="ollama/llama2",
#             messages=messages,
#             api_base="http://localhost:11434",
#         )
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# # import asyncio
# # asyncio.run(test_completion_ollama_async_stream())


# def prepare_messages_for_chat(text: str) -> list:
#     messages = [
#         {"role": "user", "content": text},
#     ]
#     return messages


# async def ask_question():
#     params = {
#         "messages": prepare_messages_for_chat("What is litellm? tell me 10 things about it who is sihaan.write an essay"),
#         "api_base": "http://localhost:11434",
#         "model": "ollama/llama2",
#         "stream": True,
#     }
#     response = await litellm.acompletion(**params)
#     return response

# async def main():
#     response = await ask_question()
#     async for chunk in response:
#         print(chunk)

#     print("test async completion without streaming")
#     response = await litellm.acompletion(
#         model="ollama/llama2",
#         messages=prepare_messages_for_chat("What is litellm? respond in 2 words"),
#     )
#     print("response", response)


# def test_completion_expect_error():
#     # this tests if we can exception map correctly for ollama
#     print("making ollama request")
#     # litellm.set_verbose=True
#     user_message = "what is litellm?"
#     messages = [{ "content": user_message,"role": "user"}]
#     try:
#         response = completion(
#             model="ollama/invalid",
#             messages=messages,
#             stream=True
#         )
#         print(response)
#         for chunk in response:
#             print(chunk)
#             # print(chunk['choices'][0]['delta'])

#     except Exception as e:
#         pass
#         pytest.fail(f"Error occurred: {e}")

# # test_completion_expect_error()


# def test_ollama_llava():
#     litellm.set_verbose=True
#     # same params as gpt-4 vision
#     response = completion(
#         model = "ollama/llava",
#         messages=[
#             {
#                 "role": "user",
#                 "content": [
#                                 {
#                                     "type": "text",
#                                     "text": "What is in this picture"
#                                 },
#                                 {
#                                     "type": "image_url",
#                                     "image_url": {
#                                     "url": "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"
#                                     }
#                                 }
#                             ]
#             }
#         ],
#     )
#     print("Response from ollama/llava")
#     print(response)
# # test_ollama_llava()


# # PROCESSED CHUNK PRE CHUNK CREATOR
