###### THESE TESTS CAN ONLY RUN LOCALLY WITH THE OLLAMA SERVER RUNNING ######

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

# async def get_response(generator):
#     response = ""
#     async for elem in generator:
#         print(elem)
#         response += elem["content"]
#     return response

# def test_completion_ollama():
#     try:
#         response = completion(model="llama2", messages=messages, api_base="http://localhost:11434", custom_llm_provider="ollama")
#         print(response)
#         string_response = asyncio.run(get_response(response))
#         print(string_response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")


# # test_completion_ollama()

# def test_completion_ollama_stream():
#     try:
#         response = completion(model="llama2", messages=messages, api_base="http://localhost:11434", custom_llm_provider="ollama", stream=True)
#         print(response)
#         string_response = asyncio.run(get_response(response))
#         print(string_response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# test_completion_ollama_stream()
