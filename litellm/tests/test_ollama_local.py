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

# def test_completion_ollama():
#     try:
#         response = completion(
#             model="ollama/llama2", 
#             messages=messages, 
#             api_base="http://localhost:11434"
#         )
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# # test_completion_ollama()

# def test_completion_ollama_stream():
#     user_message = "what is litellm?"
#     messages = [{ "content": user_message,"role": "user"}]
#     try:
#         response = completion(
#             model="ollama/llama2", 
#             messages=messages, 
#             api_base="http://localhost:11434", 
#             stream=True
#         )
#         print(response)
#         for chunk in response:
#             print(chunk)
#             # print(chunk['choices'][0]['delta'])

#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# # test_completion_ollama_stream()


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
#             print(chunk)

#             # print(chunk['choices'][0]['delta'])

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

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())
