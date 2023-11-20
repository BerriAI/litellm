# #### What this tests ####
# #    This tests if logging to the llmonitor integration actually works
# # Adds the parent directory to the system path
# import sys
# import os

# sys.path.insert(0, os.path.abspath("../.."))

# from litellm import completion, embedding
# import litellm

# litellm.success_callback = ["llmonitor"]
# litellm.failure_callback = ["llmonitor"]

# litellm.set_verbose = True


# def test_chat_openai():
#     try:
#         response = completion(
#             model="gpt-3.5-turbo",
#             messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}],
#             user="ishaan_from_litellm"
#         )

#         print(response)

#     except Exception as e:
#         print(e)


# def test_embedding_openai():
#     try:
#         response = embedding(model="text-embedding-ada-002", input=["test"])
#         # Add any assertions here to check the response
#         print(f"response: {str(response)[:50]}")
#     except Exception as e:
#         print(e)


# test_chat_openai()
# # test_embedding_openai()


# def test_llmonitor_logging_function_calling():
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
#         response = completion(model="gpt-3.5-turbo",
#                               messages=[{
#                                   "role": "user",
#                                   "content": "what's the weather in boston"
#                               }],
#                               temperature=0.1,
#                               functions=function1,
#             )
#         print(response)
#     except Exception as e:
#         print(e)

# # test_llmonitor_logging_function_calling()
