# #### What this tests ####
# #    This tests if logging to the llmonitor integration actually works
# # Adds the parent directory to the system path
# import sys
# import os

# sys.path.insert(0, os.path.abspath('../..'))

# from litellm import completion, embedding
# import litellm

# litellm.success_callback = ["promptlayer"]


# litellm.set_verbose = True


# def test_chat_openai():
#     try:
#         response = completion(model="gpt-3.5-turbo",
#                               messages=[{
#                                   "role": "user",
#                                   "content": "Hi 👋 - i'm openai"
#                               }])

#         print(response)

#     except Exception as e:
#         print(e)

# # test_chat_openai()
