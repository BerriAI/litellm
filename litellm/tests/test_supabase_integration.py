# #### What this tests ####
# #    This tests if logging to the supabase integration actually works
# # pytest mistakes intentional bad calls as failed tests -> [TODO] fix this
# import sys, os
# import traceback
# import pytest

# sys.path.insert(0, os.path.abspath('../..'))  # Adds the parent directory to the system path
# import litellm
# from litellm import embedding, completion

# litellm.input_callback = ["supabase"]
# litellm.success_callback = ["supabase"]
# litellm.failure_callback = ["supabase"]


# litellm.set_verbose = False

# user_message = "Hello, how are you?"
# messages = [{ "content": user_message,"role": "user"}]


# #openai call
# response = completion(
#     model="gpt-3.5-turbo",
#     messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}],
#     user="ishaan22"
# )

# import asyncio
# import time
# async def completion_call():
#     try:
#         response = await litellm.acompletion(
#             model="gpt-3.5-turbo", messages=messages, stream=True
#         )
#         complete_response = ""
#         start_time = time.time()
#         async for chunk in response:
#             chunk_time = time.time()
#             print(chunk)
#             complete_response += chunk["choices"][0]["delta"].get("content", "")
#             print(complete_response)
#             print(f"time since initial request: {chunk_time - start_time:.5f}")

#             if chunk["choices"][0].get("finish_reason", None) != None:
#                 print("🤗🤗🤗 DONE")
#     except:
#         print(f"error occurred: {traceback.format_exc()}")
#         pass


# asyncio.run(completion_call())

