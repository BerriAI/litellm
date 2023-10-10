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

