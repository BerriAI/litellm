# #### What this tests ####
# #    This tests if logging to the helicone integration actually works

# import sys, os
# import traceback
# import pytest

# sys.path.insert(
#     0, os.path.abspath("../..")
# )  # Adds the parent directory to the system path
# import litellm
# from litellm import embedding, completion

# litellm.success_callback = ["helicone"]

# litellm.set_verbose = True

# user_message = "Hello, how are you?"
# messages = [{"content": user_message, "role": "user"}]


# # openai call
# response = completion(
#     model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}]
# )

# # cohere call
# response = completion(
#     model="command-nightly",
#     messages=[{"role": "user", "content": "Hi 👋 - i'm cohere"}],
# )
