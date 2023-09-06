
# import sys, os
# import traceback
# from dotenv import load_dotenv

# load_dotenv()
# import os

# sys.path.insert(
#     0, os.path.abspath("../..")
# )  # Adds the parent directory to the system path
# import pytest
# import litellm
# from litellm import embedding, completion, text_completion
# from litellm.utils import completion_cost


# print(completion_cost(
#         model="togethercomputer/llama-2-2b-chat", 
#         prompt="gm", 
#         completion="hello"
#     ))