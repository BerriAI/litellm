#### What this tests ####
#    This tests calling litellm.max_budget by making back-to-back gpt-4 calls
# commenting out this test for circle ci, as it causes other tests to fail, since litellm.max_budget would impact other litellm imports
# import sys, os, json
# import traceback
# import pytest 

# sys.path.insert(
#     0, os.path.abspath("../..")
# )  # Adds the parent directory to the system path
# import litellm 
# litellm.set_verbose = True
# from litellm import completion

# litellm.max_budget = 0.001 # sets a max budget of $0.001

# messages = [{"role": "user", "content": "Hey, how's it going"}]
# completion(model="gpt-4", messages=messages)
# completion(model="gpt-4", messages=messages)
# print(litellm._current_cost)

