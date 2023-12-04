# #### What this tests ####
# #    This tests the cost tracking function works with consecutive calls (~10 consecutive calls)

# import sys, os
# import traceback
# import pytest
# sys.path.insert(
#     0, os.path.abspath("../..")
# )  # Adds the parent directory to the system path
# import litellm

# async def test_proxy_cost_tracking(): 
#     """
#     Get expected cost. 
#     Create new key.
#     Run 10 parallel calls. 
#     Check cost for key at the end. 
#     assert it's = expected cost. 
#     """
#     model = "gpt-3.5-turbo"
#     messages = [{"role": "user", "content": "Hey, how's it going?"}]
#     number_of_calls = 10
#     expected_cost = litellm.completion_cost(model=model, messages=messages) * number_of_calls
#     async def litellm_acompletion(): 



