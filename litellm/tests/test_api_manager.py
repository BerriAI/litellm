#### What this tests ####
#    This tests calling batch_completions by running 100 messages together

import sys, os
import traceback

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm 
from litellm import apiManager, completion 

litellm.success_callback = ["api_manager"]


## Scenario 1: User budget enough to make call
def test_user_budget_enough():
    user = "1234"
    # create a budget for a user
    apiManager.create_budget(total_budget=10, user=user)

    # check if a given call can be made
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hey, how's it going?"}]
    }
    model = data["model"]
    messages = data["messages"]
    if apiManager.projected_cost(**data, user=user) <= apiManager.get_total_budget(user):
        response = completion(**data)
    else:
        response = "Sorry - no budget!"

    print(f"response: {response}")

## Scenario 2: User budget not enough to make call
def test_user_budget_not_enough():
    user = "12345"
    # create a budget for a user
    apiManager.create_budget(total_budget=0, user=user)

    # check if a given call can be made
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hey, how's it going?"}]
    }
    model = data["model"]
    messages = data["messages"]
    projectedCost = apiManager.projected_cost(**data, user=user)
    print(f"projectedCost: {projectedCost}")
    totalBudget = apiManager.get_total_budget(user)
    print(f"totalBudget: {totalBudget}")
    if projectedCost <= totalBudget:
        response = completion(**data)
    else:
        response = "Sorry - no budget!"

    print(f"response: {response}")

test_user_budget_not_enough()