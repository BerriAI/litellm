#### What this tests ####
#    This tests calling batch_completions by running 100 messages together

import sys, os
import traceback

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm 
from litellm import budget_manager, completion 

## Scenario 1: User budget enough to make call
def test_user_budget_enough():
    user = "1234"
    # create a budget for a user
    budget_manager.create_budget(total_budget=10, user=user)

    # check if a given call can be made
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hey, how's it going?"}]
    }
    model = data["model"]
    messages = data["messages"]
    if budget_manager.get_current_cost(user=user) <= budget_manager.get_total_budget(user):
        response = completion(**data)
    else:
        response = "Sorry - no budget!"

    print(f"response: {response}")

## Scenario 2: User budget not enough to make call
def test_user_budget_not_enough():
    user = "12345"
    # create a budget for a user
    budget_manager.create_budget(total_budget=0, user=user)

    # check if a given call can be made
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hey, how's it going?"}]
    }
    model = data["model"]
    messages = data["messages"]
    if budget_manager.get_current_cost(user=user) < budget_manager.get_total_budget(user=user):
        response = completion(**data)
        budget_manager.update_cost(completion_obj=response, user=user)
    else:
        response = "Sorry - no budget!"

    print(f"response: {response}")

test_user_budget_not_enough()