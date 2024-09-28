# #### What this tests ####
# #    This tests calling batch_completions by running 100 messages together

# import sys, os, json
# import traceback
# import pytest

# sys.path.insert(
#     0, os.path.abspath("../..")
# )  # Adds the parent directory to the system path
# import litellm
# litellm.set_verbose = True
# from litellm import completion, BudgetManager

# budget_manager = BudgetManager(project_name="test_project", client_type="hosted")

# ## Scenario 1: User budget enough to make call
# def test_user_budget_enough():
#     try:
#         user = "1234"
#         # create a budget for a user
#         budget_manager.create_budget(total_budget=10, user=user, duration="daily")

#         # check if a given call can be made
#         data = {
#             "model": "gpt-3.5-turbo",
#             "messages": [{"role": "user", "content": "Hey, how's it going?"}]
#         }
#         if budget_manager.get_current_cost(user=user) <= budget_manager.get_total_budget(user):
#             response = completion(**data)
#             print(budget_manager.update_cost(completion_obj=response, user=user))
#         else:
#             response = "Sorry - no budget!"

#         print(f"response: {response}")
#     except Exception as e:
#         pytest.fail(f"An error occurred - {str(e)}")

# ## Scenario 2: User budget not enough to make call
# def test_user_budget_not_enough():
#     try:
#         user = "12345"
#         # create a budget for a user
#         budget_manager.create_budget(total_budget=0, user=user, duration="daily")

#         # check if a given call can be made
#         data = {
#             "model": "gpt-3.5-turbo",
#             "messages": [{"role": "user", "content": "Hey, how's it going?"}]
#         }
#         model = data["model"]
#         messages = data["messages"]
#         if budget_manager.get_current_cost(user=user) < budget_manager.get_total_budget(user=user):
#             response = completion(**data)
#             print(budget_manager.update_cost(completion_obj=response, user=user))
#         else:
#             response = "Sorry - no budget!"

#         print(f"response: {response}")
#     except:
#         pytest.fail(f"An error occurred")

# ## Scenario 3: Saving budget to client
# def test_save_user_budget():
#     try:
#         response = budget_manager.save_data()
#         if response["status"] == "error":
#             raise Exception(f"An error occurred - {json.dumps(response)}")
#         print(response)
#     except Exception as e:
#         pytest.fail(f"An error occurred: {str(e)}")

# test_save_user_budget()
# ## Scenario 4: Getting list of users
# def test_get_users():
#     try:
#         response = budget_manager.get_users()
#         print(response)
#     except:
#         pytest.fail(f"An error occurred")


# ## Scenario 5: Reset budget at the end of duration
# def test_reset_on_duration():
#     try:
#         # First, set a short duration budget for a user
#         user = "123456"
#         budget_manager.create_budget(total_budget=10, user=user, duration="daily")

#         # Use some of the budget
#         data = {
#             "model": "gpt-3.5-turbo",
#             "messages": [{"role": "user", "content": "Hello!"}]
#         }
#         if budget_manager.get_current_cost(user=user) <= budget_manager.get_total_budget(user=user):
#             response = litellm.completion(**data)
#             print(budget_manager.update_cost(completion_obj=response, user=user))

#         assert budget_manager.get_current_cost(user) > 0, f"Test setup failed: Budget did not decrease after completion"

#         # Now, we need to simulate the passing of time. Since we don't want our tests to actually take days, we're going
#         # to cheat a little -- we'll manually adjust the "created_at" time so it seems like a day has passed.
#         # In a real-world testing scenario, we might instead use something like the `freezegun` library to mock the system time.
#         one_day_in_seconds = 24 * 60 * 60
#         budget_manager.user_dict[user]["last_updated_at"] -= one_day_in_seconds

#         # Now the duration should have expired, so our budget should reset
#         budget_manager.update_budget_all_users()

#         # Make sure the budget was actually reset
#         assert budget_manager.get_current_cost(user) == 0, "Budget didn't reset after duration expired"
#     except Exception as e:
#         pytest.fail(f"An error occurred - {str(e)}")

# ## Scenario 6: passing in text:
# def test_input_text_on_completion():
#     try:
#         user = "12345"
#         budget_manager.create_budget(total_budget=10, user=user, duration="daily")

#         input_text = "hello world"
#         output_text = "it's a sunny day in san francisco"
#         model = "gpt-3.5-turbo"

#         budget_manager.update_cost(user=user, model=model, input_text=input_text, output_text=output_text)
#         print(budget_manager.get_current_cost(user))
#     except Exception as e:
#         pytest.fail(f"An error occurred - {str(e)}")

# test_input_text_on_completion()
