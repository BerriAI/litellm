# 💸 User-based Rate Limiting

LiteLLM exposes the `BudgetManager` class to help with user-based rate limiting. 

## quick start

```python
from litellm import BudgetManager, completion 

budget_manager = BudgetManager(project_name="test_project")

user = "1234"

# create a budget if new user user
if not budget_manager.is_valid_user(user):
    budget_manager.create_budget(total_budget=10, user=user)

# check if a given call can be made
if budget_manager.get_current_cost(user=user) <= budget_manager.get_total_budget(user):
    response = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hey, how's it going?"}])
    budget_manager.update_cost(completion_obj=response, user=user)
else:
    response = "Sorry - no budget!"
```

[**Implementation Code**](https://github.com/BerriAI/litellm/blob/main/litellm/budget_manager.py)

## advanced usage

BudgetManager creates a dictionary to manage the user budgets, where the key is user and the object is their current cost + model-specific costs. By default this is saved to a local, but you can change it to be stored to a hosted client (either self-hosted or LiteLLM one).

### get model-breakdown per user 

```
user = "1234"
# ...
budget_manager.get_model_cost(user=user) # {"gpt-3.5-turbo-0613": 7.3e-05}
```

[**Implementation Code**](https://github.com/BerriAI/litellm/blob/817798c692207569a17c26186d10541aa83f04e7/litellm/budget_manager.py#L71)

### save budget to disk

When you call `save_data()` it will check for the self.client_type (by default this is set to local), and save the dictionary to a local `user_cost.json` file. 

```python
# ...
budget_manager.save_data() # 👈 save to user_cost.json()
```

[**Implementation Code**](https://github.com/BerriAI/litellm/blob/817798c692207569a17c26186d10541aa83f04e7/litellm/budget_manager.py#L83)

### save budget to hosted db (LiteLLM)

Set the BudgetManager type to `client_type`.
```python
budget_manager = BudgetManager(project_name="test_project", client_type="hosted")
# ...
budget_manager.save_data() # 👈 saved to hosted db 
```

[**Implementation Code**](https://github.com/BerriAI/litellm/blob/817798c692207569a17c26186d10541aa83f04e7/litellm/budget_manager.py#L11)

### save budget to hosted db (self-hosted)

Set the BudgetManager type to `client_type`. Overwrite the api_base
```python
budget_manager = BudgetManager(project_name="test_project", type="client", api_base="your_custom_api")
# ...
budget_manager.save_data() # 👈 saved to self-hosted db 
```

[**Implementation Code**](https://github.com/BerriAI/litellm/blob/817798c692207569a17c26186d10541aa83f04e7/litellm/budget_manager.py#L11)