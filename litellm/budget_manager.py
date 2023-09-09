import litellm 
from litellm.utils import ModelResponse
class BudgetManager:
    def __init__(self):
        self.user_dict = {}

    def create_budget(self, total_budget: float, user: str):
        self.user_dict[user] = {"total_budget": total_budget}
        return self.user_dict[user]
    
    def projected_cost(self, model: str, messages: list, user: str):
        text = "".join(message["content"] for message in messages)
        prompt_tokens = litellm.token_counter(model=model, text=text)
        prompt_cost, _ = litellm.cost_per_token(model=model, prompt_tokens=prompt_tokens, completion_tokens=0)
        current_cost = self.user_dict[user].get("current_cost", 0)
        projected_cost = prompt_cost + current_cost
        return projected_cost
    
    def get_total_budget(self, user: str):
        return self.user_dict[user]["total_budget"]

    def update_cost(self, completion_obj: ModelResponse, user: str):
        cost = litellm.completion_cost(completion_response=completion_obj)
        self.user_dict[user]["current_cost"] = cost + self.user_dict[user].get("current_cost", 0)
        return self.user_dict[user]["current_cost"]
    
    def get_current_cost(self, user):
        return self.user_dict[user].get("current_cost", 0)