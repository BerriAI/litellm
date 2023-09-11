import os, json
import litellm 
from litellm.utils import ModelResponse

class BudgetManager:
    def __init__(self, type: str):
        self.type = type
        ## load the data or init the initial dictionaries
        self.load_data() 
    
    def print_verbose(self, print_statement):
        if litellm.set_verbose:
            print(print_statement)
    
    def load_data(self):
        if self.type == "local":
            # Check if user dict file exists
            if os.path.isfile("user_cost.json"):
                # Load the user dict
                with open("user_cost.json", 'r') as json_file:
                    self.user_dict = json.load(json_file)
            else:
                self.print_verbose("User Dictionary not found!")
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
        model = completion_obj['model'] # if this throws an error try, model = completion_obj['model']
        self.user_dict[user]["current_cost"] = cost + self.user_dict[user].get("current_cost", 0)
        if "model_cost" in self.user_dict[user]:
            self.user_dict[user]["model_cost"][model] = cost + self.user_dict[user]["model_cost"].get(model, 0)
        else:
            self.user_dict[user]["model_cost"] = {model: cost}
        return {"user": self.user_dict[user]}
    
    def get_current_cost(self, user):
        return self.user_dict[user].get("current_cost", 0)
    
    def get_model_cost(self, user):
        return self.user_dict[user].get("model_cost", 0)
    
    def reset_cost(self, user):
        self.user_dict[user]["current_cost"] = 0
        self.user_dict[user]["model_cost"] = {}
        return {"user": self.user_dict[user]}

    def save_data(self):
        if self.type == "local":
            import json 
            
            # save the user dict 
            with open("user_cost.json", 'w') as json_file:
                json.dump(self.user_dict, json_file, indent=4)  # Indent for pretty formatting
