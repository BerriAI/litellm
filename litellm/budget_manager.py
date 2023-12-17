import os, json, time
import litellm 
from litellm.utils import ModelResponse
import requests, threading
from typing import Optional, Union, Literal

class BudgetManager:
    def __init__(self, project_name: str, client_type: str = "local", api_base: Optional[str] = None):
        self.client_type = client_type
        self.project_name = project_name
        self.api_base = api_base or "https://api.litellm.ai"
        ## load the data or init the initial dictionaries
        self.load_data() 
    
    def print_verbose(self, print_statement):
        if litellm.set_verbose:
            import logging
            logging.info(print_statement)
    
    def load_data(self):
        if self.client_type == "local":
            # Check if user dict file exists
            if os.path.isfile("user_cost.json"):
                # Load the user dict
                with open("user_cost.json", 'r') as json_file:
                    self.user_dict = json.load(json_file)
            else:
                self.print_verbose("User Dictionary not found!")
                self.user_dict = {} 
            self.print_verbose(f"user dict from local: {self.user_dict}")
        elif self.client_type == "hosted":
            # Load the user_dict from hosted db
            url = self.api_base + "/get_budget"
            headers = {'Content-Type': 'application/json'}
            data = {
                'project_name' : self.project_name
            }
            response = requests.post(url, headers=headers, json=data)
            response = response.json()
            if response["status"] == "error":
                self.user_dict = {} # assume this means the user dict hasn't been stored yet
            else:
                self.user_dict = response["data"]

    def create_budget(self, total_budget: float, user: str, duration: Optional[Literal["daily", "weekly", "monthly", "yearly"]] = None, created_at: float = time.time()): 
        self.user_dict[user] = {"total_budget": total_budget}
        if duration is None:
            return self.user_dict[user]
        
        if duration == 'daily':
            duration_in_days = 1
        elif duration == 'weekly':
            duration_in_days = 7
        elif duration == 'monthly':
            duration_in_days = 28
        elif duration == 'yearly':
            duration_in_days = 365
        else:
            raise ValueError("""duration needs to be one of ["daily", "weekly", "monthly", "yearly"]""")
        self.user_dict[user] = {"total_budget": total_budget, "duration": duration_in_days, "created_at": created_at, "last_updated_at": created_at}
        self._save_data_thread() # [Non-Blocking] Update persistent storage without blocking execution
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
    
    def update_cost(self, user: str, completion_obj: Optional[ModelResponse] = None, model: Optional[str] = None, input_text: Optional[str] = None, output_text: Optional[str] = None):
        if model and input_text and output_text:
            prompt_tokens = litellm.token_counter(model=model, messages=[{"role": "user", "content": input_text}])
            completion_tokens = litellm.token_counter(model=model, messages=[{"role": "user", "content": output_text}])
            prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar = litellm.cost_per_token(model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
            cost = prompt_tokens_cost_usd_dollar + completion_tokens_cost_usd_dollar
        elif completion_obj: 
            cost = litellm.completion_cost(completion_response=completion_obj)
            model = completion_obj['model'] # if this throws an error try, model = completion_obj['model']
        else:
            raise ValueError("Either a chat completion object or the text response needs to be passed in. Learn more - https://docs.litellm.ai/docs/budget_manager")

        self.user_dict[user]["current_cost"] = cost + self.user_dict[user].get("current_cost", 0)
        if "model_cost" in self.user_dict[user]:
            self.user_dict[user]["model_cost"][model] = cost + self.user_dict[user]["model_cost"].get(model, 0)
        else:
            self.user_dict[user]["model_cost"] = {model: cost}

        self._save_data_thread() # [Non-Blocking] Update persistent storage without blocking execution
        return {"user": self.user_dict[user]}
    
    
    def get_current_cost(self, user):
        return self.user_dict[user].get("current_cost", 0)
    
    def get_model_cost(self, user):
        return self.user_dict[user].get("model_cost", 0)
    
    def is_valid_user(self, user: str) -> bool:
        return user in self.user_dict
    
    def get_users(self):
        return list(self.user_dict.keys())
    
    def reset_cost(self, user):
        self.user_dict[user]["current_cost"] = 0
        self.user_dict[user]["model_cost"] = {}
        return {"user": self.user_dict[user]}
    
    def reset_on_duration(self, user: str):
        # Get current and creation time
        last_updated_at = self.user_dict[user]["last_updated_at"]
        current_time = time.time()

        # Convert duration from days to seconds
        duration_in_seconds = self.user_dict[user]["duration"] * 24 * 60 * 60
        
        # Check if duration has elapsed
        if current_time - last_updated_at >= duration_in_seconds:
            # Reset cost if duration has elapsed and update the creation time
            self.reset_cost(user)
            self.user_dict[user]["last_updated_at"] = current_time
            self._save_data_thread()  # Save the data
    
    def update_budget_all_users(self):
        for user in self.get_users():
            if "duration" in self.user_dict[user]:
                self.reset_on_duration(user)

    def _save_data_thread(self):
        thread = threading.Thread(target=self.save_data) # [Non-Blocking]: saves data without blocking execution
        thread.start()

    def save_data(self):
        if self.client_type == "local":
            import json 
            
            # save the user dict 
            with open("user_cost.json", 'w') as json_file:
                json.dump(self.user_dict, json_file, indent=4)  # Indent for pretty formatting
            return {"status": "success"}
        elif self.client_type == "hosted":
            url = self.api_base + "/set_budget"
            headers = {'Content-Type': 'application/json'}
            data = {
                'project_name' : self.project_name, 
                "user_dict": self.user_dict
            }
            response = requests.post(url, headers=headers, json=data)
            response = response.json()
            return response