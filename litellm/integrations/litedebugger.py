import requests, traceback, json
class LiteDebugger:
    def __init__(self):
        self.api_url = "https://api.litellm.ai/debugger"
        pass
    
    def input_log_event(self, model, messages, end_user, litellm_call_id, print_verbose):
        try:
            print_verbose(
                f"LiteLLMDebugger: Logging - Enters input logging function for model {model}"
            )
            litellm_data_obj = {
                "model": model, 
                "messages": messages, 
                "end_user": end_user, 
                "status": "initiated",
                "litellm_call_id": litellm_call_id
            }
            response = requests.post(url=self.api_url, headers={"content-type": "application/json"}, data=json.dumps(litellm_data_obj))
            print_verbose(f"LiteDebugger: api response - {response.text}")
        except:
            print_verbose(f"LiteDebugger: Logging Error - {traceback.format_exc()}")
            pass
    
    def log_event(self, model,
        messages,
        end_user,
        response_obj,
        start_time,
        end_time,
        litellm_call_id,
        print_verbose,):
        try:
            print_verbose(
                f"LiteLLMDebugger: Logging - Enters input logging function for model {model}"
            )
            total_cost = 0 # [TODO] implement cost tracking
            response_time = (end_time - start_time).total_seconds()
            if "choices" in response_obj:
                litellm_data_obj = {
                    "response_time": response_time,
                    "model": response_obj["model"],
                    "total_cost": total_cost,
                    "messages": messages,
                    "response": response_obj["choices"][0]["message"]["content"],
                    "end_user": end_user,
                    "litellm_call_id": litellm_call_id,
                    "status": "success"
                }
                print_verbose(
                    f"LiteDebugger: Logging - final data object: {litellm_data_obj}"
                )
                response = requests.post(url=self.api_url, headers={"content-type": "application/json"}, data=json.dumps(litellm_data_obj))
            elif "error" in response_obj:
                if "Unable to map your input to a model." in response_obj["error"]:
                    total_cost = 0
                litellm_data_obj = {
                    "response_time": response_time,
                    "model": response_obj["model"],
                    "total_cost": total_cost,
                    "messages": messages,
                    "error": response_obj["error"],
                    "end_user": end_user,
                    "litellm_call_id": litellm_call_id,
                    "status": "failure"
                }
                print_verbose(
                    f"LiteDebugger: Logging - final data object: {litellm_data_obj}"
                )
                response = requests.post(url=self.api_url, headers={"content-type": "application/json"}, data=json.dumps(litellm_data_obj))
                print_verbose(f"LiteDebugger: api response - {response.text}")
        except:
            print_verbose(f"LiteDebugger: Logging Error - {traceback.format_exc()}")
            pass