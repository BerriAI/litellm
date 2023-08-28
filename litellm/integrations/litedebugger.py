import requests, traceback, json, os

class LiteDebugger:
    user_email = None
    dashboard_url = None

    def __init__(self, email=None):
        self.api_url = "https://api.litellm.ai/debugger"
        # self.api_url = "http://0.0.0.0:4000/debugger"
        self.validate_environment(email)
        pass

    def validate_environment(self, email):
        try:
            self.user_email = os.getenv("LITELLM_EMAIL") or email
            self.dashboard_url = "https://admin.litellm.ai/" + self.user_email
            try:
                print(f"\033[92mHere's your LiteLLM Dashboard 👉 \033[94m\033[4m{self.dashboard_url}\033[0m")
            except:
                print(f"Here's your LiteLLM Dashboard 👉 {self.dashboard_url}")
            if self.user_email == None:
                raise Exception(
                    "[Non-Blocking Error] LiteLLMDebugger: Missing LITELLM_EMAIL. Set it in your environment. Eg.: os.environ['LITELLM_EMAIL']= <your_email>"
                )
        except Exception as e:
            raise ValueError(
                "[Non-Blocking Error] LiteLLMDebugger: Missing LITELLM_EMAIL. Set it in your environment. Eg.: os.environ['LITELLM_EMAIL']= <your_email>"
            )

    def input_log_event(
        self, model, messages, end_user, litellm_call_id, print_verbose, litellm_params, optional_params
    ):
        try:
            print_verbose(
                f"LiteLLMDebugger: Logging - Enters input logging function for model {model}"
            )
            def remove_key_value(dictionary, key):
                new_dict = dictionary.copy()  # Create a copy of the original dictionary
                new_dict.pop(key)  # Remove the specified key-value pair from the copy
                return new_dict
            
            updated_litellm_params = remove_key_value(litellm_params, "logger_fn")

            litellm_data_obj = {
                "model": model,
                "messages": messages,
                "end_user": end_user,
                "status": "initiated",
                "litellm_call_id": litellm_call_id,
                "user_email": self.user_email,
                "litellm_params": updated_litellm_params,
                "optional_params": optional_params
            }
            print_verbose(
                f"LiteLLMDebugger: Logging - logged data obj {litellm_data_obj}"
            )
            response = requests.post(
                url=self.api_url,
                headers={"content-type": "application/json"},
                data=json.dumps(litellm_data_obj),
            )
            print_verbose(f"LiteDebugger: api response - {response.text}")
        except:
            print_verbose(
                f"[Non-Blocking Error] LiteDebugger: Logging Error - {traceback.format_exc()}"
            )
            pass
    
    def post_call_log_event(
        self, original_response, litellm_call_id, print_verbose
    ):
        try:
            litellm_data_obj = {
                "status": "received",
                "additional_details": {"original_response": original_response},
                "litellm_call_id": litellm_call_id,
                "user_email": self.user_email,
            }
            response = requests.post(
                url=self.api_url,
                headers={"content-type": "application/json"},
                data=json.dumps(litellm_data_obj),
            )
            print_verbose(f"LiteDebugger: api response - {response.text}")
        except:
            print_verbose(
                f"[Non-Blocking Error] LiteDebugger: Logging Error - {traceback.format_exc()}"
            )

    def log_event(
        self,
        model,
        messages,
        end_user,
        response_obj,
        start_time,
        end_time,
        litellm_call_id,
        print_verbose,
    ):
        try:
            print_verbose(
                f"LiteLLMDebugger: Logging - Enters handler logging function for model {model} with response object {response_obj}"
            )
            total_cost = 0  # [TODO] implement cost tracking
            response_time = (end_time - start_time).total_seconds()
            if "choices" in response_obj:
                litellm_data_obj = {
                    "response_time": response_time,
                    "model": response_obj["model"],
                    "total_cost": total_cost,
                    "messages": messages,
                    "response": response['choices'][0]['message']['content'],
                    "end_user": end_user,
                    "litellm_call_id": litellm_call_id,
                    "status": "success",
                    "user_email": self.user_email,
                }
                print_verbose(
                    f"LiteDebugger: Logging - final data object: {litellm_data_obj}"
                )
                response = requests.post(
                    url=self.api_url,
                    headers={"content-type": "application/json"},
                    data=json.dumps(litellm_data_obj),
                )
            elif "data" in response_obj and isinstance(response_obj["data"], list) and len(response_obj["data"]) > 0 and "embedding" in response_obj["data"][0]:
                print(f"messages: {messages}")
                litellm_data_obj = {
                    "response_time": response_time,
                    "model": response_obj["model"],
                    "total_cost": total_cost,
                    "messages": messages,
                    "response": str(response_obj["data"][0]["embedding"][:5]),
                    "end_user": end_user,
                    "litellm_call_id": litellm_call_id,
                    "status": "success",
                    "user_email": self.user_email,
                }
                print_verbose(
                    f"LiteDebugger: Logging - final data object: {litellm_data_obj}"
                )
                response = requests.post(
                    url=self.api_url,
                    headers={"content-type": "application/json"},
                    data=json.dumps(litellm_data_obj),
                )
            elif isinstance(response_obj, object) and response_obj.__class__.__name__ == "CustomStreamWrapper":
                litellm_data_obj = {
                    "response_time": response_time,
                    "total_cost": total_cost,
                    "messages": messages,
                    "response": "Streamed response",
                    "end_user": end_user,
                    "litellm_call_id": litellm_call_id,
                    "status": "success",
                    "user_email": self.user_email,
                }
                print_verbose(
                    f"LiteDebugger: Logging - final data object: {litellm_data_obj}"
                )
                response = requests.post(
                    url=self.api_url,
                    headers={"content-type": "application/json"},
                    data=json.dumps(litellm_data_obj),
                )
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
                    "status": "failure",
                    "user_email": self.user_email,
                }
                print_verbose(
                    f"LiteDebugger: Logging - final data object: {litellm_data_obj}"
                )
                response = requests.post(
                    url=self.api_url,
                    headers={"content-type": "application/json"},
                    data=json.dumps(litellm_data_obj),
                )
                print_verbose(f"LiteDebugger: api response - {response.text}")
        except:
            print_verbose(
                f"[Non-Blocking Error] LiteDebugger: Logging Error - {traceback.format_exc()}"
            )
            pass
