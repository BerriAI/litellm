import requests, traceback, json, os
import types

class LiteDebugger:
    user_email = None
    dashboard_url = None

    def __init__(self, email=None):
        self.api_url = "https://api.litellm.ai/debugger"
        self.validate_environment(email)
        pass

    def validate_environment(self, email):
        try:
            self.user_email = (email or os.getenv("LITELLM_TOKEN") or os.getenv("LITELLM_EMAIL"))
            if self.user_email == None: # if users are trying to use_client=True but token not set
                raise ValueError("litellm.use_client = True but no token or email passed. Please set it in litellm.token")
            self.dashboard_url = "https://admin.litellm.ai/" + self.user_email
            try:
                print(
                    f"\033[92mHere's your LiteLLM Dashboard 👉 \033[94m\033[4m{self.dashboard_url}\033[0m"
                )
            except:
                print(f"Here's your LiteLLM Dashboard 👉 {self.dashboard_url}")
            if self.user_email == None:
                raise ValueError(
                    "[Non-Blocking Error] LiteLLMDebugger: Missing LITELLM_TOKEN. Set it in your environment. Eg.: os.environ['LITELLM_TOKEN']= <your_email>"
                )
        except Exception as e:
            raise ValueError(
                "[Non-Blocking Error] LiteLLMDebugger: Missing LITELLM_TOKEN. Set it in your environment. Eg.: os.environ['LITELLM_TOKEN']= <your_email>"
            )

    def input_log_event(
        self,
        model,
        messages,
        end_user,
        litellm_call_id,
        call_type,
        print_verbose,
        litellm_params,
        optional_params,
    ):
        print_verbose(f"LiteDebugger: Pre-API Call Logging for call id {litellm_call_id}")
        try:
            print_verbose(
                f"LiteLLMDebugger: Logging - Enters input logging function for model {model}"
            )

            def remove_key_value(dictionary, key):
                new_dict = dictionary.copy()  # Create a copy of the original dictionary
                new_dict.pop(key)  # Remove the specified key-value pair from the copy
                return new_dict

            updated_litellm_params = remove_key_value(litellm_params, "logger_fn")

            if call_type == "embedding":
                for message in messages: # assuming the input is a list as required by the embedding function
                    litellm_data_obj = {
                        "model": model,
                        "messages": [{"role": "user", "content": message}],
                        "end_user": end_user,
                        "status": "initiated",
                        "litellm_call_id": litellm_call_id,
                        "user_email": self.user_email,
                        "litellm_params": updated_litellm_params,
                        "optional_params": optional_params,
                    }
                    print_verbose(
                        f"LiteLLMDebugger: Logging - logged data obj {litellm_data_obj}"
                    )
                    response = requests.post(
                        url=self.api_url,
                        headers={"content-type": "application/json"},
                        data=json.dumps(litellm_data_obj),
                    )
                print_verbose(f"LiteDebugger: embedding api response - {response.text}")
            elif call_type == "completion":
                litellm_data_obj = {
                    "model": model,
                    "messages": messages if isinstance(messages, list) else [{"role": "user", "content": messages}],
                    "end_user": end_user,
                    "status": "initiated",
                    "litellm_call_id": litellm_call_id,
                    "user_email": self.user_email,
                    "litellm_params": updated_litellm_params,
                    "optional_params": optional_params,
                }
                print_verbose(
                    f"LiteLLMDebugger: Logging - logged data obj {litellm_data_obj}"
                )
                response = requests.post(
                    url=self.api_url,
                    headers={"content-type": "application/json"},
                    data=json.dumps(litellm_data_obj),
                )
                print_verbose(f"LiteDebugger: completion api response - {response.text}")
        except:
            print_verbose(
                f"[Non-Blocking Error] LiteDebugger: Logging Error - {traceback.format_exc()}"
            )
            pass

    def post_call_log_event(self, original_response, litellm_call_id, print_verbose, call_type, stream):
        print_verbose(f"LiteDebugger: Post-API Call Logging for call id {litellm_call_id}")
        try:
            if call_type == "embedding":
                litellm_data_obj = {
                    "status": "received",
                    "additional_details": {"original_response": str(original_response["data"][0]["embedding"][:5])}, # don't store the entire vector
                    "litellm_call_id": litellm_call_id,
                    "user_email": self.user_email,
                }
            elif call_type == "completion" and not stream:
                litellm_data_obj = {
                    "status": "received",
                    "additional_details": {"original_response": original_response},
                    "litellm_call_id": litellm_call_id,
                    "user_email": self.user_email,
                }
            elif call_type == "completion" and stream:
                litellm_data_obj = {
                    "status": "received",
                    "additional_details": {"original_response": "Streamed response" if isinstance(original_response, types.GeneratorType) else original_response},
                    "litellm_call_id": litellm_call_id,
                    "user_email": self.user_email,
                }
            print_verbose(f"litedebugger post-call data object - {litellm_data_obj}")
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
        end_user,
        response_obj,
        start_time,
        end_time,
        litellm_call_id,
        print_verbose,
        call_type, 
        stream = False
    ):
        print_verbose(f"LiteDebugger: Success/Failure Call Logging for call id {litellm_call_id}")
        try:
            print_verbose(
                f"LiteLLMDebugger: Success/Failure Logging - Enters handler logging function for function {call_type} and stream set to {stream} with response object {response_obj}"
            )
            total_cost = 0  # [TODO] implement cost tracking
            response_time = (end_time - start_time).total_seconds()
            if call_type == "completion" and stream == False:
                litellm_data_obj = {
                    "response_time": response_time,
                    "total_cost": total_cost,
                    "response": response_obj["choices"][0]["message"]["content"],
                    "litellm_call_id": litellm_call_id,
                    "status": "success",
                }
                print_verbose(
                    f"LiteDebugger: Logging - final data object: {litellm_data_obj}"
                )
                response = requests.post(
                    url=self.api_url,
                    headers={"content-type": "application/json"},
                    data=json.dumps(litellm_data_obj),
                )
            elif call_type == "embedding":
                litellm_data_obj = {
                    "response_time": response_time,
                    "total_cost": total_cost,
                    "response": str(response_obj["data"][0]["embedding"][:5]),
                    "litellm_call_id": litellm_call_id,
                    "status": "success",
                }
                response = requests.post(
                    url=self.api_url,
                    headers={"content-type": "application/json"},
                    data=json.dumps(litellm_data_obj),
                )
            elif call_type == "completion" and stream == True:
                if len(response_obj["content"]) > 0: # don't log the empty strings
                    litellm_data_obj = {
                        "response_time": response_time,
                        "total_cost": total_cost,
                        "response": response_obj["content"],
                        "litellm_call_id": litellm_call_id,
                        "status": "success",
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
