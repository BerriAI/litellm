#### What this does ####
#    On success + failure, log events to Supabase

import dotenv, os
import requests

dotenv.load_dotenv()  # Loading env variables using dotenv
import traceback
import datetime, subprocess, sys

model_cost = {
    "gpt-3.5-turbo": {
        "max_tokens": 4000,
        "input_cost_per_token": 0.0000015,
        "output_cost_per_token": 0.000002,
    },
    "gpt-35-turbo": {
        "max_tokens": 4000,
        "input_cost_per_token": 0.0000015,
        "output_cost_per_token": 0.000002,
    },  # azure model name
    "gpt-3.5-turbo-0613": {
        "max_tokens": 4000,
        "input_cost_per_token": 0.0000015,
        "output_cost_per_token": 0.000002,
    },
    "gpt-3.5-turbo-0301": {
        "max_tokens": 4000,
        "input_cost_per_token": 0.0000015,
        "output_cost_per_token": 0.000002,
    },
    "gpt-3.5-turbo-16k": {
        "max_tokens": 16000,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000004,
    },
    "gpt-35-turbo-16k": {
        "max_tokens": 16000,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000004,
    },  # azure model name
    "gpt-3.5-turbo-16k-0613": {
        "max_tokens": 16000,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000004,
    },
    "gpt-4": {
        "max_tokens": 8000,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.00006,
    },
    "gpt-4-0613": {
        "max_tokens": 8000,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.00006,
    },
    "gpt-4-32k": {
        "max_tokens": 8000,
        "input_cost_per_token": 0.00006,
        "output_cost_per_token": 0.00012,
    },
    "claude-instant-1": {
        "max_tokens": 100000,
        "input_cost_per_token": 0.00000163,
        "output_cost_per_token": 0.00000551,
    },
    "claude-2": {
        "max_tokens": 100000,
        "input_cost_per_token": 0.00001102,
        "output_cost_per_token": 0.00003268,
    },
    "text-bison-001": {
        "max_tokens": 8192,
        "input_cost_per_token": 0.000004,
        "output_cost_per_token": 0.000004,
    },
    "chat-bison-001": {
        "max_tokens": 4096,
        "input_cost_per_token": 0.000002,
        "output_cost_per_token": 0.000002,
    },
    "command-nightly": {
        "max_tokens": 4096,
        "input_cost_per_token": 0.000015,
        "output_cost_per_token": 0.000015,
    },
}


class Supabase:
    # Class variables or attributes
    supabase_table_name = "request_logs"

    def __init__(self):
        # Instance variables
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        try:
            import supabase
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "supabase"])
            import supabase
        self.supabase_client = supabase.create_client(
            self.supabase_url, self.supabase_key
        )

    def price_calculator(self, model, response_obj, start_time, end_time):
        # try and find if the model is in the model_cost map
        # else default to the average of the costs
        prompt_tokens_cost_usd_dollar = 0
        completion_tokens_cost_usd_dollar = 0
        if model in model_cost:
            prompt_tokens_cost_usd_dollar = (
                model_cost[model]["input_cost_per_token"]
                * response_obj["usage"]["prompt_tokens"]
            )
            completion_tokens_cost_usd_dollar = (
                model_cost[model]["output_cost_per_token"]
                * response_obj["usage"]["completion_tokens"]
            )
        elif "replicate" in model:
            # replicate models are charged based on time
            # llama 2 runs on an nvidia a100 which costs $0.0032 per second - https://replicate.com/replicate/llama-2-70b-chat
            model_run_time = end_time - start_time  # assuming time in seconds
            cost_usd_dollar = model_run_time * 0.0032
            prompt_tokens_cost_usd_dollar = cost_usd_dollar / 2
            completion_tokens_cost_usd_dollar = cost_usd_dollar / 2
        else:
            # calculate average input cost
            input_cost_sum = 0
            output_cost_sum = 0
            for model in model_cost:
                input_cost_sum += model_cost[model]["input_cost_per_token"]
                output_cost_sum += model_cost[model]["output_cost_per_token"]
            avg_input_cost = input_cost_sum / len(model_cost.keys())
            avg_output_cost = output_cost_sum / len(model_cost.keys())
            prompt_tokens_cost_usd_dollar = (
                model_cost[model]["input_cost_per_token"]
                * response_obj["usage"]["prompt_tokens"]
            )
            completion_tokens_cost_usd_dollar = (
                model_cost[model]["output_cost_per_token"]
                * response_obj["usage"]["completion_tokens"]
            )
        return prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar

    def input_log_event(
        self, model, messages, end_user, litellm_call_id, print_verbose
    ):
        try:
            print_verbose(
                f"Supabase Logging - Enters input logging function for model {model}"
            )
            supabase_data_obj = {
                "model": model,
                "messages": messages,
                "end_user": end_user,
                "status": "initiated",
                "litellm_call_id": litellm_call_id,
            }
            data, count = (
                self.supabase_client.table(self.supabase_table_name)
                .insert(supabase_data_obj)
                .execute()
            )
            print(f"data: {data}")
        except:
            print_verbose(f"Supabase Logging Error - {traceback.format_exc()}")
            pass

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
                f"Supabase Logging - Enters logging function for model {model}, response_obj: {response_obj}"
            )

            (
                prompt_tokens_cost_usd_dollar,
                completion_tokens_cost_usd_dollar,
            ) = self.price_calculator(model, response_obj, start_time, end_time)
            total_cost = (
                prompt_tokens_cost_usd_dollar + completion_tokens_cost_usd_dollar
            )

            response_time = (end_time - start_time).total_seconds()
            if "choices" in response_obj:
                supabase_data_obj = {
                    "response_time": response_time,
                    "model": response_obj["model"],
                    "total_cost": total_cost,
                    "messages": messages,
                    "response": response_obj["choices"][0]["message"]["content"],
                    "end_user": end_user,
                    "litellm_call_id": litellm_call_id,
                    "status": "success",
                }
                print_verbose(
                    f"Supabase Logging - final data object: {supabase_data_obj}"
                )
                data, count = (
                    self.supabase_client.table(self.supabase_table_name)
                    .upsert(supabase_data_obj)
                    .execute()
                )
            elif "error" in response_obj:
                if "Unable to map your input to a model." in response_obj["error"]:
                    total_cost = 0
                supabase_data_obj = {
                    "response_time": response_time,
                    "model": response_obj["model"],
                    "total_cost": total_cost,
                    "messages": messages,
                    "error": response_obj["error"],
                    "end_user": end_user,
                    "litellm_call_id": litellm_call_id,
                    "status": "failure",
                }
                print_verbose(
                    f"Supabase Logging - final data object: {supabase_data_obj}"
                )
                data, count = (
                    self.supabase_client.table(self.supabase_table_name)
                    .upsert(supabase_data_obj)
                    .execute()
                )

        except:
            # traceback.print_exc()
            print_verbose(f"Supabase Logging Error - {traceback.format_exc()}")
            pass
