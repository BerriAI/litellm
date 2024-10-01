#### What this does ####
#    On success + failure, log events to aispend.io
import datetime
import os
import traceback

import dotenv

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


class AISpendLogger:
    # Class variables or attributes
    def __init__(self):
        # Instance variables
        self.account_id = os.getenv("AISPEND_ACCOUNT_ID")
        self.api_key = os.getenv("AISPEND_API_KEY")

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
            prompt_tokens_cost_usd_dollar = (
                model_cost[model]["input_cost_per_token"]
                * response_obj["usage"]["prompt_tokens"]
            )
            completion_tokens_cost_usd_dollar = (
                model_cost[model]["output_cost_per_token"]
                * response_obj["usage"]["completion_tokens"]
            )
        return prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar

    def log_event(self, model, response_obj, start_time, end_time, print_verbose):
        # Method definition
        try:
            print_verbose(
                f"AISpend Logging - Enters logging function for model {model}"
            )

            response_timestamp = datetime.datetime.fromtimestamp(
                int(response_obj["created"])
            ).strftime("%Y-%m-%d")

            (
                prompt_tokens_cost_usd_dollar,
                completion_tokens_cost_usd_dollar,
            ) = self.price_calculator(model, response_obj, start_time, end_time)
            prompt_tokens_cost_usd_cent = prompt_tokens_cost_usd_dollar * 100
            completion_tokens_cost_usd_cent = completion_tokens_cost_usd_dollar * 100
            data = [
                {
                    "requests": 1,
                    "requests_context": 1,
                    "context_tokens": response_obj["usage"]["prompt_tokens"],
                    "requests_generated": 1,
                    "generated_tokens": response_obj["usage"]["completion_tokens"],
                    "recorded_date": response_timestamp,
                    "model_id": response_obj["model"],
                    "generated_tokens_cost_usd_cent": prompt_tokens_cost_usd_cent,
                    "context_tokens_cost_usd_cent": completion_tokens_cost_usd_cent,
                }
            ]

            print_verbose(f"AISpend Logging - final data object: {data}")
        except Exception:
            print_verbose(f"AISpend Logging Error - {traceback.format_exc()}")
            pass
