#### What this does ####
#    On success + failure, log events to aispend.io
import datetime
import os
import traceback

import dotenv
import requests  # type: ignore

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


class BerriSpendLogger:
    # Class variables or attributes
    def __init__(self):
        # Instance variables
        self.account_id = os.getenv("BERRISPEND_ACCOUNT_ID")

    def price_calculator(self, model, response_obj, start_time, end_time):
        return

    def log_event(
        self, model, messages, response_obj, start_time, end_time, print_verbose
    ):
        """
        This integration is not implemented yet.
        """
        return
