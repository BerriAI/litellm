#### What this does ####
#    On success + failure, log events to aispend.io
import datetime
import traceback
import dotenv
import os
import requests
dotenv.load_dotenv()  # Loading env variables using dotenv


class LLMonitorLogger:
    # Class variables or attributes
    def __init__(self):
        # Instance variables
        self.api_url = os.getenv(
            "LLMONITOR_API_URL") or "https://app.llmonitor.com"
        self.account_id = os.getenv("LLMONITOR_APP_ID")

    def log_event(self, model, messages, response_obj, start_time, end_time, print_verbose):
        # Method definition
        try:
            print_verbose(
                f"LLMonitor Logging - Enters logging function for model {model}")

            print(model, messages, response_obj, start_time, end_time)

            # headers = {
            #     'Content-Type': 'application/json'
            # }

            # prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar = self.price_calculator(
            #     model, response_obj, start_time, end_time)
            # total_cost = prompt_tokens_cost_usd_dollar + completion_tokens_cost_usd_dollar

            # response_time = (end_time-start_time).total_seconds()
            # if "response" in response_obj:
            #     data = [{
            #         "response_time": response_time,
            #         "model_id": response_obj["model"],
            #         "total_cost": total_cost,
            #         "messages": messages,
            #         "response": response_obj['choices'][0]['message']['content'],
            #         "account_id": self.account_id
            #     }]
            # elif "error" in response_obj:
            #     data = [{
            #         "response_time": response_time,
            #         "model_id": response_obj["model"],
            #         "total_cost": total_cost,
            #         "messages": messages,
            #         "error": response_obj['error'],
            #         "account_id": self.account_id
            #     }]

            # print_verbose(f"BerriSpend Logging - final data object: {data}")
            # response = requests.post(url, headers=headers, json=data)
        except:
            # traceback.print_exc()
            print_verbose(
                f"LLMonitor Logging Error - {traceback.format_exc()}")
            pass
