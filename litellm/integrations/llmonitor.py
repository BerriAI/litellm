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
        self.app_id = os.getenv("LLMONITOR_APP_ID")

    def log_event(self, type, run_id, error, usage, model, messages,
                  response_obj, user_id, time, print_verbose):
        # Method definition
        try:
            print_verbose(
                f"LLMonitor Logging - Enters logging function for model {model}"
            )

            print(type, model, messages, response_obj, time, end_user)

            headers = {'Content-Type': 'application/json'}

            data = {
                "type": "llm",
                "name": model,
                "runId": run_id,
                "app": self.app_id,
                "error": error,
                "event": type,
                "timestamp": time.isoformat(),
                "userId": user_id,
                "input": messages,
                "output": response_obj['choices'][0]['message']['content'],
            }

            print_verbose(f"LLMonitor Logging - final data object: {data}")
            # response = requests.post(url, headers=headers, json=data)
        except:
            # traceback.print_exc()
            print_verbose(
                f"LLMonitor Logging Error - {traceback.format_exc()}")
            pass
