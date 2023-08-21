#### What this does ####
#    On success + failure, log events to aispend.io
import datetime
import traceback
import dotenv
import os
import requests

dotenv.load_dotenv()  # Loading env variables using dotenv


# convert to {completion: xx, tokens: xx}
def parse_usage(usage):
    return {
        "completion": usage["completion_tokens"],
        "prompt": usage["prompt_tokens"],
    }


def parse_messages(input):

    if input is None:
        return None

    def clean_message(message):
        if "message" in message:
            return clean_message(message["message"])

        return {
            "role": message["role"],
            "text": message["content"],
        }

    if isinstance(input, list):
        if len(input) == 1:
            return clean_message(input[0])
        else:
            return [clean_message(msg) for msg in input]
    else:
        return clean_message(input)


class LLMonitorLogger:
    # Class variables or attributes
    def __init__(self):
        # Instance variables
        self.api_url = os.getenv(
            "LLMONITOR_API_URL") or "https://app.llmonitor.com"
        self.app_id = os.getenv("LLMONITOR_APP_ID")

    def log_event(
            self,
            type,
            run_id,
            model,
            print_verbose,
            messages=None,
            user_id=None,
            response_obj=None,
            time=datetime.datetime.now(),
            error=None,
    ):
        # Method definition
        try:
            print_verbose(
                f"LLMonitor Logging - Enters logging function for model {model}"
            )

            if response_obj:
                usage = parse_usage(response_obj['usage'])
                output = response_obj['choices']
            else:
                usage = None
                output = None

            print(type, run_id, model, messages, usage, output, time, user_id,
                  error)

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
                "input": parse_messages(messages),
                "usage": usage,
                "output": parse_messages(output),
            }

            print_verbose(f"LLMonitor Logging - final data object: {data}")
            # response = requests.post(url, headers=headers, json=data)
        except:
            # traceback.print_exc()
            print_verbose(
                f"LLMonitor Logging Error - {traceback.format_exc()}")
            pass
