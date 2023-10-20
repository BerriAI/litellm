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
        "completion": usage["completion_tokens"] if "completion_tokens" in usage else 0,
        "prompt": usage["prompt_tokens"] if "prompt_tokens" in usage else 0,
    }


def parse_messages(input):
    if input is None:
        return None

    def clean_message(message):
        # if is strin, return as is
        if isinstance(message, str):
            return message

        if "message" in message:
            return clean_message(message["message"])
        text = message["content"]
        if text == None:
            text = message.get("function_call", None)

        return {
            "role": message["role"],
            "text": text,
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
        self.api_url = os.getenv("LLMONITOR_API_URL") or "https://app.llmonitor.com"
        self.app_id = os.getenv("LLMONITOR_APP_ID")

    def log_event(
        self,
        type,
        event,
        run_id,
        model,
        print_verbose,
        input=None,
        user_id=None,
        response_obj=None,
        start_time=datetime.datetime.now(),
        end_time=datetime.datetime.now(),
        error=None,
    ):
        # Method definition
        try:
            print_verbose(f"LLMonitor Logging - Logging request for model {model}")

            if response_obj:
                usage = (
                    parse_usage(response_obj["usage"])
                    if "usage" in response_obj
                    else None
                )
                output = response_obj["choices"] if "choices" in response_obj else None
            else:
                usage = None
                output = None

            if error:
                error_obj = {"stack": error}

            else:
                error_obj = None

            data = [
                {
                    "type": type,
                    "name": model,
                    "runId": run_id,
                    "app": self.app_id,
                    "event": "start",
                    "timestamp": start_time.isoformat(),
                    "userId": user_id,
                    "input": parse_messages(input),
                },
                {
                    "type": type,
                    "runId": run_id,
                    "app": self.app_id,
                    "event": event,
                    "error": error_obj,
                    "timestamp": end_time.isoformat(),
                    "userId": user_id,
                    "output": parse_messages(output),
                    "tokensUsage": usage,
                },
            ]

            print_verbose(f"LLMonitor Logging - final data object: {data}")

            response = requests.post(
                self.api_url + "/api/report",
                headers={"Content-Type": "application/json"},
                json={"events": data},
            )

            print_verbose(f"LLMonitor Logging - response: {response}")
        except:
            # traceback.print_exc()
            print_verbose(f"LLMonitor Logging Error - {traceback.format_exc()}")
            pass
