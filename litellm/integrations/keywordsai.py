from litellm.utils import ModelResponse
from litellm import stream_chunk_builder, print_verbose
from litellm.integrations.custom_logger import CustomLogger
import requests
import os
import json
import datetime

class BgColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

KEYWORDS_AI_API_KEY = os.getenv("KEYWORDS_AI_API_KEY")
def setup_keywordsai_params(model, messages, kwargs):
    url = "https://api.keywordsai.co/api/request-logs/create/"

    if not KEYWORDS_AI_API_KEY:
        print(f"{BgColors.WARNING}!!!!WARNING: KEYWORDS_AI_API_KEY not found in environment!!!!{BgColors.ENDC}")
    headers = {
        "Authorization": f"Bearer {KEYWORDS_AI_API_KEY}",
        "Content-Type": "application/json",
    }

    keywordsai_params = kwargs.get("extra_body", {}).pop("keywordsai_params", {})
    kwargs["keywordsai_url"] = url
    kwargs["keywordsai_headers"] = headers
    kwargs["keywordsai_stream_collector"] = []
    kwargs["keywordsai_payload"] = {
        "model": kwargs.get("model"),
        "prompt_messages": kwargs.get("messages"),
        "tool_choice": kwargs.get("tool_choice"),
        "tools": kwargs.get("tools"),
        "customer_identifier": keywordsai_params.get("customer_identifier", ""),
        "thread_identifier": keywordsai_params.get("thread_identifier", ""),
        "metadata": keywordsai_params.get("metadata"),
        "stream": kwargs.get("stream", False),
    }


def modify_params(
    kwargs, response_obj, start_time: datetime.datetime, end_time, status_code=200
):
    stream = kwargs.get("stream", False)
    if stream:
        if (
            "time_to_first_token" not in kwargs["keywordsai_payload"]
            and end_time is not None
        ):
            time_to_first_token = (end_time - start_time).total_seconds()
            kwargs["keywordsai_payload"]["time_to_first_token"] = time_to_first_token

        if isinstance(response_obj, ModelResponse):
            if stream:
                kwargs["keywordsai_stream_collector"].append(response_obj)
            else:
                kwargs["keywordsai_payload"][
                    "full_response"
                ] = response_obj.model_dump()
    else:
        original_response = kwargs.get("original_response")
        if isinstance(original_response, str):
            kwargs["keywordsai_payload"]["full_response"] = json.loads(
                original_response
            )
        elif isinstance(original_response, ModelResponse):
            kwargs["keywordsai_payload"][
                "full_response"
            ] = original_response.model_dump()


def commit_to_keywordsai(kwargs, start_time, end_time, success=True):
    url = kwargs.get("keywordsai_url")
    headers = kwargs.get("keywordsai_headers")
    payload = kwargs.get("keywordsai_payload")
    stream = kwargs.get("stream", False)
    if success:
        payload["status_code"] = 200
    else:
        payload["status_code"] = 400

    if stream:
        full_response = stream_chunk_builder(
            kwargs.get("keywordsai_stream_collector")
        ).model_dump()
    else:
        full_response = payload["full_response"]

    payload["completion_message"] = full_response["choices"][0]["message"]
    payload["tool_calls"] = full_response["choices"][0].get("tool_calls")
    payload["latency"] = (datetime.datetime.now() - start_time).total_seconds()
    if "committed" not in payload:
        response = requests.request("POST", url, headers=headers, json=payload)
        payload["comitted"] = True
        print_verbose(f"Keywords AI Callback: committed to KeywordsAI: {response.status_code}")


class KeywordsAILogger(CustomLogger):
    def log_pre_api_call(self, model, messages, kwargs):
        print_verbose(f"Keywords AI Callback: Pre-API Call")
        setup_keywordsai_params(model, messages, kwargs)

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
        print_verbose(f"Keywords AI Callback: Post-API Call")
        if kwargs.get("stream", False): # Due to async nature, pre_api_call is not guaranteed to be called before post_api_call
            setup_keywordsai_params(kwargs.get("model"), kwargs.get("messages"), kwargs)
        modify_params(kwargs, response_obj, start_time, end_time)

    def log_stream_event(self, kwargs, response_obj, start_time, end_time):
        pass

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        print_verbose(f"Keywords AI Callback: On Success")
        stream = kwargs.get("stream", False)
        if stream:
            return  # Async streaming will handle this
        commit_to_keywordsai(kwargs, start_time, end_time, success=True)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        print_verbose(f"Keywords AI Callback: On Failure")
        # modify_params(kwargs, response_obj, start_time, end_time, status_code=400)
        stream = kwargs.get("stream", False)
        if stream:
            return  # Async streaming will handle this
        commit_to_keywordsai(kwargs, start_time, end_time, success=False)

    #### ASYNC #### - for acompletion/aembeddings

    async def async_log_stream_event(self, kwargs, response_obj, start_time, end_time):
        print_verbose(f"Keywords AI Callback: On Async Streaming")
        modify_params(kwargs, response_obj, start_time, end_time)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print_verbose(f"Keywords AI Callback: On Async Success")
        commit_to_keywordsai(kwargs, start_time, end_time, success=True)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        print_verbose(f"Keywords AI Callback: On Async Failure")
        commit_to_keywordsai(kwargs, start_time, end_time, success=False)
