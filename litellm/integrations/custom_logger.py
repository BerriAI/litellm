#### What this does ####
#    On success, logs events to Promptlayer
import dotenv, os
import requests
import requests

dotenv.load_dotenv()  # Loading env variables using dotenv
import traceback


class CustomLogger: # https://docs.litellm.ai/docs/observability/custom_callback#callback-class
    # Class variables or attributes
    def __init__(self):
        pass

    def log_pre_api_call(self, model, messages, kwargs): 
        pass

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time): 
        pass
    
    def log_stream_event(self, kwargs, response_obj, start_time, end_time):
        pass

    def log_success_event(self, kwargs, response_obj, start_time, end_time): 
        pass

    def log_failure_event(self, kwargs, response_obj, start_time, end_time): 
        pass

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        pass

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        pass

    #### SINGLE-USE #### - https://docs.litellm.ai/docs/observability/custom_callback#using-your-custom-callback-function

    def log_input_event(self, model, messages, kwargs, print_verbose, callback_func):
        try: 
            kwargs["model"] = model
            kwargs["messages"] = messages
            kwargs["log_event_type"] = "pre_api_call"
            callback_func(
                kwargs,
            )
            print_verbose(
                f"Custom Logger - model call details: {kwargs}"
            )
        except: 
            traceback.print_exc()
            print_verbose(f"Custom Logger Error - {traceback.format_exc()}")

    def log_event(self, kwargs, response_obj, start_time, end_time, print_verbose, callback_func):
        # Method definition
        try:
            kwargs["log_event_type"] = "post_api_call"
            callback_func(
                kwargs, # kwargs to func
                response_obj,
                start_time,
                end_time,
            )
            print_verbose(
                f"Custom Logger - final response object: {response_obj}"
            )
        except:
            # traceback.print_exc()
            print_verbose(f"Custom Logger Error - {traceback.format_exc()}")
            pass
    
    async def async_log_event(self, kwargs, response_obj, start_time, end_time, print_verbose, callback_func):
        # Method definition
        try:
            kwargs["log_event_type"] = "post_api_call"
            await callback_func(
                kwargs, # kwargs to func
                response_obj,
                start_time,
                end_time,
            )
            print_verbose(
                f"Custom Logger - final response object: {response_obj}"
            )
        except:
            # traceback.print_exc()
            print_verbose(f"Custom Logger Error - {traceback.format_exc()}")
            pass