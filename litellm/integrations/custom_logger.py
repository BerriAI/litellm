#### What this does ####
#    On success, logs events to Promptlayer
import dotenv, os
import requests
import requests

dotenv.load_dotenv()  # Loading env variables using dotenv
import traceback


class CustomLogger:
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


    #### DEPRECATED ####

    def log_input_event(self, model, messages, kwargs, print_verbose, callback_func):
        try: 
            print_verbose(
                    f"Custom Logger - Enters logging function for model {kwargs}"
                )
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
            print_verbose(
                f"Custom Logger - Enters logging function for model {kwargs}"
            )
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
