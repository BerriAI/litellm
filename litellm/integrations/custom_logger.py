#### What this does ####
#    On success, logs events to Promptlayer
import dotenv, os
import requests
import requests

dotenv.load_dotenv()  # Loading env variables using dotenv
import traceback


class CustomLogger:
    # Class variables or attributes
    def __init__(self, callback_func):
        # Instance variables
        self.callback_func = callback_func

    def log_event(self, kwargs, response_obj, start_time, end_time, print_verbose):
        # Method definition
        try:
            print_verbose(
                f"Custom Logger - Enters logging function for model {kwargs}"
            )
            self.callback_func(
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
