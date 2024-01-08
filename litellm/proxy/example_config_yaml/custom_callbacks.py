import sys, os, traceback

# this file is to test litellm/proxy

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from litellm.integrations.custom_logger import CustomLogger
import litellm
import inspect


# This file includes the custom callbacks for LiteLLM Proxy
# Once defined, these can be passed in proxy_config.yaml
def print_verbose(print_statement):
    if litellm.set_verbose:
        print(print_statement)  # noqa


class MyCustomHandler(CustomLogger):
    def __init__(self):
        blue_color_code = "\033[94m"
        reset_color_code = "\033[0m"
        print_verbose(f"{blue_color_code}Initialized LiteLLM custom logger")
        try:
            print_verbose(f"Logger Initialized with following methods:")
            methods = [
                method
                for method in dir(self)
                if inspect.ismethod(getattr(self, method))
            ]

            # Pretty print_verbose the methods
            for method in methods:
                print_verbose(f" - {method}")
            print_verbose(f"{reset_color_code}")
        except:
            pass

    def log_pre_api_call(self, model, messages, kwargs):
        print_verbose(f"Pre-API Call")

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
        print_verbose(f"Post-API Call")

    def log_stream_event(self, kwargs, response_obj, start_time, end_time):
        print_verbose(f"On Stream")

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        print_verbose("On Success!")

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print_verbose(f"On Async Success!")
        response_cost = litellm.completion_cost(completion_response=response_obj)
        assert response_cost > 0.0
        return

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            print_verbose(f"On Async Failure !")
        except Exception as e:
            print_verbose(f"Exception: {e}")


proxy_handler_instance = MyCustomHandler()


# need to set litellm.callbacks = [customHandler] # on the proxy

# litellm.success_callback = [async_on_succes_logger]
