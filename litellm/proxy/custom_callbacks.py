from litellm.integrations.custom_logger import CustomLogger
import litellm


# This file includes the custom callbacks for LiteLLM Proxy
# Once defined, these can be passed in proxy_config.yaml
class MyCustomHandler(CustomLogger):
    def log_pre_api_call(self, model, messages, kwargs):
        print(f"Pre-API Call")  # noqa

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
        print(f"Post-API Call")  # noqa

    def log_stream_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Stream")  # noqa

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        print("On Success")  # noqa

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Failure")  # noqa

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print(f"ishaan async_log_success_event")  # noqa
        # log: key, user, model, prompt, response, tokens, cost
        # Access kwargs passed to litellm.completion()
        model = kwargs.get("model", None)
        messages = kwargs.get("messages", None)
        user = kwargs.get("user", None)

        # Access litellm_params passed to litellm.completion(), example access `metadata`
        litellm_params = kwargs.get("litellm_params", {})
        metadata = litellm_params.get(
            "metadata", {}
        )  # headers passed to LiteLLM proxy, can be found here

        return

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            print(f"On Async Failure !")  # noqa
            print("\nkwargs", kwargs)  # noqa
            # Access kwargs passed to litellm.completion()
            model = kwargs.get("model", None)
            messages = kwargs.get("messages", None)
            user = kwargs.get("user", None)

            # Access litellm_params passed to litellm.completion(), example access `metadata`
            litellm_params = kwargs.get("litellm_params", {})
            metadata = litellm_params.get(
                "metadata", {}
            )  # headers passed to LiteLLM proxy, can be found here

            # Acess Exceptions & Traceback
            exception_event = kwargs.get("exception", None)
            traceback_event = kwargs.get("traceback_exception", None)

            # Calculate cost using  litellm.completion_cost()
        except Exception as e:
            print(f"Exception: {e}")  # noqa


proxy_handler_instance = MyCustomHandler()

# Set litellm.callbacks = [proxy_handler_instance] on the proxy
# need to set litellm.callbacks = [proxy_handler_instance] # on the proxy
