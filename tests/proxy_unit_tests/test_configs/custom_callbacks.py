from litellm.integrations.custom_logger import CustomLogger
import inspect
import litellm


class testCustomCallbackProxy(CustomLogger):
    def __init__(self):
        self.success: bool = False  # type: ignore
        self.failure: bool = False  # type: ignore
        self.async_success: bool = False  # type: ignore
        self.async_success_embedding: bool = False  # type: ignore
        self.async_failure: bool = False  # type: ignore
        self.async_failure_embedding: bool = False  # type: ignore

        self.async_completion_kwargs = None  # type: ignore
        self.async_embedding_kwargs = None  # type: ignore
        self.async_embedding_response = None  # type: ignore

        self.async_completion_kwargs_fail = None  # type: ignore
        self.async_embedding_kwargs_fail = None  # type: ignore

        self.streaming_response_obj = None  # type: ignore
        blue_color_code = "\033[94m"
        reset_color_code = "\033[0m"
        print(f"{blue_color_code}Initialized LiteLLM custom logger")
        try:
            print(f"Logger Initialized with following methods:")
            methods = [
                method
                for method in dir(self)
                if inspect.ismethod(getattr(self, method))
            ]

            # Pretty print the methods
            for method in methods:
                print(f" - {method}")
            print(f"{reset_color_code}")
        except Exception:
            pass

    def log_pre_api_call(self, model, messages, kwargs):
        print(f"Pre-API Call")

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
        print(f"Post-API Call")

    def log_stream_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Stream")

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Success")
        self.success = True

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Failure")
        self.failure = True

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Async success")
        self.async_success = True
        print("Value of async success: ", self.async_success)
        print("\n kwargs: ", kwargs)
        if (
            kwargs.get("model") == "text-embedding-ada-002"
            or kwargs.get("model") == "ada"
        ):
            print("Got an embedding model", kwargs.get("model"))
            print("Setting embedding success to True")
            self.async_success_embedding = True
            print("Value of async success embedding: ", self.async_success_embedding)
            self.async_embedding_kwargs = kwargs
            self.async_embedding_response = response_obj
        if kwargs.get("stream") == True:
            self.streaming_response_obj = response_obj

        self.async_completion_kwargs = kwargs

        model = kwargs.get("model", None)
        messages = kwargs.get("messages", None)
        user = kwargs.get("user", None)

        # Access litellm_params passed to litellm.completion(), example access `metadata`
        litellm_params = kwargs.get("litellm_params", {})
        metadata = litellm_params.get(
            "metadata", {}
        )  # headers passed to LiteLLM proxy, can be found here

        # Calculate cost using  litellm.completion_cost()
        cost = litellm.completion_cost(completion_response=response_obj)
        response = response_obj
        # tokens used in response
        usage = response_obj["usage"]

        print("\n\n in custom callback vars my custom logger, ", vars(my_custom_logger))

        print(
            f"""
                Model: {model},
                Messages: {messages},
                User: {user},
                Usage: {usage},
                Cost: {cost},
                Response: {response}
                Proxy Metadata: {metadata}
            """
        )
        return

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Async Failure")
        self.async_failure = True
        print("Value of async failure: ", self.async_failure)
        print("\n kwargs: ", kwargs)
        if kwargs.get("model") == "text-embedding-ada-002":
            self.async_failure_embedding = True
            self.async_embedding_kwargs_fail = kwargs

        self.async_completion_kwargs_fail = kwargs


# Global instance that can be accessed from tests
my_custom_logger = testCustomCallbackProxy()
