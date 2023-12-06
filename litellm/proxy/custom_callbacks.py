from litellm.integrations.custom_logger import CustomLogger
import litellm

# This file includes the custom callbacks for LiteLLM Proxy
# Once defined, these can be passed in proxy_config.yaml
class MyCustomHandler(CustomLogger):
    def log_pre_api_call(self, model, messages, kwargs): 
        print(f"Pre-API Call")
    
    def log_post_api_call(self, kwargs, response_obj, start_time, end_time): 
        print(f"Post-API Call")

    def log_stream_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Stream")
        
    def log_success_event(self, kwargs, response_obj, start_time, end_time): 
        # log: key, user, model, prompt, response, tokens, cost
        print("\nOn Success")
        ### Access kwargs passed to litellm.completion()
        model = kwargs.get("model", None)
        messages = kwargs.get("messages", None)
        user = kwargs.get("user", None)

        #### Access litellm_params passed to litellm.completion(), example access `metadata`
        litellm_params = kwargs.get("litellm_params", {})
        metadata = litellm_params.get("metadata", {})   # headers passed to LiteLLM proxy, can be found here
        #################################################

        ##### Calculate cost using  litellm.completion_cost() #######################
        cost = litellm.completion_cost(completion_response=response_obj)
        response = response_obj
        # tokens used in response 
        usage = response_obj["usage"]

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

    def log_failure_event(self, kwargs, response_obj, start_time, end_time): 
        print(f"On Failure")

proxy_handler_instance = MyCustomHandler()

# need to set litellm.callbacks = [customHandler] # on the proxy

## setting only one function 
async def async_on_succes_logger(kwargs, response_obj, start_time, end_time):
    print(f"On Async Success!")
    # log: key, user, model, prompt, response, tokens, cost
    print("\nOn Success")
    # Access kwargs passed to litellm.completion()
    model = kwargs.get("model", None)
    messages = kwargs.get("messages", None)
    user = kwargs.get("user", None)

    # Access litellm_params passed to litellm.completion(), example access `metadata`
    litellm_params = kwargs.get("litellm_params", {})
    metadata = litellm_params.get("metadata", {})   # headers passed to LiteLLM proxy, can be found here

    # Calculate cost using  litellm.completion_cost()
    cost = litellm.completion_cost(completion_response=response_obj)
    response = response_obj
    # tokens used in response 
    usage = response_obj["usage"]

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


async def async_on_fail_logger(kwargs, response_obj, start_time, end_time):
    print(f"On Async Failure!")
    print(kwargs)

    # Access kwargs passed to litellm.completion()
    model = kwargs.get("model", None)
    messages = kwargs.get("messages", None)
    user = kwargs.get("user", None)

    # Access litellm_params passed to litellm.completion(), example access `metadata`
    litellm_params = kwargs.get("litellm_params", {})
    metadata = litellm_params.get("metadata", {})   # headers passed to LiteLLM proxy, can be found here

    # Acess Exceptions & Traceback
    exception_event = kwargs.get("exception", None)
    traceback_event = kwargs.get("traceback_exception", None)

    # Calculate cost using  litellm.completion_cost()
    cost = litellm.completion_cost(completion_response=response_obj)
    response = response_obj
    # tokens used in response 
    usage = response_obj.get("usage", {})

    print(
        f"""
            Model: {model},
            Messages: {messages},
            User: {user},
            Usage: {usage},
            Cost: {cost},
            Response: {response}
            Proxy Metadata: {metadata}
            Exception: {exception_event}
            Traceback: {traceback_event}
        """
    )

# litellm.success_callback = [async_on_succes_logger]