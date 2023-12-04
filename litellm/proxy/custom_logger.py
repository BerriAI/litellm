from litellm.integrations.custom_logger import CustomLogger
import litellm
class MyCustomHandler(CustomLogger):
    def log_pre_api_call(self, model, messages, kwargs): 
        print(f"Pre-API Call")
    
    def log_post_api_call(self, kwargs, response_obj, start_time, end_time): 
        print(f"Post-API Call")

    def log_stream_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Stream")
        
    def log_success_event(self, kwargs, response_obj, start_time, end_time): 
        print(f"On Success")
        # log: key, user, model, prompt, response, tokens, cost
        print("\n kwargs\n")
        print(kwargs)
        ### Access kwargs passed to litellm.completion()
        model = kwargs["model"]
        messages = kwargs["messages"]
        user = kwargs.get("user")
        #################################################

        ### Calculate cost #######################
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
            """
        )

        print(usage)


    def log_failure_event(self, kwargs, response_obj, start_time, end_time): 
        print(f"On Failure")

proxy_handler_instance = MyCustomHandler()

# need to set litellm.callbacks = [customHandler] # on the proxy
