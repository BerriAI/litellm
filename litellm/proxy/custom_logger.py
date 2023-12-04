from litellm.integrations.custom_logger import CustomLogger
import litellm
class MyCustomHandler(CustomLogger):
    def log_pre_api_call(self, model, messages, kwargs): 
        print(f"Pre-API Call")
    
    def log_post_api_call(self, kwargs, response_obj, start_time, end_time): 
        # log: key, user, model, prompt, response, tokens, cost
        print(f"Post-API Call")
        print("\n kwargs\n")
        print(kwargs)
        model = kwargs["model"]
        messages = kwargs["messages"]
        cost = litellm.completion_cost(completion_response=response_obj)

        # tokens used in response 
        usage = response_obj.usage
        print(usage)

    
    def log_stream_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Stream")
        
    def log_success_event(self, kwargs, response_obj, start_time, end_time): 
        print(f"On Success")

    def log_failure_event(self, kwargs, response_obj, start_time, end_time): 
        print(f"On Failure")

proxy_handler_instance = MyCustomHandler()

# need to set litellm.callbacks = [customHandler] # on the proxy
