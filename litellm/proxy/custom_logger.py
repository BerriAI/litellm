from litellm.integrations.custom_logger import CustomLogger
class MyCustomHandler(CustomLogger):
    def log_pre_api_call(self, model, messages, kwargs): 
        print(f"Pre-API Call")
    
    def log_post_api_call(self, kwargs, response_obj, start_time, end_time): 
        # log: key, user, model, prompt, response, tokens, cost
        print(f"Post-API Call")
    
    def log_stream_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Stream")
        
    def log_success_event(self, kwargs, response_obj, start_time, end_time): 
        print(f"On Success")

    def log_failure_event(self, kwargs, response_obj, start_time, end_time): 
        print(f"On Failure")

customHandler = MyCustomHandler()
