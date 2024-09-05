from litellm.integrations.custom_logger import CustomLogger


class MyCustomHandler(CustomLogger):
    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        # print("Call failed")
        pass


proxy_handler_instance = MyCustomHandler()
