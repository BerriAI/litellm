from litellm.integrations.custom_logger import CustomLogger


class MyCustomHandler(CustomLogger):
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        # input_tokens = response_obj.get("usage", {}).get("prompt_tokens", 0)
        # output_tokens = response_obj.get("usage", {}).get("completion_tokens", 0)
        input_tokens = (
            response_obj.usage.prompt_tokens
            if hasattr(response_obj.usage, "prompt_tokens")
            else 0
        )
        output_tokens = (
            response_obj.usage.completion_tokens
            if hasattr(response_obj.usage, "completion_tokens")
            else 0
        )


proxy_handler_instance = MyCustomHandler()
